import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from model import StudentBehaviorGenerator, DualDiscriminator, PRLModel, GANLosses

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KTCSVDataset(Dataset):
    def __init__(self, csv_file, top_m=64, window_w=5, lambda_saliency=0.5, 
                 c_vocab=None, uid_map=None, fold_id=None, mode='all'):
        self.df = pd.read_csv(csv_file)
        self.top_m = top_m
        self.window_w = window_w
        self.lambda_saliency = lambda_saliency
        
        self.samples = []
        
        self.c_vocab = c_vocab if c_vocab else {}
        build_vocab = c_vocab is None
        self.uid_map = uid_map if uid_map else {}
        build_uid = uid_map is None
        self.fold_id = fold_id
        self.mode = mode
        
        self._preprocess(build_vocab, build_uid)
        
    def _get_saliency_scores(self, r_seq):
        seq_len = len(r_seq)
        scores = []
        w = self.window_w
        lam = self.lambda_saliency
        
        for t in range(seq_len):
            start = max(0, t - w // 2)
            end = min(seq_len, t + w // 2 + 1)
            window_r = r_seq[start:end]
            
            if len(window_r) == 0:
                scores.append(0)
                continue
                
            p_t = sum(window_r) / len(window_r)
            p_t_clipped = max(min(p_t, 1 - 1e-6), 1e-6)
            h_t = -p_t_clipped * np.log(p_t_clipped) - (1 - p_t_clipped) * np.log(1 - p_t_clipped)
            
            flips = 0
            if len(window_r) > 1:
                for i in range(len(window_r) - 1):
                    if window_r[i] != window_r[i+1]:
                        flips += 1
                flip_t = flips / len(window_r)
            else:
                flip_t = 0
                
            s_t = lam * h_t + (1 - lam) * flip_t
            scores.append(s_t)
            
        return scores

    def _preprocess(self, build_vocab, build_uid):
        if build_vocab:
            unique_cs = set()
            for c_str in self.df['concepts']:
                c_str = str(c_str).replace('[', '').replace(']', '')
                cs = c_str.split(',')
                for c in cs:
                    if c.strip() and c.strip() != '-1': unique_cs.add(int(c.strip()))
            sorted_cs = sorted(list(unique_cs))
            self.c_vocab = {c: i+1 for i, c in enumerate(sorted_cs)}

        if 'uid' not in self.df.columns: 
            self.df['uid'] = self.df.index

        if build_uid:
            unique_uids = self.df['uid'].unique()
            self.uid_map = {uid: i for i, uid in enumerate(unique_uids)}
        
        self.num_students = len(self.uid_map)

        for _, row in self.df.iterrows():
            if self.fold_id is not None and 'fold' in row:
                curr_fold = int(row['fold'])
                if self.mode == 'train' and curr_fold == self.fold_id: continue
                elif self.mode == 'valid' and curr_fold != self.fold_id: continue
            
            c_str = str(row['concepts']).replace('[', '').replace(']', '')
            r_str = str(row['responses']).replace('[', '').replace(']', '')
            
            if 'timestamps' in row:
                t_str = str(row['timestamps']).replace('[', '').replace(']', '')
                t_raw = [float(x) for x in t_str.split(',') if x.strip()]
            else:
                t_raw = [0.0] * len(c_str.split(','))

            c_raw = [int(x) for x in c_str.split(',') if x.strip()]
            r_raw = [int(x) for x in r_str.split(',') if x.strip()]
            
            c_seq_full, r_seq_full, t_seq_full = [], [], []
            for i in range(min(len(c_raw), len(r_raw), len(t_raw))):
                c, r, t = c_raw[i], r_raw[i], t_raw[i]
                if c != -1 and r != -1:
                    c_seq_full.append(c)
                    r_seq_full.append(r)
                    t_seq_full.append(t)
            
            if len(c_seq_full) < 2: continue
            
            if len(c_seq_full) > self.top_m:
                s_scores = self._get_saliency_scores(r_seq_full)
                top_indices = np.argsort(s_scores)[-self.top_m:]
                top_indices = sorted(top_indices)
                
                c_seq_trunc = [c_seq_full[i] for i in top_indices]
                r_seq_trunc = [r_seq_full[i] for i in top_indices]
                t_seq_trunc = [t_seq_full[i] for i in top_indices]
            else:
                c_seq_trunc = c_seq_full
                r_seq_trunc = r_seq_full
                t_seq_trunc = t_seq_full
            
            t_intervals = []
            prev_t = t_seq_trunc[0]
            for t_val in t_seq_trunc:
                delta = max(0, t_val - prev_t)
                delta_min = delta / (1000.0 * 60.0) 
                t_intervals.append(np.log(delta_min + 1.0))
                prev_t = t_val
                
            c_ids = [self.c_vocab.get(c, 0) for c in c_seq_trunc]
            inp_c = c_ids
            inp_r = [0] + [r + 1 for r in r_seq_trunc[:-1]]
            inp_t = [0.0] + t_intervals[:-1]
            
            target_y = r_seq_trunc
            curr_len = len(inp_c)
            pad_len = self.top_m - curr_len
            
            raw_uid = row['uid']
            stu_idx = self.uid_map.get(raw_uid, 0) 
            
            real_seq_gan = torch.tensor([0]*pad_len + r_seq_trunc, dtype=torch.float).unsqueeze(0)
            kt_c = torch.tensor([0]*pad_len + inp_c, dtype=torch.long)
            kt_r = torch.tensor([0]*pad_len + inp_r, dtype=torch.long)
            kt_t = torch.tensor([0.0]*pad_len + inp_t, dtype=torch.float) 
            kt_y = torch.tensor([0]*pad_len + target_y, dtype=torch.float)
            kt_mask = torch.tensor([0]*pad_len + [1]*curr_len, dtype=torch.float)
            q_ids_gan = kt_c
            times_gan = torch.rand(self.top_m, 1)

            self.samples.append({
                'stu_idx': torch.tensor(stu_idx, dtype=torch.long),
                'q_ids': q_ids_gan, 'times': times_gan, 'real_seq_gan': real_seq_gan,
                'kt_c': kt_c, 'kt_r': kt_r, 'kt_t': kt_t, 'kt_y': kt_y, 'kt_mask': kt_mask,
                'seq_len': curr_len
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def evaluate_kt_with_gcs(model, gan_gen, loader, device, prev_structures=None, gcs_top_k=5):
    model.eval()
    gan_gen.eval()
    all_y, all_pred = [], []
    last_y, last_pred = [], []
    current_structures = []
    total_gcs_sum = 0
    total_users_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            c = batch['kt_c'].to(device)
            r = batch['kt_r'].to(device)
            t = batch['kt_t'].to(device)
            y = batch['kt_y'].to(device)
            mask = batch['kt_mask'].to(device)
            real_lens = batch['seq_len']
            
            _, u_i, A_hat = gan_gen(history_q=c, history_r=r, history_t=t, history_mask=mask)
            
            _, topk_indices = torch.topk(A_hat, k=gcs_top_k, dim=-1)
            current_batch_indices = topk_indices.cpu().numpy()
            if current_batch_indices.ndim == 2:
                B, k = current_batch_indices.shape
                m = A_hat.shape[1]
                current_batch_indices = np.repeat(current_batch_indices[:, np.newaxis, :], m, axis=1)
            current_structures.append(current_batch_indices)
            
            if prev_structures is not None and batch_idx < len(prev_structures):
                prev_batch_indices = prev_structures[batch_idx]
                for b in range(len(current_batch_indices)):
                    user_gcs_sum = 0
                    num_nodes = current_batch_indices.shape[1]
                    curr_node_indices = current_batch_indices[b]
                    prev_node_indices = prev_batch_indices[b]
                    
                    for node_idx in range(num_nodes):
                        curr_neighbors = curr_node_indices[node_idx]
                        prev_neighbors = prev_node_indices[node_idx]
                        if not hasattr(curr_neighbors, '__iter__') or isinstance(curr_neighbors, (np.integer, int)):
                            curr_neighbors = [int(curr_neighbors)]
                        if not hasattr(prev_neighbors, '__iter__') or isinstance(prev_neighbors, (np.integer, int)):
                            prev_neighbors = [int(prev_neighbors)]
                        set_curr = set(curr_neighbors)
                        set_prev = set(prev_neighbors)
                        
                        intersection = len(set_curr.intersection(set_prev))
                        union = len(set_curr.union(set_prev))
                        if union > 0: user_gcs_sum += intersection / union
                        else: user_gcs_sum += 1.0
                    total_gcs_sum += user_gcs_sum / num_nodes
                    total_users_count += 1
            
            y_hat, _, _ = model(c=c, r=r, t=t, u_i=u_i)
            
            mask_bool = mask == 1
            all_pred.extend(y_hat[mask_bool].cpu().tolist())
            all_y.extend(y[mask_bool].cpu().tolist())
            
            for i in range(len(y)):
                if real_lens[i] > 0:
                    last_pred.append(y_hat[i, -1].item())
                    last_y.append(y[i, -1].item())

    res = {}
    if all_y:
        res['auc_all'] = roc_auc_score(all_y, all_pred)
        res['acc_all'] = accuracy_score(all_y, [1 if p>=0.5 else 0 for p in all_pred])
        res['rmse_all'] = np.sqrt(mean_squared_error(all_y, all_pred))
        
    if last_y:
        res['auc_last'] = roc_auc_score(last_y, last_pred)
        res['acc_last'] = accuracy_score(last_y, [1 if p>=0.5 else 0 for p in last_pred])
        res['rmse_last'] = np.sqrt(mean_squared_error(last_y, last_pred))
        
    if total_users_count > 0: res['gcs'] = total_gcs_sum / total_users_count
    else: res['gcs'] = 0.0
        
    return res, current_structures

def train_pipeline(data_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed_everything(42)
    
    if not os.path.exists(data_path):
        print(f"[ERROR] CSV not found at {data_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    
    print(">>> Loading Datasets...")
    
    train_ds = KTCSVDataset(data_path, top_m=64, window_w=3, 
                            lambda_saliency=0.9, fold_id=0, mode='train')
    valid_ds = KTCSVDataset(data_path, top_m=64, window_w=3,
                            lambda_saliency=0.9,
                            c_vocab=train_ds.c_vocab, uid_map=train_ds.uid_map, fold_id=0, mode='valid')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
    
    num_concepts = len(train_ds.c_vocab)
    
    gan_gen = StudentBehaviorGenerator(num_questions=num_concepts+1, dim_u=256, temperature=0.05).to(device)

    gan_disc = DualDiscriminator(seq_len=64).to(device)
    kt_model = PRLModel(n_concepts=num_concepts, dim_u=256, max_len=64, 
                        num_experts_k=8).to(device)
    
    opt_G = optim.Adam(gan_gen.parameters(), lr=1e-4)
    opt_D = optim.Adam(gan_disc.parameters(), lr=1e-4)
    opt_KT = optim.Adam(list(kt_model.parameters()) + list(gan_gen.encoder.parameters()), lr=1e-3)
    
    print(">>> Start Training Loop...")
    best_auc = -1
    prev_val_structures = None

    for epoch in range(12):
        gan_gen.train(); gan_disc.train(); kt_model.train()
        kt_losses = []
        
        for batch in train_loader:
            stu_idx = batch['stu_idx'].to(device)
            q_ids = batch['q_ids'].to(device)
            times = batch['times'].to(device)
            real_seq_gan = batch['real_seq_gan'].to(device)
            kt_c, kt_r, kt_t, kt_y, kt_mask = [batch[k].to(device) for k in ['kt_c', 'kt_r', 'kt_t', 'kt_y', 'kt_mask']]
            
            opt_D.zero_grad()
            fake_seq, _, _ = gan_gen(history_q=kt_c, history_r=kt_r, history_t=kt_t, history_mask=kt_mask, q_ids_gen=q_ids, times_gen=times)
            d_real_val, d_real_style = gan_disc(real_seq_gan)
            d_fake_val, d_fake_style = gan_disc(fake_seq.detach()) 
            
            loss_wgan = torch.mean(d_fake_val) - torch.mean(d_real_val)
            loss_gp = GANLosses.wgan_gp_gradient_penalty(gan_disc, real_seq_gan, fake_seq.detach(), device)
            loss_cont = GANLosses.wasserstein_style_contrastive_loss(d_real_style, d_fake_style, stu_idx)
            loss_D = loss_wgan + 10.0 * loss_gp + 1.0 * loss_cont
            loss_D.backward()
            opt_D.step()
            
            opt_G.zero_grad()
            fake_seq, _, A_hat = gan_gen(history_q=kt_c, history_r=kt_r, history_t=kt_t, history_mask=kt_mask, q_ids_gen=q_ids, times_gen=times)
            d_fake_val, d_fake_style = gan_disc(fake_seq)
            _, d_real_style = gan_disc(real_seq_gan) 
            
            loss_G_adv = -torch.mean(d_fake_val) + 1.0 * GANLosses.wasserstein_style_contrastive_loss(d_real_style.detach(), d_fake_style, stu_idx)
            loss_dag = torch.trace(torch.exp(A_hat * A_hat)) - (64 + 1)
            loss_sparsity = torch.mean(torch.abs(A_hat))
            loss_GE = loss_G_adv + 1.0 * loss_dag + 5.0 * loss_sparsity
            loss_GE.backward()
            opt_G.step()
            
            opt_KT.zero_grad()
            _, u_i, A_hat = gan_gen(history_q=kt_c, history_r=kt_r, history_t=kt_t, history_mask=kt_mask)
            y_hat, _, _ = kt_model(c=kt_c, r=kt_r, t=kt_t, u_i=u_i)
            
            loss_bce = F.binary_cross_entropy(y_hat, kt_y, reduction='none')
            loss_main = (loss_bce * kt_mask).sum() / (kt_mask.sum() + 1e-8)

            loss_kt = loss_main + 5.0 * torch.mean(torch.abs(A_hat))
            loss_kt.backward()
            opt_KT.step()
            kt_losses.append(loss_kt.item())

        val_metrics, curr_val_structures = evaluate_kt_with_gcs(
            kt_model, gan_gen, valid_loader, device, 
            prev_structures=prev_val_structures, 
            gcs_top_k=3
        )
        prev_val_structures = curr_val_structures
        
        print(f"Epoch {epoch+1} | KT Loss:{np.mean(kt_losses):.4f} | "
              f"Val AUC:{val_metrics['auc_last']:.4f} | "
              f"Val ACC:{val_metrics['acc_last']:.4f} | "
              f"Val RMSE:{val_metrics['rmse_last']:.4f} | "
              f"GCS:{val_metrics['gcs']:.4f}")
        
        if val_metrics['auc_last'] > best_auc:
            best_auc = val_metrics['auc_last']
            torch.save(kt_model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "kt_model_best.pth"))
            print(f"[BEST] Saved at epoch {epoch+1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/root/workcopy/datasets/data/assist2009/train_valid_sequences.csv")
    args = parser.parse_args()
    train_pipeline(args.data_path)