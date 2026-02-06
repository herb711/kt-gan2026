import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MyLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta

class Smooth(nn.Module):
    def __init__(self, dropout, hidden_size, kernel_size):
        super().__init__()
        self.out_dropout = nn.Dropout(dropout)
        self.LayerNorm = MyLayerNorm(hidden_size)
        self.causal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, 
                                     padding=(kernel_size-1), dilation=1)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        x_p = input_tensor.permute(0, 2, 1)
        trend = self.causal_conv(x_p)
        trend = trend[:, :, :-self.causal_conv.padding[0]].permute(0, 2, 1)
        random = input_tensor - trend
        return self.LayerNorm(self.out_dropout(trend + (self.sqrt_beta**2) * random) + input_tensor)

class TimeCosineEmbedding(nn.Module):
    def __init__(self, d_model, max_period=10000.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        div_term = torch.exp(torch.arange(0, half_dim, 1).float() * -(math.log(max_period) / half_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        scaled_time = x.unsqueeze(-1) * self.div_term
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        
        if self.d_model % 2 == 1:
            pe = F.pad(pe, (0, 1))
            
        return pe

class CausalUserEncoder(nn.Module):
    def __init__(self, num_questions, dim_model=64, dim_u=256, temperature=0.01):
        super().__init__()
        self.q_emb = nn.Embedding(num_questions, dim_model, padding_idx=0)
        self.r_emb = nn.Embedding(3, dim_model, padding_idx=0) 
        self.time_emb = TimeCosineEmbedding(dim_model)
        
        self.virtual_node = nn.Parameter(torch.randn(1, dim_model))
        
        self.struct_head = nn.Sequential(
            nn.Linear(dim_model * 2, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
        
        self.temperature = temperature
        
        self.aggregator = nn.Sequential(
            nn.Linear(dim_model, dim_u),
            nn.LeakyReLU(0.2)
        )

    def forward(self, q_seq, r_seq, t_seq, mask=None):
        B, L = q_seq.shape
        h_events = self.q_emb(q_seq) + self.r_emb(r_seq) + self.time_emb(t_seq)
        
        h_virtual_expanded = self.virtual_node.expand(B, L, -1)
        
        pair_features = torch.cat([h_events, h_virtual_expanded], dim=-1)
        logits = self.struct_head(pair_features) 
        
        A_hat = torch.sigmoid(logits / self.temperature)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            A_hat = A_hat * mask_expanded
            
        h_trans = self.aggregator(h_events) 
        u_i = (h_trans * A_hat).sum(dim=1)
        
        return u_i, A_hat.squeeze(-1)

class ConditionAwareAdaIN(nn.Module):
    def __init__(self, feature_dim, dim_u, q_embed_dim):
        super().__init__()
        self.feature_dim = feature_dim
        interaction_dim = dim_u * q_embed_dim
        
        self.W = nn.Linear(interaction_dim, feature_dim * 2, bias=False)
        self.V = nn.Linear(1, feature_dim * 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(feature_dim * 2))
        self.instance_norm = nn.InstanceNorm1d(feature_dim)

    def forward(self, x, u_i, e_qid, t):
        normalized_x = self.instance_norm(x)
        B, C, L = x.shape
        u_exp = u_i.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, L)
        e_exp = e_qid.unsqueeze(1)
        interaction = (u_exp * e_exp).reshape(B, -1, L)
        interaction_perm = interaction.permute(0, 2, 1)
        style_part = self.W(interaction_perm).permute(0, 2, 1)
        t_perm = t.permute(0, 2, 1)
        time_part = self.V(t_perm).permute(0, 2, 1)
        params = style_part + time_part + self.bias.view(1, -1, 1)
        gamma, beta = params.chunk(2, dim=1)
        return (1 + gamma) * normalized_x + beta

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim_u, q_embed_dim):
        super().__init__()
        self.alpha_noise = nn.Parameter(torch.tensor(0.002))
        self.adain1 = ConditionAwareAdaIN(in_channels, dim_u, q_embed_dim)
        self.attn = nn.MultiheadAttention(in_channels, 4, batch_first=True)
        self.adain2 = ConditionAwareAdaIN(in_channels, dim_u, q_embed_dim)
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, x, u_i, e_qid, t):
        x = x + self.alpha_noise * torch.randn_like(x)
        x = F.leaky_relu(self.adain1(x, u_i, e_qid, t), 0.2)
        x_perm = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x_perm, x_perm, x_perm)
        x = x + attn_out.permute(0, 2, 1)
        x = F.leaky_relu(self.adain2(x, u_i, e_qid, t), 0.2)
        return self.conv(x)

class StudentBehaviorGenerator(nn.Module):
    def __init__(self, num_questions, q_embed_dim=64, dim_u=256, temperature=0.01):
        super().__init__()
        self.encoder = CausalUserEncoder(num_questions, dim_model=q_embed_dim, dim_u=dim_u, temperature=temperature)
        self.q_embedding = nn.Embedding(num_questions, q_embed_dim)
        
        self.const_input = nn.Parameter(torch.randn(1, 512, 4)) 
        
        self.block1 = SynthesisBlock(512, 512, dim_u, q_embed_dim)
        self.block2 = SynthesisBlock(512, 256, dim_u, q_embed_dim)
        self.to_out = nn.Conv1d(256, 1, kernel_size=1)

    def forward(self, history_q, history_r, history_t, history_mask=None, q_ids_gen=None, times_gen=None):
        u_i, A_hat = self.encoder(history_q, history_r, history_t, history_mask)
        
        if q_ids_gen is None: 
            return None, u_i, A_hat
            
        B = u_i.size(0)
        e_qid_full = self.q_embedding(q_ids_gen).permute(0, 2, 1)
        times_full = times_gen.permute(0, 2, 1)
        
        x = self.const_input.repeat(B, 1, 1)
        
        curr_len = x.size(2)
        e_curr = F.interpolate(e_qid_full, size=curr_len, mode='linear', align_corners=False)
        t_curr = F.interpolate(times_full, size=curr_len, mode='linear', align_corners=False)
        x = self.block1(x, u_i, e_curr, t_curr)
        x = F.interpolate(x, scale_factor=4, mode='linear', align_corners=False) 
        
        curr_len = x.size(2)
        e_curr = F.interpolate(e_qid_full, size=curr_len, mode='linear', align_corners=False)
        t_curr = F.interpolate(times_full, size=curr_len, mode='linear', align_corners=False)
        x = self.block2(x, u_i, e_curr, t_curr)
        
        target_len = q_ids_gen.shape[1]
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        
        return self.to_out(x), u_i, A_hat

class DualDiscriminator(nn.Module):
    def __init__(self, seq_len=64, feature_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, feature_dim, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        flat_dim = feature_dim * (seq_len // 8)
        self.validity_head = nn.Linear(flat_dim, 1)
        self.style_head = nn.Linear(flat_dim, 256)

    def forward(self, x):
        feat = self.backbone(x)
        return self.validity_head(feat), self.style_head(feat)

class RATEAttention(nn.Module):
    def __init__(self, d_model, nhead, lamda_att=1.0, dropout=0.1):
        super().__init__()
        self.nhead, self.d_k = nhead, d_model // nhead
        self.lamda_att = lamda_att
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, delta_r, mask=None):
        B, L, _ = x.size()
        v_scaled = x * delta_r.unsqueeze(-1)
        q = self.q_linear(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(v_scaled).view(B, L, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        pos = torch.arange(L, device=x.device)
        dist = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        
        scores = scores - (dist * 0.1)
        rel_bias = self.lamda_att * torch.log(delta_r + 1e-6)
        scores = scores + rel_bias.view(B, 1, 1, L)
        if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        return self.out_linear(context), attn.mean(dim=1)

class RATEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout, d_ff):
        super().__init__()
        self.self_attn = RATEAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, delta_r, mask):
        attn_out, avg_attn = self.self_attn(x, delta_r, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, avg_attn

class PRLModel(nn.Module):
    def __init__(self, n_concepts, d_model=64, nhead=4, num_layers=2, 
                 alpha_ks=0.5, max_len=64, dropout=0.05, d_ff=512,
                 dim_u=256, num_experts_k=8):
        super().__init__()
        self.concept_embedding = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)
        self.response_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        self.time_embedding = TimeCosineEmbedding(d_model)
        
        self.u_adapter = nn.Sequential(
            nn.Linear(dim_u, d_model),
            nn.LeakyReLU(0.2),
            nn.Linear(d_model, d_model)
        )
        self.smooth = Smooth(dropout, d_model, kernel_size=5)
        self.alpha_ks = alpha_ks
        
        self.reliability_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid()
        )
        
        self.layers = nn.ModuleList([
            RATEEncoderLayer(d_model, nhead, dropout, d_ff) for _ in range(num_layers)
        ])
        
        self.num_experts = num_experts_k
        self.gate = nn.Sequential(
            nn.Linear(dim_u, num_experts_k),
            nn.Softmax(dim=-1)
        )
        
        self.experts = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_experts_k)])

    def forward(self, c, r, t, u_i=None, h_in=None, delta_in=None):
        if h_in is None:
            B, L = c.shape
            pos = torch.arange(L, device=c.device).unsqueeze(0).repeat(B, 1)
            
            h = (self.concept_embedding(c) + 
                 self.response_embedding(r) + 
                 self.pos_embedding(pos) + 
                 self.time_embedding(t))
                 
            if u_i is not None:
                u_feat = self.u_adapter(u_i)
                h = h + u_feat.unsqueeze(1) 
            h = self.smooth(h)
        else:
            h = h_in
            B, L, _ = h.shape

        causal_mask = torch.tril(torch.ones(L, L, device=h.device)).bool().view(1, 1, L, L)
        delta_0 = self.reliability_predictor(h).squeeze(-1) if delta_in is None else delta_in
        delta_l = delta_0
        
        for layer in self.layers:
            h, avg_attn = layer(h, delta_l, mask=causal_mask)
            struct_delta = torch.bmm(avg_attn, delta_l.unsqueeze(-1)).squeeze(-1)
            delta_l = self.alpha_ks * struct_delta + (1 - self.alpha_ks) * delta_0
            
        if u_i is not None:
            gate_weights = self.gate(u_i) 
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(torch.sigmoid(expert(h)).squeeze(-1))
            expert_outputs = torch.stack(expert_outputs, dim=-1)
            y_pred = (expert_outputs * gate_weights.unsqueeze(1)).sum(dim=-1)
        else:
            y_pred = torch.sigmoid(self.experts[0](h)).squeeze(-1)
            
        return y_pred, delta_l, h

class GANLosses:
    @staticmethod
    def wgan_gp_gradient_penalty(discriminator, real_samples, fake_samples, device):
        alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates, _ = discriminator(interpolates)
        fake = torch.ones(real_samples.size(0), 1).to(device)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    @staticmethod
    def wasserstein_style_contrastive_loss(real_feat, fake_feat, uids, margin=2.0):
        pos_dist = torch.norm(real_feat - fake_feat, p=2, dim=1)
        loss_pos = pos_dist.mean()
        uids = uids.view(-1, 1)
        neg_mask = (uids != uids.t()).float()
        if neg_mask.sum() == 0:
            return loss_pos
        dist_matrix = torch.cdist(real_feat, real_feat, p=2)
        valid_neg_dists = dist_matrix[neg_mask.bool()]
        loss_neg = torch.clamp(margin - valid_neg_dists, min=0).mean()
        return loss_pos + loss_neg