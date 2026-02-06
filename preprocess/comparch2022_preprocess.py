# _*_ coding:utf-8 _*_

import pandas as pd
from .utils import sta_infos, write_txt, format_list2str

# Keys for stats display. Map to actual column names in comparch2022
KEYS = ["id_col", "skill_col", "prob_col"]

def read_data_from_csv(read_file, write_file):
    stares = []

    # Read CSV
    # sample: id_col,prob_col,skill_col,time_col,score_col,score_col_correct,probability
    # We use: id_col (user), prob_col (problem), skill_col (concept), time_col (timestamp), score_col_correct (correctness, assume 0/1)
    # Using score_col_correct instead of score_col to ensure 0/1 binary values
    df = pd.read_csv(read_file, dtype={'id_col': str, 'prob_col': str, 'skill_col': str, 'score_col_correct': str, 'time_col': float})

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    # Pre-filter NA on essential columns
    df = df.dropna(subset=["id_col", "prob_col", "skill_col", "score_col_correct", "time_col"])

    # Handle Multi-Skill: "8,14,15,19" -> Keep as "8_14_15_19" for split_datasets.py to handle is_repeat
    # 1. Split to list
    df['skill_col'] = df['skill_col'].str.split(',')
    # 2. Join with '_' (and strip whitespace)
    df['skill_col'] = df['skill_col'].apply(lambda skills: "_".join([s.strip() for s in skills if s.strip()]))
    
    # Create temporary index for stable sort if needed, though we rely on time_col mainly
    df['tmp_index'] = range(len(df))
    
    # Drop rows where skills might be empty
    _df = df.dropna(subset=["id_col", "prob_col", "skill_col", "score_col_correct", "time_col"])
    _df = _df[_df['skill_col'] != ''] # Remove empty string skills

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(_df, KEYS, stares)
    # Note: ins count here will be smaller (folded), cs might count unique strings like "8_14" as a concept temporarily
    print(f"after drop and merge multi-skills interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    ui_df = _df.groupby('id_col', sort=False)

    user_inters = []
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        
        # Sort by time, then by tmp_index (original order preserved for exploded items)
        tmp_inter = tmp_inter.sort_values(by=['time_col', 'tmp_index'])
        
        seq_len = len(tmp_inter)
        seq_problems = tmp_inter['prob_col'].tolist()
        seq_skills = tmp_inter['skill_col'].tolist()
        # Use score_col_correct
        seq_ans = tmp_inter['score_col_correct'].astype(float).astype(int).astype(str).tolist()
        
        # In assist2009 preprocess, these are placeholder 'NA' lists?
        # But wait, KTCSVDataset reads t_intervals from 'timestamps' column if present.
        # assist2009_preprocess.py sets seq_start_time = ['NA']? 
        # Let's check assist2009_preprocess.py again.
        # It sets seq_start_time = ['NA'] and seq_response_cost = ['NA'].
        # The writer writes them.
        
        # NOTE: If we want to use timestamps in the model, we should probably output them.
        # But for 'like assist2009', I should follow its pattern.
        # assist2009_preprocess.py seems to IGNORE timestamps in output?
        # "seq_start_time = ['NA']"
        # The output keys in data.txt: 
        # But wait, KTCSVDataset:
        # if 'timestamps' in row: ...
        # The file written by write_txt is a text file.
        # Let's check format_list2str and write_txt in utils.py.
        # But if assist2009_preprocess.py output 'NA', then the model won't get times for assist2009.
        # The user asked to be "like assist2009_preprocess.py".
        # However, for comparch2022 we HAVE times.
        # If I look at KTCSVDataset in main.py, it handles 'timestamps'.
        # If I output 'NA', it uses 0.0.
        # Do I want 0.0?
        # assist2009_preprocess.py: "seq_start_time = ['NA']"
        # Maybe assist2009 DOESN'T use timestamps in this pipeline version?
        # But the main.py `train_pipeline` sets `data_path` to `assist2009/train_valid_sequences.csv`.
        # Is that the output of this preprocess?
        # `data_preprocess.py` creates `dname/data.txt` then calls `split_datasets`.
        # `split_datasets.py` reads `data.txt`?
        # Line 50 of `preprocess/split_datasets.py`: `lines = fin.readlines()`...
        # It parses the weird format.
        # If I put 'NA', it reads 'NA'.
        # If `assist2009` preprocess explicitly puts 'NA', I will stick to it unless I see reason not to.
        # Wait, usually `assist2009` HAS timestamps (order_id is order, but time is time).
        # In `comparch2022` we have `time_col`. 
        # I'll stick to `seq_start_time = ['NA']` to be exactly "like assist2009_preprocess", 
        # UNLESS the user implies utilizing the data fully.
        # But `assist2009_preprocess.py` specifically uses `order_id` to sort but outputs `NA` for time.
        # This implies the downstream/model might not need it or it's a simplified preprocessing.
        # HOWEVER, looking at `prl-sum2026/main.py`: `t_raw = [float(x) for x in t_str.split(',') if x.strip()]`.
        # If `t_str` is "NA", this might crash or produce valid handling if `split` result is empty?
        # If `t_str` is "NA", `t_str.split(',')` is `['NA']`. `float('NA')` -> ValueError.
        # So `assist2009` pipeline seemingly DOES NOT use Time?
        # In `main.py`: `else: t_raw = [0.0] * len(...)`.
        # So yes, defaulting to No Time.
        # I will follow the pattern.
        
        seq_start_time = ['NA']
        seq_response_cost = ['NA']

        user_inters.append(
            [[str(user), str(seq_len)], format_list2str(seq_problems), format_list2str(seq_skills), format_list2str(seq_ans), seq_start_time, seq_response_cost])

    write_txt(write_file, user_inters)

    print("\n".join(stares))

    return
