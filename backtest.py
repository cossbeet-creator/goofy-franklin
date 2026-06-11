import os
import pickle
import numpy as np
import pandas as pd
import logging
import sys
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backtest")

WIN_MODEL_PATH = "horse_racing_model.pkl"
PLACE_MODEL_PATH = "horse_racing_place_model.pkl"
RESULT_CSV_PATH = "data/19860105-20210731_race_result.csv"
ODDS_PATH = "data/19860105-20210731_odds.csv"

def run_backtest():
    if not os.path.exists(PLACE_MODEL_PATH):
        logger.error(f"Place Model file not found.")
        return
        
    logger.info("Loading datasets for advanced select-top-1 backtesting...")
    
    with open(PLACE_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
    encoders = model_data['encoders']
    
    # 賞金(万円) カラムをuse_colsに追加
    use_cols_res = ['レースID', 'レース日付', '芝・ダート区分', '馬場状態1', '着順', '枠番', '馬番', '馬名', '単勝', '人気', '斤量', '上り', '賞金(万円)']
    df_raw = pd.read_csv(RESULT_CSV_PATH, encoding='utf-8-sig', usecols=use_cols_res)
    df_raw['レース日付'] = pd.to_datetime(df_raw['レース日付'])
    df = df_raw[df_raw['レース日付'] >= '2020-01-01'].copy()
    del df_raw
    
    df = df.sort_values(by=['馬名', 'レース日付']).reset_index(drop=True)
    df['着順_num'] = pd.to_numeric(df['着順'], errors='coerce').fillna(99).astype(int)
    df['上り_num'] = pd.to_numeric(df['上り'], errors='coerce').fillna(36.0).astype(float)
    df['枠番_num'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(1).astype(int)
    df['馬番_num'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(1).astype(int)
    df['斤量_num'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(55.0).astype(float)
    df['odds'] = pd.to_numeric(df['単勝'], errors='coerce').fillna(99.0).astype(float)
    df['popularity'] = pd.to_numeric(df['人気'], errors='coerce').fillna(10).astype(int)
    df['賞金_num'] = pd.to_numeric(df['賞金(万円)'], errors='coerce').fillna(0.0).astype(float)
    
    # 過去成績基本特徴量
    df['prev_rank'] = df.groupby('馬名')['着順_num'].shift(1).fillna(8).astype(int)
    df['avg_rank'] = df.groupby('馬名')['着順_num'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(8.0)
    df['avg_last_3f'] = df.groupby('馬名')['上り_num'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(36.0)
    df['is_prev_top3'] = df['prev_rank'].apply(lambda x: 1 if x <= 3 else 0)
    
    # 【新設】クラス（レース賞金）特徴量
    # 各レースの1着想定賞金 (レース内最大獲得賞金)
    df['レース賞金'] = df.groupby('レースID')['賞金_num'].transform('max').fillna(500.0)
    # 前走のレース賞金
    df['prev_prize'] = df.groupby('馬名')['レース賞金'].shift(1).fillna(500.0).astype(float)
    # クラス変動（今回 － 前走）
    df['prize_diff'] = df['レース賞金'] - df['prev_prize']
    # 過去5走の平均レース賞金
    df['avg_prize'] = df.groupby('馬名')['レース賞金'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(500.0)
    
    df['waku'] = df['枠番_num']
    df['umaban'] = df['馬番_num']
    df['weight'] = df['斤量_num']
    
    # エンコード
    le_track = encoders['track_type']
    df['track_type_code'] = le_track.transform(df['芝・ダート区分'].astype(str))
    le_cond = encoders['track_condition']
    df['track_condition_code'] = df['馬場状態1'].astype(str).apply(
        lambda x: le_cond.transform([x])[0] if x in le_cond.classes_ else 0
    )
    
    df['is_winner'] = df['着順_num'].apply(lambda x: 1 if x == 1 else 0)
    df['is_place'] = df['着順_num'].apply(lambda x: 1 if x <= 3 else 0)
    
    # 複勝配当データのマージ
    logger.info("Merging place payouts...")
    use_cols_odds = [
        'レースID', 
        '単勝1_オッズ', '複勝1_馬番', '複勝2_馬番', '複勝3_馬番', '複勝4_馬番', '複勝5_馬番',
        '複勝1_オッズ', '複勝2_オッズ', '複勝3_オッズ', '複勝4_オッズ', '複勝5_オッズ',
        '馬連1_組合せ1', '馬連1_組合せ2', '馬連1_オッズ',
        '馬連2_組合せ1', '馬連2_組合せ2', '馬連2_オッズ',
        'ワイド1_組合せ1', 'ワイド1_組合せ2', 'ワイド1_オッズ',
        'ワイド2_組合せ1', 'ワイド2_組合せ2', 'ワイド2_オッズ',
        'ワイド3_組合せ1', 'ワイド3_組合せ2', 'ワイド3_オッズ',
        'ワイド4_組合せ1', 'ワイド4_組合せ2', 'ワイド4_オッズ',
        'ワイド5_組合せ1', 'ワイド5_組合せ2', 'ワイド5_オッズ',
        'ワイド6_組合せ1', 'ワイド6_組合せ2', 'ワイド6_オッズ',
        'ワイド7_組合せ1', 'ワイド7_組合せ2', 'ワイド7_オッズ'
    ]
    df_odds = pd.read_csv(ODDS_PATH, encoding='utf-8-sig', usecols=use_cols_odds)
    df = pd.merge(df, df_odds, on='レースID', how='left')
    del df_odds
    
    place_payouts = []
    win_payouts = []
    for idx, row in df.iterrows():
        umaban = row['umaban']
        p_payout = 0.0
        w_payout = 0.0
        
        if row['is_place'] == 1:
            for i in range(1, 6):
                if row[f'複勝{i}_馬番'] == umaban:
                    val = row[f'複勝{i}_オッズ']
                    p_payout = float(val) / 100.0 if not pd.isna(val) else 0.0
                    break
        if row['is_winner'] == 1:
            w_payout = row['odds']
            
        place_payouts.append(p_payout)
        win_payouts.append(w_payout)
        
    df['place_payout'] = place_payouts
    df['win_payout'] = win_payouts
    
    # 1. 複勝確率予測と期待値の計算
    logger.info("Predicting place probabilities...")
    df['place_prob_raw'] = model.predict(df[features])
    df['place_prob_sum'] = df.groupby('レースID')['place_prob_raw'].transform('sum')
    df['place_prob'] = (df['place_prob_raw'] / df['place_prob_sum']) * 3.0
    
    df['odds_support_rate_place'] = 0.8 / (df['odds'] * 0.3)
    df['estimated_place_odds'] = df['odds'] * 0.3
    
    # 2. 単勝確率予測と期待値の計算 (追加)
    logger.info("Loading win model for win-based backtesting...")
    win_model = None
    if os.path.exists(WIN_MODEL_PATH):
        with open(WIN_MODEL_PATH, 'rb') as f:
            win_model_data = pickle.load(f)
        win_model = win_model_data['model']
        
        logger.info("Predicting win probabilities...")
        df['win_prob_raw'] = win_model.predict(df[features])
        df['win_prob_sum'] = df.groupby('レースID')['win_prob_raw'].transform('sum')
        df['win_prob'] = df['win_prob_raw'] / df['win_prob_sum']
        df['odds_support_rate_win'] = 0.8 / df['odds']
    
    # パラメータ設定
    alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
    pop_limits = [3, 4, 5, 99]
    ev_thresholds = [0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    
    # --- 戦略1: 複勝期待値ベースで単勝/複勝を買う ---
    best_roi_place_strat = 0.0
    best_params_place_strat = {}
    place_strat_results = []
    
    logger.info("Running Place-EV-based Select-Top-1 strategy grid search...")
    for alpha in alphas:
        df['blended_place_prob'] = alpha * df['place_prob'] + (1 - alpha) * df['odds_support_rate_place']
        df['blended_place_sum'] = df.groupby('レースID')['blended_place_prob'].transform('sum')
        df['final_place_prob'] = (df['blended_place_prob'] / df['blended_place_sum']) * 3.0
        df['expected_value_place'] = df['final_place_prob'] * df['estimated_place_odds']
        
        for pop_limit in pop_limits:
            pop_cond = (df['odds'] <= 100.0)
            if pop_limit != 99:
                pop_cond = pop_cond & (df['popularity'] <= pop_limit)
                
            df_filtered = df[pop_cond].copy()
            if len(df_filtered) == 0:
                continue
                
            max_evs = df_filtered.groupby('レースID')['expected_value_place'].transform('max')
            df_filtered['is_max_ev_in_race'] = (df_filtered['expected_value_place'] == max_evs)
            
            for ev_th in ev_thresholds:
                purchased = df_filtered[df_filtered['is_max_ev_in_race'] & (df_filtered['expected_value_place'] >= ev_th)]
                total_bets = len(purchased)
                
                if total_bets >= 50:
                    total_returns_place = purchased['place_payout'].sum()
                    roi_place = (total_returns_place / total_bets) * 100
                    hit_rate_place = (len(purchased[purchased['is_place'] == 1]) / total_bets) * 100
                    
                    total_returns_win = purchased['win_payout'].sum()
                    roi_win = (total_returns_win / total_bets) * 100
                    hit_rate_win = (len(purchased[purchased['is_winner'] == 1]) / total_bets) * 100
                    
                    place_strat_results.append({
                        "alpha": alpha,
                        "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                        "ev_threshold": ev_th,
                        "total_bets": total_bets,
                        "hit_rate_place": hit_rate_place,
                        "roi_place": roi_place,
                        "roi_win": roi_win
                    })
                    
                    if roi_place > best_roi_place_strat:
                        best_roi_place_strat = roi_place
                        best_params_place_strat = {
                            "alpha": alpha,
                            "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                            "ev_threshold": ev_th,
                            "total_bets": total_bets,
                            "hit_rate_place": f"{hit_rate_place:.1f}%",
                            "roi_place": f"{roi_place:.2f}%",
                            "roi_win_ref": f"{roi_win:.2f}%"
                        }
                        
    # --- 戦略2: 単勝期待値ベースで単勝を買う ---
    best_roi_win_strat = 0.0
    best_params_win_strat = {}
    win_strat_results = []
    
    if win_model is not None:
        logger.info("Running Win-EV-based Select-Top-1 strategy grid search...")
        for alpha in alphas:
            df['blended_win_prob'] = alpha * df['win_prob'] + (1 - alpha) * df['odds_support_rate_win']
            df['blended_win_sum'] = df.groupby('レースID')['blended_win_prob'].transform('sum')
            df['final_win_prob'] = df['blended_win_prob'] / df['blended_win_sum']
            df['expected_value_win'] = df['final_win_prob'] * df['odds']
            
            for pop_limit in pop_limits:
                pop_cond = (df['odds'] <= 100.0)
                if pop_limit != 99:
                    pop_cond = pop_cond & (df['popularity'] <= pop_limit)
                    
                df_filtered = df[pop_cond].copy()
                if len(df_filtered) == 0:
                    continue
                    
                max_evs = df_filtered.groupby('レースID')['expected_value_win'].transform('max')
                df_filtered['is_max_ev_in_race'] = (df_filtered['expected_value_win'] == max_evs)
                
                for ev_th in ev_thresholds:
                    purchased = df_filtered[df_filtered['is_max_ev_in_race'] & (df_filtered['expected_value_win'] >= ev_th)]
                    total_bets = len(purchased)
                    
                    if total_bets >= 50:
                        total_returns_win = purchased['win_payout'].sum()
                        roi_win = (total_returns_win / total_bets) * 100
                        hit_rate_win = (len(purchased[purchased['is_winner'] == 1]) / total_bets) * 100
                        
                        win_strat_results.append({
                            "alpha": alpha,
                            "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                            "ev_threshold": ev_th,
                            "total_bets": total_bets,
                            "hit_rate_win": hit_rate_win,
                            "roi_win": roi_win
                        })
                        
                        if roi_win > best_roi_win_strat:
                            best_roi_win_strat = roi_win
                            best_params_win_strat = {
                                "alpha": alpha,
                                "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                                "ev_threshold": ev_th,
                                "total_bets": total_bets,
                                "hit_rate_win": f"{hit_rate_win:.1f}%",
                                "roi_win": f"{roi_win:.2f}%"
                            }
                            
    # --- 戦略3: 馬連・ワイドのグリッドサーチ ---
    logger.info("Running Quinella & Wide backtest preparation...")
    # 単勝の最適 alpha = 0.15 を使用
    alpha_win_opt = 0.15
    df['blended_win_prob_opt'] = alpha_win_opt * df['win_prob'] + (1 - alpha_win_opt) * df['odds_support_rate_win']
    df['blended_win_sum_opt'] = df.groupby('レースID')['blended_win_prob_opt'].transform('sum')
    df['final_win_prob_opt'] = df['blended_win_prob_opt'] / df['blended_win_sum_opt']
    
    # 配当辞書の構築
    logger.info("Building Quinella & Wide payout lookup dictionaries...")
    umaren_payouts_dict = {}
    wide_payouts_dict = {}
    
    df_unique_odds = df[['レースID', 
                         '馬連1_組合せ1', '馬連1_組合せ2', '馬連1_オッズ',
                         '馬連2_組合せ1', '馬連2_組合せ2', '馬連2_オッズ',
                         'ワイド1_組合せ1', 'ワイド1_組合せ2', 'ワイド1_オッズ',
                         'ワイド2_組合せ1', 'ワイド2_組合せ2', 'ワイド2_オッズ',
                         'ワイド3_組合せ1', 'ワイド3_組合せ2', 'ワイド3_オッズ',
                         'ワイド4_組合せ1', 'ワイド4_組合せ2', 'ワイド4_オッズ',
                         'ワイド5_組合せ1', 'ワイド5_組合せ2', 'ワイド5_オッズ',
                         'ワイド6_組合せ1', 'ワイド6_組合せ2', 'ワイド6_オッズ',
                         'ワイド7_組合せ1', 'ワイド7_組合せ2', 'ワイド7_オッズ']].drop_duplicates(subset=['レースID'])
                         
    for idx, row in df_unique_odds.iterrows():
        r_id = row['レースID']
        # 馬連 1, 2
        for m in range(1, 3):
            u1, u2 = row[f'馬連{m}_組合せ1'], row[f'馬連{m}_組合せ2']
            odds_val = row[f'馬連{m}_オッズ']
            if not pd.isna(u1) and not pd.isna(u2) and not pd.isna(odds_val):
                p1, p2 = int(u1), int(u2)
                pair = (min(p1, p2), max(p1, p2))
                umaren_payouts_dict[(r_id, pair)] = float(odds_val) / 100.0
                
        # ワイド 1〜7
        for k in range(1, 8):
            u1, u2 = row[f'ワイド{k}_組合せ1'], row[f'ワイド{k}_組合せ2']
            odds_val = row[f'ワイド{k}_オッズ']
            if not pd.isna(u1) and not pd.isna(u2) and not pd.isna(odds_val):
                p1, p2 = int(u1), int(u2)
                pair = (min(p1, p2), max(p1, p2))
                wide_payouts_dict[(r_id, pair)] = float(odds_val) / 100.0

    # ペア確率の計算
    logger.info("Calculating Quinella & Wide probabilities for all test races...")
    umaren_records = []
    wide_records = []
    
    grouped = df.groupby('レースID')
    
    for race_id, group in grouped:
        umaban_list = group['umaban'].tolist()
        win_probs = group['final_win_prob_opt'].tolist()
        odds_list = group['odds'].tolist()
        pop_list = group['popularity'].tolist()
        
        n = len(umaban_list)
        if n < 4:
            continue
            
        prob_dict = dict(zip(umaban_list, win_probs))
        odds_dict = dict(zip(umaban_list, odds_list))
        pop_dict = dict(zip(umaban_list, pop_list))
        
        # 1. 馬連確率
        umaren_probs = {}
        for i in range(n):
            for j in range(i + 1, n):
                u1, u2 = umaban_list[i], umaban_list[j]
                p1, p2 = win_probs[i], win_probs[j]
                p_12 = p1 * (p2 / max(1.0 - p1, 1e-5))
                p_21 = p2 * (p1 / max(1.0 - p2, 1e-5))
                umaren_probs[(u1, u2)] = p_12 + p_21
                
        # 2. ワイド確率 (Harville 3着推定の最適化)
        triplet_probs = {}
        for a in umaban_list:
            pa = prob_dict[a]
            if pa < 1e-5: continue
            for b in umaban_list:
                if a == b: continue
                pb = prob_dict[b]
                if pb < 1e-5: continue
                p_ab = pa * (pb / max(1.0 - pa, 1e-5))
                for c in umaban_list:
                    if a == c or b == c: continue
                    pc = prob_dict[c]
                    if pc < 1e-5: continue
                    p_abc = p_ab * (pc / max(1.0 - pa - pb, 1e-5))
                    triplet_probs[(a, b, c)] = p_abc
                    
        # 各ペアのワイド確率を足し込む (重複を許容した3重ループの最適化)
        wide_probs = defaultdict(float)
        for (a, b, c), p_abc in triplet_probs.items():
            wide_probs[(min(a, b), max(a, b))] += p_abc
            wide_probs[(min(b, c), max(b, c))] += p_abc
            wide_probs[(min(a, c), max(a, c))] += p_abc
            
        for i in range(n):
            for j in range(i + 1, n):
                u1, u2 = umaban_list[i], umaban_list[j]
                sum_w_prob = wide_probs.get((u1, u2), 0.0)
                                
                # 決定論的想定オッズ
                est_odds_umaren = max(odds_dict[u1] * odds_dict[u2] * 0.20, 1.1)
                est_odds_wide = max(odds_dict[u1] * odds_dict[u2] * 0.08, 1.1)
                
                payout_umaren = umaren_payouts_dict.get((race_id, (min(u1, u2), max(u1, u2))), 0.0)
                payout_wide = wide_payouts_dict.get((race_id, (min(u1, u2), max(u1, u2))), 0.0)
                
                pop_sum = pop_dict[u1] + pop_dict[u2]
                
                umaren_records.append({
                    'レースID': race_id,
                    'umaban_1': u1,
                    'umaban_2': u2,
                    'probability': umaren_probs[(u1, u2)],
                    'estimated_odds': est_odds_umaren,
                    'expected_value': umaren_probs[(u1, u2)] * est_odds_umaren,
                    'popularity_sum': pop_sum,
                    'payout': payout_umaren
                })
                
                wide_records.append({
                    'レースID': race_id,
                    'umaban_1': u1,
                    'umaban_2': u2,
                    'probability': sum_w_prob,
                    'estimated_odds': est_odds_wide,
                    'expected_value': sum_w_prob * est_odds_wide,
                    'popularity_sum': pop_sum,
                    'payout': payout_wide
                })
                
    df_umaren_all = pd.DataFrame(umaren_records)
    df_wide_all = pd.DataFrame(wide_records)
    
    # パラメータ設定 (馬連・ワイド)
    pop_sum_limits = [8, 10, 12, 14, 16, 99]
    ev_thresholds_pair = [0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    
    # --- 馬連グリッドサーチ ---
    best_roi_umaren = 0.0
    best_params_umaren = {}
    umaren_results = []
    
    logger.info("Running Quinella-EV select-top-1 strategy grid search...")
    for pop_limit in pop_sum_limits:
        pop_cond = (df_umaren_all['popularity_sum'] <= pop_limit) if pop_limit != 99 else pd.Series([True]*len(df_umaren_all))
        df_filtered = df_umaren_all[pop_cond].copy()
        if len(df_filtered) == 0:
            continue
            
        max_evs = df_filtered.groupby('レースID')['expected_value'].transform('max')
        df_filtered['is_max_ev_in_race'] = (df_filtered['expected_value'] == max_evs)
        
        for ev_th in ev_thresholds_pair:
            purchased = df_filtered[df_filtered['is_max_ev_in_race'] & (df_filtered['expected_value'] >= ev_th)]
            total_bets = len(purchased)
            
            if total_bets >= 50:
                total_returns = purchased['payout'].sum()
                roi = (total_returns / total_bets) * 100
                hit_rate = (len(purchased[purchased['payout'] > 0]) / total_bets) * 100
                
                umaren_results.append({
                    "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                    "ev_threshold": ev_th,
                    "total_bets": total_bets,
                    "hit_rate": hit_rate,
                    "roi": roi
                })
                
                if roi > best_roi_umaren:
                    best_roi_umaren = roi
                    best_params_umaren = {
                        "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                        "ev_threshold": ev_th,
                        "total_bets": total_bets,
                        "hit_rate": f"{hit_rate:.1f}%",
                        "roi": f"{roi:.2f}%"
                    }
                    
    # --- ワイドグリッドサーチ ---
    best_roi_wide = 0.0
    best_params_wide = {}
    wide_results = []
    
    logger.info("Running Wide-EV select-top-1 strategy grid search...")
    for pop_limit in pop_sum_limits:
        pop_cond = (df_wide_all['popularity_sum'] <= pop_limit) if pop_limit != 99 else pd.Series([True]*len(df_wide_all))
        df_filtered = df_wide_all[pop_cond].copy()
        if len(df_filtered) == 0:
            continue
            
        max_evs = df_filtered.groupby('レースID')['expected_value'].transform('max')
        df_filtered['is_max_ev_in_race'] = (df_filtered['expected_value'] == max_evs)
        
        for ev_th in ev_thresholds_pair:
            purchased = df_filtered[df_filtered['is_max_ev_in_race'] & (df_filtered['expected_value'] >= ev_th)]
            total_bets = len(purchased)
            
            if total_bets >= 50:
                total_returns = purchased['payout'].sum()
                roi = (total_returns / total_bets) * 100
                hit_rate = (len(purchased[purchased['payout'] > 0]) / total_bets) * 100
                
                wide_results.append({
                    "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                    "ev_threshold": ev_th,
                    "total_bets": total_bets,
                    "hit_rate": hit_rate,
                    "roi": roi
                })
                
                if roi > best_roi_wide:
                    best_roi_wide = roi
                    best_params_wide = {
                        "pop_limit": pop_limit if pop_limit != 99 else "NoLimit",
                        "ev_threshold": ev_th,
                        "total_bets": total_bets,
                        "hit_rate": f"{hit_rate:.1f}%",
                        "roi": f"{roi:.2f}%"
                    }

    # 結果表示
    if place_strat_results:
        df_p_results = pd.DataFrame(place_strat_results).sort_values(by="roi_place", ascending=False)
        print("\n=== TOP 5 CONFIGURATIONS (PLACE-EV STRATEGY) ===")
        print(df_p_results[["alpha", "pop_limit", "ev_threshold", "total_bets", "hit_rate_place", "roi_place", "roi_win"]].head(5).to_string(index=False))
        
    if win_strat_results:
        df_w_results = pd.DataFrame(win_strat_results).sort_values(by="roi_win", ascending=False)
        print("\n=== TOP 5 CONFIGURATIONS (WIN-EV STRATEGY) ===")
        print(df_w_results[["alpha", "pop_limit", "ev_threshold", "total_bets", "hit_rate_win", "roi_win"]].head(5).to_string(index=False))
        
    if umaren_results:
        df_u_results = pd.DataFrame(umaren_results).sort_values(by="roi", ascending=False)
        print("\n=== TOP 5 CONFIGURATIONS (QUINELLA-EV STRATEGY) ===")
        print(df_u_results[["pop_limit", "ev_threshold", "total_bets", "hit_rate", "roi"]].head(5).to_string(index=False))

    if wide_results:
        df_wi_results = pd.DataFrame(wide_results).sort_values(by="roi", ascending=False)
        print("\n=== TOP 5 CONFIGURATIONS (WIDE-EV STRATEGY) ===")
        print(df_wi_results[["pop_limit", "ev_threshold", "total_bets", "hit_rate", "roi"]].head(5).to_string(index=False))

    print("\n[BEST PLACE-EV CONFIGURATION FOUND]")
    if best_params_place_strat:
        for k, v in best_params_place_strat.items():
            print(f" - {k}: {v}")
            
    print("\n[BEST WIN-EV CONFIGURATION FOUND]")
    if best_params_win_strat:
        for k, v in best_params_win_strat.items():
            print(f" - {k}: {v}")
            
    print("\n[BEST QUINELLA-EV CONFIGURATION FOUND]")
    if best_params_umaren:
        for k, v in best_params_umaren.items():
            print(f" - {k}: {v}")

    print("\n[BEST WIDE-EV CONFIGURATION FOUND]")
    if best_params_wide:
        for k, v in best_params_wide.items():
            print(f" - {k}: {v}")

    # 最良パラメータの保存
    try:
        # 複勝モデルに複勝・ワイド戦略パラメータを保存
        if best_params_place_strat:
            model_data['optimal_params'] = {
                'alpha': best_params_place_strat['alpha'],
                'pop_limit': 5 if best_params_place_strat['pop_limit'] == "NoLimit" else int(best_params_place_strat['pop_limit']),
                'ev_threshold': float(best_params_place_strat['ev_threshold']),
                'strategy': 'place_ev'
            }
            if best_params_wide:
                model_data['optimal_params_wide'] = {
                    'pop_limit': 99 if best_params_wide['pop_limit'] == "NoLimit" else int(best_params_wide['pop_limit']),
                    'ev_threshold': float(best_params_wide['ev_threshold'])
                }
            with open(PLACE_MODEL_PATH, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Saved optimal place & wide strategy parameters to Place model.")
            
        # 単勝モデルに単勝・馬連戦略パラメータを保存
        if best_params_win_strat and win_model is not None:
            win_model_data['optimal_params'] = {
                'alpha': best_params_win_strat['alpha'],
                'pop_limit': 5 if best_params_win_strat['pop_limit'] == "NoLimit" else int(best_params_win_strat['pop_limit']),
                'ev_threshold': float(best_params_win_strat['ev_threshold']),
                'strategy': 'win_ev'
            }
            if best_params_umaren:
                win_model_data['optimal_params_umaren'] = {
                    'pop_limit': 99 if best_params_umaren['pop_limit'] == "NoLimit" else int(best_params_umaren['pop_limit']),
                    'ev_threshold': float(best_params_umaren['ev_threshold'])
                }
            with open(WIN_MODEL_PATH, 'wb') as f:
                pickle.dump(win_model_data, f)
            logger.info("Saved optimal win & umaren strategy parameters to Win model.")
            
    except Exception as e:
        logger.error(f"Error saving optimal params: {e}")

if __name__ == "__main__":
    run_backtest()

