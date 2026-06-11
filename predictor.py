import os
import pickle
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("horse_predictor")

MODEL_PATH = "horse_racing_model.pkl"
PLACE_MODEL_PATH = "horse_racing_place_model.pkl"

def load_trained_model(path=MODEL_PATH):
    """
    保存されている学習済みモデルとメタデータをロードする
    """
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"Successfully loaded model from {path}.")
            return model_data
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
    return None

def preprocess_realtime_features(df_horses, histories, encoders, features_list, race_info):
    """
    リアルタイムデータを学習済みモデルが要求する特徴量形式に変換する。
    クラス格差に対応する賞金特徴量を算出。
    """
    df = df_horses.copy()
    
    # 今回のレース賞金額 (万円)
    current_prize = float(race_info.get("prize", 1000.0))
    
    # 過去成績に基づく特徴量の作成
    prev_ranks = []
    avg_ranks = []
    avg_last_3fs = []
    is_prev_top3s = []
    
    # 新設特徴量用のリスト
    prev_prizes = []
    prize_diffs = []
    avg_prizes = []
    
    for idx, row in df.iterrows():
        h_id = row['horse_id']
        history = histories.get(h_id, [])
        
        if history:
            # 1. 過去成績基本特徴
            p_rank = history[0]['rank']
            prev_ranks.append(p_rank if p_rank != 99 else 8)
            is_prev_top3s.append(1 if p_rank <= 3 else 0)
            
            ranks = [h['rank'] for h in history if h['rank'] != 99]
            avg_ranks.append(np.mean(ranks) if ranks else 8.0)
            
            last_3fs = [h['last_3f'] for h in history if h['last_3f'] > 30.0]
            avg_last_3fs.append(np.mean(last_3fs) if last_3fs else 36.0)
            
            # 2. クラスレベル（賞金）特徴量
            # 前走のレース賞金
            p_prize = float(history[0].get('estimated_race_prize', 500.0))
            prev_prizes.append(p_prize)
            # クラスレベルの差（今回 － 前走）
            prize_diffs.append(current_prize - p_prize)
            # 過去5走の平均賞金
            prizes = [float(h.get('estimated_race_prize', 500.0)) for h in history]
            avg_prizes.append(np.mean(prizes) if prizes else 500.0)
            
        else:
            prev_ranks.append(8)
            avg_ranks.append(8.0)
            avg_last_3fs.append(36.0)
            is_prev_top3s.append(0)
            # デフォルト値の設定
            prev_prizes.append(500.0)
            prize_diffs.append(current_prize - 500.0)
            avg_prizes.append(500.0)
            
    df['prev_rank'] = prev_ranks
    df['avg_rank'] = avg_ranks
    df['avg_last_3f'] = avg_last_3fs
    df['is_prev_top3'] = is_prev_top3s
    
    # 新設カラム追加
    df['prev_prize'] = prev_prizes
    df['prize_diff'] = prize_diffs
    df['avg_prize'] = avg_prizes
    
    # カテゴリ変数のエンコード
    track_type = race_info.get("track_type", "芝")
    track_condition = race_info.get("track_condition", "良")
    
    le_track = encoders.get('track_type')
    if le_track:
        try:
            if track_type in le_track.classes_:
                df['track_type_code'] = le_track.transform([track_type])[0]
            else:
                df['track_type_code'] = 0
        except Exception:
            df['track_type_code'] = 0
    else:
        df['track_type_code'] = 0
        
    le_cond = encoders.get('track_condition')
    if le_cond:
        try:
            if track_condition in le_cond.classes_:
                df['track_condition_code'] = le_cond.transform([track_condition])[0]
            else:
                df['track_condition_code'] = 0
        except Exception:
            df['track_condition_code'] = 0
    else:
        df['track_condition_code'] = 0
        
    # 型調整と欠損補完
    df['waku'] = df['waku'].astype(int)
    df['umaban'] = df['umaban'].astype(int)
    df['weight'] = df['weight'].fillna(55.0).astype(float)
    
    return df[features_list]

def calculate_base_win_rates(df_horses, histories, race_info=None):
    """
    出馬表と近走履歴から、単勝勝率および複勝勝率を算出する。
    クラス格差に対応。
    """
    if df_horses is None or len(df_horses) == 0:
        return df_horses
        
    if race_info is None:
        race_info = {
            "track_type": "芝",
            "track_condition": "良",
            "prize": 1000.0
        }
        
    df = df_horses.copy()
    
    # モデルのロード
    model_data_win = load_trained_model(MODEL_PATH)
    model_data_place = load_trained_model(PLACE_MODEL_PATH)
    
    df['odds'] = df['odds'].apply(lambda x: x if x > 1.0 else 99.0)
    df['place_odds_est'] = (df['odds'] * 0.3).round(1)
    
    is_ml_used = False
    
    if model_data_win is not None and model_data_place is not None:
        try:
            logger.info("Using trained LightGBM models (Win & Place - Class Aware) for prediction.")
            
            features_list = model_data_win['features']
            encoders = model_data_win['encoders']
            
            # 特徴量抽出 (新設の賞金特徴量が含まれる)
            X_realtime = preprocess_realtime_features(df_horses, histories, encoders, features_list, race_info)
            
            # 1. 単勝予測 (is_winner)
            preds_win = model_data_win['model'].predict(X_realtime)
            df['ml_score_win'] = preds_win
            sum_win = df['ml_score_win'].sum()
            df['base_win_rate'] = df['ml_score_win'] / sum_win if sum_win > 0 else 1.0 / len(df)
            
            # 2. 複勝予測 (is_place)
            preds_place = model_data_place['model'].predict(X_realtime)
            df['ml_score_place'] = preds_place
            sum_place = df['ml_score_place'].sum()
            df['base_place_rate'] = (df['ml_score_place'] / sum_place) * 3.0 if sum_place > 0 else 3.0 / len(df)
            
            is_ml_used = True
            
        except Exception as e:
            logger.error(f"Error during ML model inference: {e}. Falling back to rule-based.")
            
    # フォールバック処理
    if not is_ml_used:
        logger.info("Using statistical rule-based fallback calculation.")
        df['implied_probability'] = 0.8 / df['odds']
        
        adjustments = []
        for idx, row in df.iterrows():
            h_id = row['horse_id']
            history = histories.get(h_id, [])
            score = 0.0
            if history:
                ranks = [h['rank'] for h in history if h['rank'] != 99]
                avg_rank = np.mean(ranks) if ranks else 8.0
                score += (8.0 - avg_rank) * 0.25
                if history[0]['rank'] <= 3:
                    score += 0.5
                elif history[0]['rank'] >= 10:
                    score -= 0.5
                last_3fs = [h['last_3f'] for h in history if h['last_3f'] > 30.0]
                if last_3fs:
                    score += (35.0 - np.mean(last_3fs)) * 0.3
            adjustments.append(score)
            
        df['performance_score'] = adjustments
        multiplier = (1.0 + (df['performance_score'] * 0.1)).clip(lower=0.1)
        df['adjusted_probability'] = df['implied_probability'] * multiplier
        
        sum_prob = df['adjusted_probability'].sum()
        df['base_win_rate'] = df['adjusted_probability'] / sum_prob if sum_prob > 0 else 1.0 / len(df)
        df['base_place_rate'] = (df['base_win_rate'] * 2.8).clip(upper=0.95)
        
    df['is_ml_used'] = is_ml_used
    return df

def calculate_expected_values(df_evaluated, gemini_corrections):
    """
    ベース勝率にGemini補正を加え、モデルに保存された最適ブレンド係数（alpha）を用いて
    オッズ支持率とブレンドした最終的な単勝および複勝の期待値を計算する。
    """
    df = df_evaluated.copy()
    
    # モデルのロードから optimal_params を取得
    model_data_win = load_trained_model(MODEL_PATH)
    model_data_place = load_trained_model(PLACE_MODEL_PATH)
    
    opt_win = model_data_win.get('optimal_params', {}) if model_data_win else {}
    opt_place = model_data_place.get('optimal_params', {}) if model_data_place else {}
    
    # デフォルトのブレンド係数
    alpha_win = opt_win.get('alpha', 0.15) # バックテスト最適値
    alpha_place = opt_place.get('alpha', 0.3) # バックテスト最適値
    
    corrections = []
    for idx, row in df.iterrows():
        umaban = row['umaban']
        horse_name = row['horse_name']
        corr = gemini_corrections.get(umaban, 0.0)
        if corr == 0.0:
            corr = gemini_corrections.get(horse_name, 0.0)
        corrections.append(corr)
        
    df['gemini_correction'] = corrections
    
    # 1. 単勝：Gemini補正後のML確率 ＋ オッズ支持率 のブレンド
    ml_win_corrected = (df['base_win_rate'] + df['gemini_correction']).clip(0.005, 0.99)
    sum_ml_win = ml_win_corrected.sum()
    ml_win_prob = ml_win_corrected / sum_ml_win if sum_ml_win > 0 else 1.0 / len(df)
    
    # オッズ支持率の計算
    df['odds_support_rate_win'] = 0.8 / df['odds'].clip(lower=1.0)
    
    # ブレンド
    df['blended_win_prob'] = alpha_win * ml_win_prob + (1.0 - alpha_win) * df['odds_support_rate_win']
    sum_blended_win = df['blended_win_prob'].sum()
    df['final_win_rate'] = df['blended_win_prob'] / sum_blended_win if sum_blended_win > 0 else 1.0 / len(df)
    df['expected_value'] = df['final_win_rate'] * df['odds']
    
    # 2. 複勝：Gemini補正後のML確率 ＋ オッズ支持率 のブレンド
    df['gemini_correction_place'] = df['gemini_correction'] * 2.5
    ml_place_corrected = (df['base_place_rate'] + df['gemini_correction_place']).clip(0.01, 0.99)
    sum_ml_place = ml_place_corrected.sum()
    ml_place_prob = (ml_place_corrected / sum_ml_place) * 3.0 if sum_ml_place > 0 else 3.0 / len(df)
    
    # オッズ支持率の計算
    df['odds_support_rate_place'] = 0.8 / (df['odds'].clip(lower=1.0) * 0.3)
    
    # ブレンド
    df['blended_place_prob'] = alpha_place * ml_place_prob + (1.0 - alpha_place) * df['odds_support_rate_place']
    sum_blended_place = df['blended_place_prob'].sum()
    df['final_place_rate'] = (df['blended_place_prob'] / sum_blended_place) * 3.0 if sum_blended_place > 0 else 3.0 / len(df)
    df['expected_value_place'] = df['final_place_rate'] * df['place_odds_est']
    
    # メタデータ保持用カラムの追加
    df['opt_win_alpha'] = alpha_win
    df['opt_win_pop_limit'] = opt_win.get('pop_limit', 4)
    df['opt_win_ev_threshold'] = opt_win.get('ev_threshold', 1.05)
    
    df['opt_place_alpha'] = alpha_place
    df['opt_place_pop_limit'] = opt_place.get('pop_limit', 4)
    df['opt_place_ev_threshold'] = opt_place.get('ev_threshold', 1.1)
    
    return df

def calculate_pair_probabilities(df_final):
    """
    単勝最終確率(final_win_rate)から、Harville公式を用いて
    馬連およびワイドの各組み合わせ（ペア）の確率を厳密に算出する。
    """
    horses = df_final.copy()
    horses['umaban'] = horses['umaban'].astype(int)
    
    # 馬番と単勝確率の対応辞書
    win_probs = dict(zip(horses['umaban'], horses['final_win_rate']))
    umaban_list = sorted(list(win_probs.keys()))
    
    # 1. 馬連確率の計算: P(iが1着, jが2着) + P(jが1着, iが2着)
    umaren_probs = {}
    for i in range(len(umaban_list)):
        for j in range(i + 1, len(umaban_list)):
            u1 = umaban_list[i]
            u2 = umaban_list[j]
            p1 = win_probs[u1]
            p2 = win_probs[u2]
            
            p_12 = p1 * (p2 / max(1.0 - p1, 1e-5))
            p_21 = p2 * (p1 / max(1.0 - p2, 1e-5))
            umaren_probs[(u1, u2)] = p_12 + p_21
            
    # 2. ワイド確率の計算: 3着内に入賞する全組み合わせ(3頭)の確率を総当たり計算 (Harville 3着推定)
    triplet_probs = {}
    for a in umaban_list:
        for b in umaban_list:
            if a == b: continue
            for c in umaban_list:
                if a == c or b == c: continue
                
                pa = win_probs[a]
                pb = win_probs[b]
                pc = win_probs[c]
                
                # P(aが1着, bが2着, cが3着)
                p_abc = pa * (pb / max(1.0 - pa, 1e-5)) * (pc / max(1.0 - pa - pb, 1e-5))
                triplet_probs[(a, b, c)] = p_abc
                
    # ワイド確率の集計: i, jの両方が 3頭の組合せ (a,b,c) に含まれる確率の総和
    wide_probs = {}
    for i in range(len(umaban_list)):
        for j in range(i + 1, len(umaban_list)):
            u1 = umaban_list[i]
            u2 = umaban_list[j]
            
            sum_prob = 0.0
            for a in umaban_list:
                for b in umaban_list:
                    if a == b: continue
                    for c in umaban_list:
                        if a == c or b == c: continue
                        
                        # 3頭の中に u1 と u2 が含まれているかを判定
                        if (u1 in (a, b, c)) and (u2 in (a, b, c)):
                            sum_prob += triplet_probs.get((a, b, c), 0.0)
                            
            # 重複カウント(順列順の6パターン)が足し合わされているため、組合せ(組み合わせ)単位に集計済み
            wide_probs[(u1, u2)] = sum_prob
            
    return umaren_probs, wide_probs

def calculate_pair_expected_values(df_final, umaren_odds, wide_odds):
    """
    馬連・ワイドの確率とオッズを紐付け、期待値(EV)を計算する。
    """
    # 各ペアの確率を算出
    umaren_probs, wide_probs = calculate_pair_probabilities(df_final)
    
    umaren_evs = []
    wide_evs = []
    
    # 出走馬のリスト取得
    horses = df_final.copy()
    horses['umaban'] = horses['umaban'].astype(int)
    name_dict = dict(zip(horses['umaban'], horses['horse_name']))
    pop_dict = dict(zip(horses['umaban'], horses['popularity']))
    
    umaban_list = sorted(list(name_dict.keys()))
    
    # 1. 馬連期待値の集計
    for i in range(len(umaban_list)):
        for j in range(i + 1, len(umaban_list)):
            u1 = umaban_list[i]
            u2 = umaban_list[j]
            
            prob_ur = umaren_probs.get((u1, u2), 0.0)
            odds_ur = umaren_odds.get((u1, u2), 0.0)
            ev_ur = prob_ur * odds_ur
            
            pop_sum = pop_dict.get(u1, 10) + pop_dict.get(u2, 10)
            
            umaren_evs.append({
                "umaban_1": u1,
                "umaban_2": u2,
                "horse_1": name_dict[u1],
                "horse_2": name_dict[u2],
                "probability": prob_ur,
                "odds": odds_ur,
                "expected_value": ev_ur,
                "popularity_sum": pop_sum
            })
            
    # 2. ワイド期待値の集計
    for i in range(len(umaban_list)):
        for j in range(i + 1, len(umaban_list)):
            u1 = umaban_list[i]
            u2 = umaban_list[j]
            
            prob_w = wide_probs.get((u1, u2), 0.0)
            odds_w_list = wide_odds.get((u1, u2), [0.0, 0.0])
            odds_w_low = odds_w_list[0]
            odds_w_high = odds_w_list[1]
            
            ev_w_low = prob_w * odds_w_low
            ev_w_high = prob_w * odds_w_high
            
            pop_sum = pop_dict.get(u1, 10) + pop_dict.get(u2, 10)
            
            wide_evs.append({
                "umaban_1": u1,
                "umaban_2": u2,
                "horse_1": name_dict[u1],
                "horse_2": name_dict[u2],
                "probability": prob_w,
                "odds_low": odds_w_low,
                "odds_high": odds_w_high,
                "expected_value_low": ev_w_low,
                "expected_value_high": ev_w_high,
                "popularity_sum": pop_sum
            })
            
    df_umaren = pd.DataFrame(umaren_evs)
    df_wide = pd.DataFrame(wide_evs)
    
    return df_umaren, df_wide
