import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, roc_auc_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_training")

MODEL_PATH = "horse_racing_model.pkl"
PLACE_MODEL_PATH = "horse_racing_place_model.pkl"
DATA_DIR = "data"
CSV_FILE_NAME = "19860105-20210731_race_result.csv"

# 【改修】クラス格差を考慮するため、賞金ベースの特徴量をFEATURESに追加！
FEATURES = [
    'waku', 'umaban', 'weight', 
    'prev_rank', 'avg_rank', 'avg_last_3f', 'is_prev_top3',
    'track_type_code', 'track_condition_code',
    'prev_prize', 'prize_diff', 'avg_prize' # 新設特徴量（レースレベル判定用）
]

USE_COLS = [
    'レースID', 'レース日付', '芝・ダート区分', '馬場状態1', '距離(m)', 
    '着順', '枠番', '馬番', '馬名', '単勝', '人気', '斤量', '上り', '賞金(万円)'
]

def generate_realistic_mock_data(num_races=1000):
    """
    ダミーデータセットの自動生成（仮想賞金データも含める）
    """
    logger.info(f"Generating {num_races} races of realistic mock data for pipeline verification...")
    
    data = []
    track_types = ['芝', 'ダ']
    conditions = ['良', '稍', '重', '不']
    
    # 仮想クラス設定
    class_prizes = [500.0, 1000.0, 2000.0, 3500.0, 7000.0, 15000.0]
    
    for race_id in range(1, num_races + 1):
        num_horses = np.random.randint(8, 17)
        track_type = np.random.choice(track_types)
        condition = np.random.choice(conditions)
        distance = np.random.choice([1200, 1400, 1600, 1800, 2000, 2200, 2400])
        base_abilities = np.random.normal(50, 10, num_horses)
        ranks_ability = np.argsort(np.argsort(-base_abilities)) + 1
        
        # 1着賞金の設定
        class_prize = np.random.choice(class_prizes)
        
        odds = []
        for r in ranks_ability:
            base_odds = r * 1.5 + np.random.exponential(2)
            odds.append(round(max(base_odds, 1.1), 1))
            
        prev_ranks = []
        avg_ranks = []
        avg_last_3fs = []
        prev_prizes = []
        
        for i in range(num_horses):
            ability = base_abilities[i]
            p_rank = max(1, int(np.random.normal(15 - (ability / 5), 2)))
            a_rank = max(1, np.random.normal(15 - (ability / 5), 1.5))
            base_3f = 35.0 if track_type == '芝' else 37.0
            last_3f = base_3f - (ability - 50) * 0.1 + np.random.normal(0, 0.5)
            
            # 前走の賞金
            p_prize = np.random.choice(class_prizes)
            
            prev_ranks.append(p_rank)
            avg_ranks.append(round(a_rank, 1))
            avg_last_3fs.append(round(last_3f, 1))
            prev_prizes.append(p_prize)
            
        race_scores = base_abilities + np.random.normal(0, 8, num_horses)
        final_ranks = np.argsort(-race_scores) + 1
        
        for i in range(num_horses):
            umaban = i + 1
            waku = (i // 2) + 1
            
            year = 2015 + (race_id // 200)
            month = 1 + ((race_id // 20) % 12)
            day = 1 + (race_id % 28)
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            
            # 獲得した賞金
            payout = 0.0
            if final_ranks[i] == 1:
                payout = class_prize
            elif final_ranks[i] == 2:
                payout = class_prize * 0.4
            elif final_ranks[i] == 3:
                payout = class_prize * 0.25
                
            data.append({
                "race_id": f"2025{race_id:08d}",
                "waku": waku,
                "umaban": umaban,
                "horse_name": f"MockHorse_{race_id}_{umaban}",
                "odds": odds[i],
                "popularity": int(ranks_ability[i]),
                "weight": float(np.random.randint(440, 540)),
                "prev_rank": prev_ranks[i],
                "avg_rank": avg_ranks[i],
                "avg_last_3f": avg_last_3fs[i],
                "track_type": track_type,
                "track_condition": condition,
                "distance": distance,
                "rank": final_ranks[i],
                "レース日付": date_str,
                "賞金(万円)": payout,
                "prev_prize_mock": prev_prizes[i],
                "class_prize_mock": class_prize
            })
            
    df = pd.DataFrame(data)
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    csv_path = os.path.join(DATA_DIR, "mock_historical_data.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved mock historical data to {csv_path}")
    return df

def load_and_preprocess_data():
    """
    CSVデータをロードし、機械学習用の特徴量と目的変数に前処理する。
    クラス格差に対応するため、賞金特徴量を作成。
    """
    real_csv_path = os.path.join(DATA_DIR, CSV_FILE_NAME)
    is_real_data = os.path.exists(real_csv_path)
    
    if is_real_data:
        logger.info(f"Detected actual historical data CSV at {real_csv_path}. Starting parsing...")
        try:
            df_raw = pd.read_csv(real_csv_path, encoding='utf-8-sig', usecols=USE_COLS)
            df_raw['レース日付'] = pd.to_datetime(df_raw['レース日付'])
            
            logger.info("Filtering data from 2015-01-01 onwards to optimize memory and training speed...")
            df = df_raw[df_raw['レース日付'] >= '2015-01-01'].copy()
            del df_raw
            
            logger.info(f"Filtered dataset contains {len(df)} rows.")
            df = df.sort_values(by=['馬名', 'レース日付']).reset_index(drop=True)
            
            df['着順_num'] = pd.to_numeric(df['着順'], errors='coerce').fillna(99).astype(int)
            df['上り_num'] = pd.to_numeric(df['上り'], errors='coerce').fillna(36.0).astype(float)
            df['枠番_num'] = pd.to_numeric(df['枠番'], errors='coerce').fillna(1).astype(int)
            df['馬番_num'] = pd.to_numeric(df['馬番'], errors='coerce').fillna(1).astype(int)
            df['斤量_num'] = pd.to_numeric(df['斤量'], errors='coerce').fillna(55.0).astype(float)
            df['賞金_num'] = pd.to_numeric(df['賞金(万円)'], errors='coerce').fillna(0.0).astype(float)
            
            # --- 【新設】レース全体の賞金規模（1着賞金 ＝ そのレースIDでの最高獲得賞金）の計算 ---
            logger.info("Calculating race class levels based on prize money...")
            # レースごとの最大賞金を「そのレースの基準賞金」とする
            df['レース賞金'] = df.groupby('レースID')['賞金_num'].transform('max').fillna(500.0)
            
            # 特徴量生成（馬ごとにグループ化し、日付でシフト）
            logger.info("Generating sequence features including class (prize) levels...")
            df['prev_rank'] = df.groupby('馬名')['着順_num'].shift(1).fillna(8).astype(int)
            df['avg_rank'] = df.groupby('馬名')['着順_num'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(8.0)
            df['avg_last_3f'] = df.groupby('馬名')['上り_num'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(36.0)
            df['is_prev_top3'] = df['prev_rank'].apply(lambda x: 1 if x <= 3 else 0)
            
            # 【新特徴量】前走のレース賞金（レースの格）
            df['prev_prize'] = df.groupby('馬名')['レース賞金'].shift(1).fillna(500.0).astype(float)
            # 【新特徴量】クラスの上下（今回賞金 － 前走賞金。マイナスなら降級で格下戦、プラスなら昇級）
            df['prize_diff'] = df['レース賞金'] - df['prev_prize']
            # 【新特徴量】過去5走の平均賞金（主戦クラスレベル）
            df['avg_prize'] = df.groupby('馬名')['レース賞金'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean()).fillna(500.0)
            
            df['waku'] = df['枠番_num']
            df['umaban'] = df['馬番_num']
            df['weight'] = df['斤量_num']
            df['track_type'] = df['芝・ダート区分']
            df['track_condition'] = df['馬場状態1']
            
            df['odds'] = pd.to_numeric(df['単勝'], errors='coerce').fillna(99.0).astype(float)
            df['popularity'] = pd.to_numeric(df['人気'], errors='coerce').fillna(10).astype(int)
            
            df['is_winner'] = df['着順_num'].apply(lambda x: 1 if x == 1 else 0)
            df['is_place'] = df['着順_num'].apply(lambda x: 1 if x <= 3 else 0)
            
        except Exception as e:
            logger.error(f"Error parsing actual CSV: {e}. Falling back to mock data.")
            df = generate_realistic_mock_data()
            df['レース日付'] = pd.to_datetime(df['レース日付'])
            df['is_winner'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)
            df['is_place'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)
            df['is_prev_top3'] = df['prev_rank'].apply(lambda x: 1 if x <= 3 else 0)
            df['prev_prize'] = df['prev_prize_mock']
            df['prize_diff'] = df['class_prize_mock'] - df['prev_prize']
            df['avg_prize'] = df['prev_prize']
    else:
        logger.warning(f"Actual CSV {CSV_FILE_NAME} not found. Generating mock dataset.")
        df = generate_realistic_mock_data()
        df['レース日付'] = pd.to_datetime(df['レース日付'])
        df['is_winner'] = df['rank'].apply(lambda x: 1 if x == 1 else 0)
        df['is_place'] = df['rank'].apply(lambda x: 1 if x <= 3 else 0)
        df['is_prev_top3'] = df['prev_rank'].apply(lambda x: 1 if x <= 3 else 0)
        df['prev_prize'] = df['prev_prize_mock']
        df['prize_diff'] = df['class_prize_mock'] - df['prev_prize']
        df['avg_prize'] = df['prev_prize']
        
    le_track = LabelEncoder()
    df['track_type_code'] = le_track.fit_transform(df['track_type'].astype(str))
    
    le_cond = LabelEncoder()
    df['track_condition_code'] = le_cond.fit_transform(df['track_condition'].astype(str))
    
    encoders = {
        'track_type': le_track,
        'track_condition': le_cond
    }
    
    return df, encoders

def train_model():
    """
    クラス特徴量を統合した単勝・複勝モデルの学習・保存。
    """
    df, encoders = load_and_preprocess_data()
    
    logger.info(f"Total dataset size: {len(df)} rows")
    
    # 時系列分割 (2020-01-01)
    split_date = pd.to_datetime("2020-01-01")
    train_df = df[df['レース日付'] < split_date]
    test_df = df[df['レース日付'] >= split_date]
    
    logger.info(f"Train set size: {len(train_df)} rows")
    logger.info(f"Test set size: {len(test_df)} rows")
    
    X_train = train_df[FEATURES]
    y_train_win = train_df['is_winner']
    y_train_place = train_df['is_place']
    
    X_test = test_df[FEATURES]
    y_test_win = test_df['is_winner']
    y_test_place = test_df['is_place']
    
    if len(X_test) == 0:
        logger.warning("Test dataset was empty. Using random stratified split instead.")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train_win, y_test_win = train_test_split(df[FEATURES], df['is_winner'], test_size=0.2, random_state=42, stratify=df['is_winner'])
        _, _, y_train_place, y_test_place = train_test_split(df[FEATURES], df['is_place'], test_size=0.2, random_state=42, stratify=df['is_place'])
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    # 1. 単勝予測モデルの学習 (is_winner)
    logger.info("--- 1. Training Class-Aware Win Model ---")
    train_data_win = lgb.Dataset(X_train, label=y_train_win)
    test_data_win = lgb.Dataset(X_test, label=y_test_win, reference=train_data_win)
    
    model_win = lgb.train(
        params,
        train_data_win,
        num_boost_round=1000,
        valid_sets=[train_data_win, test_data_win],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    preds_win = model_win.predict(X_test, num_iteration=model_win.best_iteration)
    logloss_win = log_loss(y_test_win, preds_win)
    auc_win = roc_auc_score(y_test_win, preds_win)
    logger.info(f"Win Model AUC: {auc_win:.5f} | LogLoss: {logloss_win:.5f}")
    
    # 2. 複勝予測モデルの学習 (is_place)
    logger.info("--- 2. Training Class-Aware Place Model ---")
    train_data_place = lgb.Dataset(X_train, label=y_train_place)
    test_data_place = lgb.Dataset(X_test, label=y_test_place, reference=train_data_place)
    
    model_place = lgb.train(
        params,
        train_data_place,
        num_boost_round=1000,
        valid_sets=[train_data_place, test_data_place],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    preds_place = model_place.predict(X_test, num_iteration=model_place.best_iteration)
    logloss_place = log_loss(y_test_place, preds_place)
    auc_place = roc_auc_score(y_test_place, preds_place)
    logger.info(f"Place Model AUC: {auc_place:.5f} | LogLoss: {logloss_place:.5f}")
    
    # 特徴量の重要度を表示 (複勝モデル)
    importance = model_place.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({'feature': FEATURES, 'importance': importance})
    feature_imp = feature_imp.sort_values(by='importance', ascending=False)
    logger.info("\nFeature Importance (Gain) - Class Features Included:\n" + feature_imp.to_string(index=False))
    
    # 3. モデルの保存
    model_data_win = {
        'model': model_win,
        'features': FEATURES,
        'encoders': encoders,
        'metrics': {'log_loss': logloss_win, 'auc': auc_win}
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data_win, f)
    
    model_data_place = {
        'model': model_place,
        'features': FEATURES,
        'encoders': encoders,
        'metrics': {'log_loss': logloss_place, 'auc': auc_place}
    }
    with open(PLACE_MODEL_PATH, 'wb') as f:
        pickle.dump(model_data_place, f)
        
    logger.info(f"Successfully saved Class-Aware models.")
    return auc_win, auc_place

if __name__ == "__main__":
    train_model()
