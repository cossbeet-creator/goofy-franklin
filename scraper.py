import requests
from bs4 import BeautifulSoup
import time
import random
import re
import pandas as pd
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("horse_scraper")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_html(url, retries=3, delay=1.5):
    """
    指定したURLからHTMLを取得する（リトライとウェイト付き）
    """
    for i in range(retries):
        try:
            logger.info(f"Fetching URL: {url} (Attempt {i+1})")
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code == 200:
                time.sleep(delay + random.uniform(0, 1))
                return response.content
            else:
                logger.warning(f"Failed to fetch {url} with status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
        time.sleep(delay * 2)
    return None

def parse_shutuba_page(race_id):
    """
    出馬表ページから基本情報およびレース本賞金をパースする
    """
    url = f"https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"
    html = get_html(url)
    if not html:
        logger.error("Could not retrieve HTML from netkeiba.")
        return None, None
    
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. レース名・コース情報・本賞金の取得
    race_info = {}
    try:
        race_name_element = soup.find(class_="RaceName")
        if not race_name_element:
            race_name_element = soup.find(class_="RaceList_NameBox")
        
        race_info["race_name"] = race_name_element.get_text(strip=True) if race_name_element else f"Race ID: {race_id}"
        
        # コース条件と賞金が含まれる要素
        race_data_element = soup.find(class_="RaceData01")
        if race_data_element:
            race_info["race_data"] = race_data_element.get_text(strip=True)
        else:
            race_info["race_data"] = "不明"
            
        text = race_info["race_data"]
        dist_match = re.search(r"(芝|ダ|障)(\d+)m", text)
        race_info["track_type"] = dist_match.group(1) if dist_match else "不明"
        race_info["distance"] = int(dist_match.group(2)) if dist_match else 0
        
        surface_match = re.search(r"馬場:([良|稍|重|不])", text)
        race_info["track_condition"] = surface_match.group(1) if surface_match else "良"
        
        weather_match = re.search(r"天候:([晴|曇|雨|小雨|雪])", text)
        race_info["weather"] = weather_match.group(1) if weather_match else "晴"
        
        # 【新設】1着本賞金の抽出 (例: 本賞金: 1840, 740, 460... 万円)
        # ページ全体のテキストから本賞金パターンを抽出
        page_text = soup.get_text()
        prize_match = re.search(r"本賞金:?\s*([\d,]+)", page_text)
        if prize_match:
            # カンマを除去して数値化 (例: "18,400" -> 18400)
            prize_val = int(prize_match.group(1).replace(",", ""))
            # 万の単位に統一
            race_info["prize"] = prize_val
            logger.info(f"Parsed race prize: {prize_val} (万円)")
        else:
            # 取得できない場合のデフォルト値 (クラスに応じた推定値)
            race_info["prize"] = 1000.0
            
    except Exception as e:
        logger.error(f"Error parsing race info: {e}")
        race_info = {
            "race_name": f"Race {race_id}",
            "race_data": "パース失敗",
            "track_type": "不明",
            "distance": 0,
            "track_condition": "良",
            "weather": "晴",
            "prize": 1000.0
        }
    
    # 2. 出馬表テーブルのパース
    horses = []
    try:
        table = soup.find("table", class_=re.compile(r"Shutuba_Table"))
        if not table:
            logger.error("Shutuba_Table not found in HTML.")
            return race_info, None
            
        rows = table.find_all("tr", class_=re.compile(r"HorseList"))
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 10:
                continue
            
            waku = tds[0].get_text(strip=True)
            umaban = tds[1].get_text(strip=True)
            
            horse_td = row.find(class_="HorseName")
            horse_name = ""
            horse_id = ""
            if horse_td:
                a_tag = horse_td.find("a")
                if a_tag:
                    horse_name = a_tag.get_text(strip=True)
                    href = a_tag.get("href", "")
                    id_match = re.search(r"horse/(\d+)", href)
                    if id_match:
                        horse_id = id_match.group(1)
            
            age_td = row.find(class_="Biko")
            age = age_td.get_text(strip=True) if age_td else "不詳"
            
            weight_td = tds[5]
            weight = weight_td.get_text(strip=True) if weight_td else "0.0"
            
            jockey_td = row.find(class_="Jockey")
            jockey_name = ""
            jockey_id = ""
            if jockey_td:
                a_tag = jockey_td.find("a")
                if a_tag:
                    jockey_name = a_tag.get_text(strip=True)
                    href = a_tag.get("href", "")
                    id_match = re.search(r"jockey/(\d+)", href)
                    if id_match:
                        jockey_id = id_match.group(1)
                        
            trainer_td = row.find(class_="Trainer")
            trainer_name = ""
            if trainer_td:
                a_tag = trainer_td.find("a")
                trainer_name = a_tag.get_text(strip=True) if a_tag else trainer_td.get_text(strip=True)
            
            odds_td = row.find(class_="Odds")
            odds_str = odds_td.get_text(strip=True) if odds_td else "---"
            try:
                odds = float(odds_str) if odds_str != "---" else 999.0
            except ValueError:
                odds = 999.0
                
            pop_td = row.find(class_="Popular")
            pop_str = pop_td.get_text(strip=True) if pop_td else "---"
            try:
                pop = int(pop_str) if pop_str != "---" else 99
            except ValueError:
                pop = 99
            
            horses.append({
                "waku": waku,
                "umaban": int(umaban) if umaban.isdigit() else 0,
                "horse_id": horse_id,
                "horse_name": horse_name,
                "age_sex": age,
                "jockey_id": jockey_id,
                "jockey_name": jockey_name,
                "trainer_name": trainer_name,
                "weight": float(weight) if weight.replace('.', '', 1).isdigit() else 50.0,
                "odds": odds,
                "popularity": pop
            })
            
    except Exception as e:
        logger.error(f"Error parsing horses table: {e}")
        
    return race_info, pd.DataFrame(horses)

def parse_horse_history(horse_id):
    """
    馬の過去成績を取得する
    """
    if not horse_id:
        return []
        
    url = f"https://db.netkeiba.com/horse/{horse_id}/"
    html = get_html(url, delay=1.0)
    if not html:
        logger.error(f"Could not retrieve history for horse {horse_id}")
        return []
        
    soup = BeautifulSoup(html, "html.parser")
    history = []
    
    try:
        table = soup.find("table", class_=re.compile(r"db_h_race_results"))
        if not table:
            logger.warning(f"History table not found for horse {horse_id}")
            return []
            
        # ヘッダー列から「賞金」の列インデックスを動的に見つける (仕様変更への耐久性向上)
        thead = table.find("thead")
        headers = thead.find_all("th") if thead else table.find_all("tr")[0].find_all("td")
        prize_idx = -1
        for idx, th in enumerate(headers):
            th_text = th.get_text(strip=True)
            if "賞金" in th_text:
                prize_idx = idx
                break
        
        # 見つからない場合のデフォルト列位置 (通常は最後から4番目の付近)
        if prize_idx == -1:
            prize_idx = 27
            
        rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")[1:]
        
        for row in rows[:5]:
            tds = row.find_all("td")
            if len(tds) < 18:
                continue
                
            date = tds[0].get_text(strip=True)
            race_name = tds[4].get_text(strip=True)
            
            rank_str = tds[11].get_text(strip=True)
            try:
                rank = int(rank_str) if rank_str.isdigit() else 99
            except ValueError:
                rank = 99
                
            dist_str = tds[14].get_text(strip=True)
            condition = tds[15].get_text(strip=True)
            time_str = tds[17].get_text(strip=True)
            
            # 上り3Fの列インデックスも安全策として最後から数えて取得
            last_3f_str = tds[22].get_text(strip=True) if len(tds) > 22 else "0.0"
            try:
                last_3f = float(last_3f_str) if last_3f_str.replace('.', '', 1).isdigit() else 0.0
            except ValueError:
                last_3f = 0.0
                
            weight_str = tds[23].get_text(strip=True) if len(tds) > 23 else "480"
            
            # 【新設】過去の各レースの「本賞金」のパース
            # 該当する馬の賞金セルからデータを取得し、1着賞金を模倣 (着順が1着なら満額、そうでなければ概算値から逆算)
            prize_val = 0.0
            if len(tds) > prize_idx:
                raw_prize = tds[prize_idx].get_text(strip=True)
                try:
                    # カンマなどを除去して数値に変換 (例: "1,200.0" -> 1200.0)
                    prize_val = float(raw_prize.replace(",", "")) if raw_prize else 0.0
                except ValueError:
                    prize_val = 0.0
                    
            # 1着でなくても、獲得賞金から「レース全体のレベル（1着賞金）」を推定するロジック
            # (例：2着なら1着の40%の賞金なので、逆算して1着賞金 ＝ 獲得賞金 / 0.4 とする)
            estimated_race_prize = 500.0 # デフォルト
            if prize_val > 0:
                if rank == 1:
                    estimated_race_prize = prize_val
                elif rank == 2:
                    estimated_race_prize = prize_val / 0.4
                elif rank == 3:
                    estimated_race_prize = prize_val / 0.25
                elif rank == 4:
                    estimated_race_prize = prize_val / 0.15
                elif rank == 5:
                    estimated_race_prize = prize_val / 0.1
                else:
                    # 着外で賞金が0の場合は、オッズや人気の過去統計から1000万程度と仮定
                    estimated_race_prize = 1000.0
            
            history.append({
                "date": date,
                "race_name": race_name,
                "rank": rank,
                "distance_info": dist_str,
                "track_condition": condition,
                "time": time_str,
                "last_3f": last_3f,
                "weight_info": weight_str,
                "estimated_race_prize": estimated_race_prize # この過去レースのレベル（賞金）
            })
            
    except Exception as e:
        logger.error(f"Error parsing horse history for {horse_id}: {e}")
        
    return history

def generate_mock_data(race_id):
    """
    ダミーデータの生成
    """
    logger.info("Generating mock data for stability.")
    race_info = {
        "race_name": f"デモ特別レース (G1)",
        "race_data": "芝2000m (右) 天候:晴 馬場:良",
        "track_type": "芝",
        "distance": 2000,
        "track_condition": "良",
        "weather": "晴",
        "prize": 15000.0 # G1賞金1億5千万円
    }
    
    mock_horses = [
        {"waku": "1", "umaban": 1, "horse_id": "99001", "horse_name": "ゴールドシップダム", "age_sex": "牡4", "jockey_id": "001", "jockey_name": "ルメール", "trainer_name": "国枝栄", "weight": 57.0, "odds": 2.5, "popularity": 1},
        {"waku": "2", "umaban": 2, "horse_id": "99002", "horse_name": "ディープエフェクト", "age_sex": "牡5", "jockey_id": "002", "jockey_name": "川田将雅", "trainer_name": "中内田", "weight": 57.0, "odds": 4.1, "popularity": 2},
        {"waku": "3", "umaban": 3, "horse_id": "99003", "horse_name": "アーモンドアイズ", "age_sex": "牝4", "jockey_id": "003", "jockey_name": "戸崎圭太", "trainer_name": "手塚", "weight": 55.0, "odds": 5.8, "popularity": 3},
        {"waku": "4", "umaban": 4, "horse_id": "99004", "horse_name": "メジロクイーン", "age_sex": "牝5", "jockey_id": "004", "jockey_name": "武豊", "trainer_name": "矢作芳人", "weight": 55.0, "odds": 8.4, "popularity": 4},
        {"waku": "5", "umaban": 5, "horse_id": "99005", "horse_name": "オルフェーヴルドン", "age_sex": "牡3", "jockey_id": "005", "jockey_name": "鮫島克駿", "trainer_name": "音無", "weight": 55.0, "odds": 12.1, "popularity": 5},
        {"waku": "6", "umaban": 6, "horse_id": "99006", "horse_name": "サイレンススパート", "age_sex": "牡4", "jockey_id": "006", "jockey_name": "松山弘平", "trainer_name": "池江", "weight": 57.0, "odds": 15.3, "popularity": 6},
        {"waku": "7", "umaban": 7, "horse_id": "99007", "horse_name": "トウカイチェイサー", "age_sex": "牡6", "jockey_id": "007", "jockey_name": "横山武史", "trainer_name": "堀宣行", "weight": 57.0, "odds": 22.0, "popularity": 7},
        {"waku": "8", "umaban": 8, "horse_id": "99008", "horse_name": "ハルウララセカンド", "age_sex": "牝5", "jockey_id": "008", "jockey_name": "藤田菜七子", "trainer_name": "根本", "weight": 55.0, "odds": 99.8, "popularity": 8}
    ]
    return race_info, pd.DataFrame(mock_horses)

def get_mock_history(horse_id):
    ranks = [1, 2, 3, 4, 5, 8, 12]
    history = []
    for i in range(5):
        history.append({
            "date": f"2025/{10-i}/15",
            "race_name": f"モック特別 (OOP)",
            "rank": random.choice(ranks),
            "distance_info": "芝2000m",
            "track_condition": "良",
            "time": "2:00.1",
            "last_3f": round(random.uniform(33.5, 36.5), 1),
            "weight_info": "480(+2)",
            "estimated_race_prize": random.choice([5000.0, 10000.0, 15000.0, 2000.0]) # 仮想の過去の格
        })
    return history

def get_full_race_data(race_id, use_mock=False):
    """
    出馬表と各馬の近走データを一括で取得するメイン関数
    """
    if use_mock:
        race_info, df_horses = generate_mock_data(race_id)
        histories = {}
        for _, row in df_horses.iterrows():
            histories[row["horse_id"]] = get_mock_history(row["horse_id"])
        return race_info, df_horses, histories

    race_info, df_horses = parse_shutuba_page(race_id)
    
    if df_horses is None or len(df_horses) == 0:
        logger.warning("Scraping returned no data. Falling back to mock data.")
        return get_full_race_data(race_id, use_mock=True)
        
    histories = {}
    for idx, row in df_horses.iterrows():
        h_id = row["horse_id"]
        h_name = row["horse_name"]
        logger.info(f"Retrieving history for: {h_name} ({h_id})")
        if h_id:
            h_history = parse_horse_history(h_id)
            if not h_history:
                h_history = get_mock_history(h_id)
            histories[h_id] = h_history
        else:
            histories[h_id] = get_mock_history(h_id)
            
    return race_info, df_horses, histories

def generate_mock_pair_odds(df_horses):
    """
    ローカル開発用：出走馬の単勝オッズから擬似的な馬連・ワイドのオッズを自動生成するフォールバック処理
    """
    logger.info("Generating mock Quinella/Wide pair odds...")
    umaren_odds = {}
    wide_odds = {}
    
    if df_horses is None or len(df_horses) == 0:
        return umaren_odds, wide_odds
        
    horses = df_horses.copy()
    horses['odds'] = pd.to_numeric(horses['odds'], errors='coerce').fillna(99.0)
    horses['umaban'] = pd.to_numeric(horses['umaban'], errors='coerce').fillna(0).astype(int)
    
    umaban_list = sorted(horses['umaban'].tolist())
    odds_dict = dict(zip(horses['umaban'], horses['odds']))
    
    for i in range(len(umaban_list)):
        for j in range(i + 1, len(umaban_list)):
            u1 = umaban_list[i]
            u2 = umaban_list[j]
            if u1 == 0 or u2 == 0:
                continue
                
            odds_product = odds_dict[u1] * odds_dict[u2]
            
            # 馬連擬似オッズ: 単勝の積に 0.15~0.25 の乱数を掛ける
            val_ur = odds_product * random.uniform(0.15, 0.25)
            val_ur = max(round(val_ur, 1), 1.1)
            umaren_odds[(u1, u2)] = val_ur
            
            # ワイド擬似オッズ: 単勝の積に 0.05~0.12 の乱数（下限）
            val_w_low = odds_product * random.uniform(0.05, 0.10)
            val_w_low = max(round(val_w_low, 1), 1.1)
            val_w_high = val_w_low * random.uniform(1.3, 1.8)
            val_w_high = max(round(val_w_high, 1), val_w_low + 0.1)
            wide_odds[(u1, u2)] = [val_w_low, val_w_high]
            
    return umaren_odds, wide_odds

def get_realtime_pair_odds(race_id, df_horses):
    """
    Yahoo!競馬からリアルタイムの馬連・ワイドオッズを取得する。
    DNS名前解決エラーやネットワークエラーが発生した場合は自動的にモックデータ生成にフォールバックする。
    """
    # netkeibaIDの先頭2桁(20)を削りYahooコードに変換
    yahoo_code = str(race_id)[2:]
    
    umaren_odds = {}
    wide_odds = {}
    
    # --- 1. 馬連オッズの取得 ---
    ur_url = f"https://keiba.yahoo.co.jp/odds/ur/{yahoo_code}/"
    try:
        logger.info(f"Attempting to fetch Yahoo Quinella odds from {ur_url}")
        r = requests.get(ur_url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "html.parser")
            # Yahoo馬連テーブル (通常クラス 'oddsUrTbl' または table タグ)
            tables = soup.find_all("table")
            for t in tables:
                rows = t.find_all("tr")
                for row in rows:
                    tds = row.find_all(["td", "th"])
                    if len(tds) < 2:
                        continue
                    # 軸馬番の特定
                    axis_text = tds[0].get_text(strip=True)
                    if not axis_text.isdigit():
                        continue
                    axis_num = int(axis_text)
                    
                    # 相手馬番とオッズの抽出
                    # Yahooのレイアウトでは、軸のセルの右に相手馬番とオッズが入ったサブテーブルやtdが並ぶ
                    for td in tds[1:]:
                        # サブテーブルなどの内包するテキストを検索
                        # 通常は "2 15.4" のように馬番とオッズが入る
                        sub_text = td.get_text(strip=True)
                        # 正規表現で「馬番」と「オッズ」のペアを全抽出
                        matches = re.findall(r'(\d+)\s+([\d\.-]+)', sub_text)
                        for opp_str, odds_str in matches:
                            opp_num = int(opp_str)
                            if odds_str == "-" or odds_str == "---" or odds_str == ".":
                                continue
                            try:
                                val = float(odds_str)
                                pair = (min(axis_num, opp_num), max(axis_num, opp_num))
                                umaren_odds[pair] = val
                            except ValueError:
                                pass
            logger.info(f"Successfully parsed {len(umaren_odds)} Quinella odds from Yahoo.")
        else:
            logger.warning(f"Yahoo Quinella page returned status: {r.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch/parse Yahoo Quinella odds ({e}). Using mock fallback.")
        
    # --- 2. ワイドオッズの取得 ---
    wide_url = f"https://keiba.yahoo.co.jp/odds/wide/{yahoo_code}/"
    try:
        logger.info(f"Attempting to fetch Yahoo Wide odds from {wide_url}")
        r = requests.get(wide_url, headers=HEADERS, timeout=6)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "html.parser")
            tables = soup.find_all("table")
            for t in tables:
                rows = t.find_all("tr")
                for row in rows:
                    tds = row.find_all(["td", "th"])
                    if len(tds) < 2:
                        continue
                    axis_text = tds[0].get_text(strip=True)
                    if not axis_text.isdigit():
                        continue
                    axis_num = int(axis_text)
                    
                    for td in tds[1:]:
                        sub_text = td.get_text(strip=True)
                        # ワイドはオッズが「2.5-3.8」のようにハイフンで繋がれている
                        matches = re.findall(r'(\d+)\s+([\d\.]+)-([\d\.]+)', sub_text)
                        for opp_str, low_str, high_str in matches:
                            opp_num = int(opp_str)
                            try:
                                val_low = float(low_str)
                                val_high = float(high_str)
                                pair = (min(axis_num, opp_num), max(axis_num, opp_num))
                                wide_odds[pair] = [val_low, val_high]
                            except ValueError:
                                pass
            logger.info(f"Successfully parsed {len(wide_odds)} Wide odds from Yahoo.")
        else:
            logger.warning(f"Yahoo Wide page returned status: {r.status_code}")
    except Exception as e:
        logger.warning(f"Failed to fetch/parse Yahoo Wide odds ({e}). Using mock fallback.")
        
    # いずれかのオッズ取得が空だった場合、またはエラーだった場合はモックで補完
    if not umaren_odds or not wide_odds:
        mock_ur, mock_w = generate_mock_pair_odds(df_horses)
        if not umaren_odds:
            umaren_odds = mock_ur
        if not wide_odds:
            wide_odds = mock_w
            
    return umaren_odds, wide_odds
