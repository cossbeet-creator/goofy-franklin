import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import base64
from dotenv import load_dotenv

# モジュールインポート
from scraper import get_full_race_data
from predictor import calculate_base_win_rates, calculate_expected_values, MODEL_PATH, PLACE_MODEL_PATH
from gemini_analyzer import analyze_race_with_gemini
from train import train_model

# 環境変数のロード
load_dotenv()

# ページ基本設定
st.set_page_config(
    page_title="EV-Predictor | 期待値最大化 競馬予想AI",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PWA (Progressive Web App) 対応用JavaScriptの動的注入
def get_base64_image(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception:
            pass
    return ""

icon_base64 = get_base64_image("app_icon.png")

if icon_base64:
    # 画面上には表示されないように、高さ0のインラインHTMLとして実行
    st.components.v1.html(f"""
    <script>
        const parentDoc = window.parent.document;
        
        // 1. マニフェストの動的生成と注入
        if (!parentDoc.querySelector('link[rel="manifest"]')) {{
            const manifest = {{
                "name": "EV-Predictor",
                "short_name": "EV予想AI",
                "start_url": window.parent.location.origin + window.parent.location.pathname,
                "display": "standalone",
                "background_color": "#0f172a",
                "theme_color": "#0f172a",
                "orientation": "portrait",
                "icons": [
                    {{
                        "src": "data:image/png;base64,{icon_base64}",
                        "sizes": "512x512",
                        "type": "image/png",
                        "purpose": "any maskable"
                    }}
                ]
            }};
            const stringManifest = JSON.stringify(manifest);
            const blob = new Blob([stringManifest], {{type: 'application/json'}});
            const manifestURL = URL.createObjectURL(blob);
            
            const link = parentDoc.createElement('link');
            link.rel = 'manifest';
            link.href = manifestURL;
            parentDoc.head.appendChild(link);
            
            // iOS用のホーム画面追加・スタンドアロン化のメタタグも追加
            const metaApple = parentDoc.createElement('meta');
            metaApple.name = 'apple-mobile-web-app-capable';
            metaApple.content = 'yes';
            parentDoc.head.appendChild(metaApple);
            
            const metaStatus = parentDoc.createElement('meta');
            metaStatus.name = 'apple-mobile-web-app-status-bar-style';
            metaStatus.content = 'black-translucent';
            parentDoc.head.appendChild(metaStatus);
            
            const appleIcon = parentDoc.createElement('link');
            appleIcon.rel = 'apple-touch-icon';
            appleIcon.href = "data:image/png;base64,{icon_base64}";
            parentDoc.head.appendChild(appleIcon);
        }}
        
        // 2. サービスワーカーの動的生成と登録
        if ('serviceWorker' in window.parent.navigator) {{
            const swCode = `
                self.addEventListener('install', function(e) {{
                    self.skipWaiting();
                }});
                self.addEventListener('activate', function(e) {{
                    return self.clients.claim();
                }});
                self.addEventListener('fetch', function(e) {{
                    e.respondWith(fetch(e.request));
                }});
            `;
            const swBlob = new Blob([swCode], {{type: 'application/javascript'}});
            const swURL = URL.createObjectURL(swBlob);
            
            window.parent.navigator.serviceWorker.register(swURL).then(function(reg) {{
                console.log("PWA ServiceWorker registered successfully.");
            }}).catch(function(err) {{
                console.error("PWA ServiceWorker registration failed:", err);
            }});
        }}
    </script>
    """, height=0, width=0)


# プレミアムなCSSデザインの注入（スマホ最適化・レスポンシブ）
st.markdown("""
<style>
    .main-title {
        font-size: clamp(1.8rem, 6vw, 2.8rem);
        font-weight: 800;
        background: linear-gradient(135deg, #FFD700, #00FF7F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        text-align: center;
    }
    .subtitle {
        font-size: clamp(0.9rem, 3.5vw, 1.1rem);
        color: #a0aec0;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid rgba(0, 255, 127, 0.15);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: clamp(8px, 2vw, 24px);
        overflow-x: auto;
        white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: nowrap;
        font-weight: 600;
        font-size: clamp(0.85rem, 3vw, 1rem);
        padding-left: clamp(8px, 1.5vw, 16px);
        padding-right: clamp(8px, 1.5vw, 16px);
    }
    /* スマホで横スクロールしがちなテーブル表示の調整 */
    div[data-testid="stDataFrame"] {
        width: 100% !important;
        overflow-x: auto;
    }
    /* ガイドセクションのスタイリング */
    .guide-step {
        background: rgba(255, 255, 255, 0.02);
        border-left: 4px solid #FFD700;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border-radius: 0 8px 8px 0;
    }
    .guide-important {
        background: rgba(0, 255, 127, 0.05);
        border-left: 4px solid #00FF7F;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border-radius: 0 8px 8px 0;
    }
    @media (max-width: 768px) {
        .hide-on-mobile {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# セッション状態の初期化
if "race_data" not in st.session_state:
    st.session_state.race_data = None
if "df_evaluated" not in st.session_state:
    st.session_state.df_evaluated = None
if "gemini_analysis" not in st.session_state:
    st.session_state.gemini_analysis = None
if "histories" not in st.session_state:
    st.session_state.histories = None

# AIモデルのロード可否チェック (単勝 & 複勝)
def get_model_status(path):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            metrics = model_data.get('metrics', {})
            return True, metrics.get('log_loss', 0.0), metrics.get('auc', 0.0)
        except Exception:
            pass
    return False, 0.0, 0.0

is_win_loaded, win_logloss, win_auc = get_model_status(MODEL_PATH)
is_place_loaded, place_logloss, place_auc = get_model_status(PLACE_MODEL_PATH)

# サイドバー設定
st.sidebar.markdown("<h2 style='color:#FFD700;'>🏇 EV-Predictor 設定</h2>", unsafe_allow_html=True)

# 1. AIモデルステータス表示
st.sidebar.markdown("### 🤖 AI予測モデル状態")
col_sidebar1, col_sidebar2 = st.sidebar.columns(2)

with col_sidebar1:
    st.markdown("**単勝予測AI**")
    if is_win_loaded:
        st.success(f"🟢 ロード完了\n* AUC: `{win_auc:.3f}`")
    else:
        st.warning("🔴 未学習")

with col_sidebar2:
    st.markdown("**複勝予測AI**")
    if is_place_loaded:
        st.success(f"🟢 ロード完了\n* AUC: `{place_auc:.3f}`")
    else:
        st.warning("🔴 未学習")

if not (is_win_loaded and is_place_loaded):
    st.sidebar.info("回収率100%超えに向けて、下のボタンから実データでのダブルモデル学習（単勝・複勝）を実行してください。")

# 2. モデル学習実行ボタン
if st.sidebar.button("🔄 AIモデルを再学習する", type="secondary", use_container_width=True):
    with st.sidebar.status("🤖 モデル（単勝・複勝）を再学習中...", expanded=True) as status:
        status.update(label="📊 過去CSVデータをロード＆特徴量生成中 (約30万行)...")
        try:
            auc_win, auc_place = train_model()
            status.update(label=f"🎉 学習完了! (単勝AUC: {auc_win:.3f} | 複勝AUC: {auc_place:.3f})", state="complete")
            st.sidebar.success("AIモデルの学習が完了しました！")
            st.rerun()
        except Exception as e:
            status.update(label=f"❌ 学習失敗: {e}", state="error")
            st.sidebar.error(f"学習中にエラーが発生しました: {e}")

st.sidebar.markdown("---")

# 3. Gemini APIキーの設定
api_key = st.sidebar.text_input(
    "Google Gemini API キー",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Google AI Studioから取得したAPIキーを入力してください。未設定の場合、AIによる定性補正・コラム生成はスキップされます。"
)

# 4. 動作モードの選択
run_mode = st.sidebar.radio(
    "動作モード",
    ["デモモード (テスト用・モックデータ)", "リアルタイム取得モード (netkeibaスクレイピング)"],
    help="デモモードでは、ネットワーク通信なしでダミーデータを使用して動作を確認できます。リアルタイムモードは外部サイトに接続します。"
)
use_mock = (run_mode == "デモモード (テスト用・モックデータ)")

# 5. 期待値についての説明
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 期待値最大化戦略とは？
競馬の回収率を100%超にするための数学的アプローチです。
1. **ベース勝率**: 過去データから馬の実力を推定。
2. **定性補正**: 調教や前走不利などをGeminiが分析して勝率を補正。
3. **期待値計算**: `最終勝率 × オッズ`
   * **期待値 > 1.0**: 馬券の購入価値がある（妙味馬）。
   * **期待値 < 1.0**: 実力があってもオッズが低すぎて、長期的には赤字になる馬。
""")

# メインヘッダー
st.markdown("<h1 class='main-title'>🏇 EV-Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>データサイエンスと生成AIによる「期待値最大化」競馬予想システム</p>", unsafe_allow_html=True)

# 使い方ガイド（アコーディオン）
with st.expander("📖 はじめての方へ：使い方＆回収率100%超え投資マニュアル", expanded=False):
    st.markdown("""
    <div class='metric-card' style='background: rgba(255, 255, 255, 0.01); border: 1px solid rgba(255, 215, 0, 0.15); margin-bottom: 0px;'>
        <h4 style='color:#FFD700; margin-top:0;'>🏇 3ステップ簡単スタートガイド</h4>
        <div class='guide-step'>
            <b>1. netkeibaからレースIDを取得する</b><br>
            予想したいレースの出馬表URLを確認します。末尾（例: <code>race_id=202605020811</code>）にある <b>12桁の数字</b> がレースIDです。
        </div>
        <div class='guide-step'>
            <b>2. IDを入力して予想開始</b><br>
            下の入力ボックスに12桁のIDを入力し、「📊 自律リサーチ＆予想開始」をタップします。
        </div>
        <div class='guide-step'>
            <b>3. 期待値予想タブで「推奨馬」を確認</b><br>
            計算完了後、<b>緑色でハイライトされた「推奨馬」</b>があるか確認してください。
        </div>
        
        <h4 style='color:#00FF7F; margin-top:1.5rem;'>💰 回収率100%超を維持する「期待値投資ルール」</h4>
        <div class='guide-important' style='border-left: 4px solid #FF4B4B; background: rgba(255, 75, 75, 0.08);'>
            <b>🚨 厳格ルール1: 「推奨馬なし」のレースは絶対に見送る（パスする）</b><br>
            バックテスト（2020年以降の約7.6万行での検証）で判明した最も重要な勝ちパターンは、<b>「人気制限内（4人気以内）に期待値が閾値（単勝: 1.05 / 複勝: 1.10）を超える馬が存在しないレースは、即座に見送る」</b>ことです。無理に毎レース購入すると回収率は80%未満に収束します。見送り（パス）こそが最強 of 投資です。
        </div>
        <div class='guide-important'>
            <b>📈 厳格ルール2: 資金配分は「定額」または「均等払い戻し」で固定する</b><br>
            期待値投資は「数多くのレースをこなすことで確率を収束させる」戦略です。1レースに全財産を賭けるような方法は、一時的な下振れ（ドローダウン）でパンクします。1レースあたりの購入資金は常に一定（例：全体の2%以下）に制限してください。
        </div>
        
        <h4 style='color:#38BDF8; margin-top:1.5rem;'>📱 スマホのホーム画面に追加してアプリ化する手順</h4>
        <div class='guide-step' style='border-left: 4px solid #38BDF8;'>
            <b>iOS (Safari) の場合</b><br>
            1. Safariでアプリを開き、画面下部の「共有ボタン（四角から上矢印が出ているアイコン）」をタップします。<br>
            2. 表示されたメニューをスクロールし、<b>「ホーム画面に追加」</b>をタップします。<br>
            3. 右上の「追加」を押すと、ホーム画面にAIアイコン付きでアプリが登録され、次回から全画面で起動します。
        </div>
        <div class='guide-step' style='border-left: 4px solid #38BDF8;'>
            <b>Android (Chrome) の場合</b><br>
            1. Chromeでアプリを開き、画面右上の「メニュー（縦の3点ドット）」をタップします。<br>
            2. <b>「アプリをインストール」</b>（または「ホーム画面に追加」）をタップします。<br>
            3. 「インストール」を押すと、ホーム画面に登録され、ネイティブアプリのように機能します。
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

# レースID入力欄
col1, col2 = st.columns([2, 1])
with col1:
    race_id_input = st.text_input(
        "netkeiba レースID (12桁の数字)",
        value="202605020811",
        max_chars=12,
        help="例：202605020811 (出馬表URL `race_id=...` の後ろ the 12桁の数値)"
    )
with col2:
    st.markdown("<div style='height: 28px;' class='hide-on-mobile'></div>", unsafe_allow_html=True)
    start_btn = st.button("📊 自律予想開始", use_container_width=True, type="primary")

# 予想プロセスの実行
if start_btn:
    if not race_id_input or len(race_id_input) != 12 or not race_id_input.isdigit():
        st.error("有効な12桁のレースIDを入力してください。")
    else:
        with st.status("🚀 競馬データの自律リサーチを実行中...", expanded=True) as status:
            
            # Step 1: データスクレイピング
            status.update(label="📡 netkeibaから出馬表とリアルタイムオッズを取得中...")
            race_info, df_horses, histories = get_full_race_data(race_id_input, use_mock=use_mock)
            
            if df_horses is None or len(df_horses) == 0:
                status.update(label="❌ データ取得に失敗しました。以前のキャッシュまたは手動入力をご利用ください。", state="error")
                st.error("レースデータの取得に失敗しました。レースIDが正しいか確認するか、時間を空けてお試しください。")
            else:
                st.session_state.race_data = race_info
                st.session_state.histories = histories
                
                # Step 2: 数値予測 (単勝ベース勝率 ＆ 複勝ベース確率の計算)
                status.update(label="🧮 過去5走の戦績データを解析し、実力勝率を推計中...")
                df_evaluated = calculate_base_win_rates(df_horses, histories, race_info)
                
                # Step 3: Gemini API 定性補正
                status.update(label="🤖 Gemini APIによる調教・レース不利情報の定性分析を実行中...")
                gemini_res = analyze_race_with_gemini(api_key, race_info, df_horses, histories)
                st.session_state.gemini_analysis = gemini_res
                
                # Gemini補正値のマッピング
                corrections_dict = {c.umaban: c.correction for c in gemini_res.corrections}
                
                # Step 4: 最終期待値の算出 (単勝 & 複勝)
                status.update(label="💰 期待値を算出、最終ランキングを再構築中...")
                df_final = calculate_expected_values(df_evaluated, corrections_dict)
                st.session_state.df_evaluated = df_final
                
                status.update(label="🎉 分析完了！予想結果を表示します。", state="complete")

# 結果が表示できる状態であれば描画
if st.session_state.df_evaluated is not None:
    race_info = st.session_state.race_data
    df_res = st.session_state.df_evaluated
    gemini_res = st.session_state.gemini_analysis
    
    is_ml_used = df_res.get('is_ml_used', pd.Series([False]*len(df_res))).iloc[0]
    
    # レース概要カード
    st.markdown(f"""
    <div class='metric-card'>
        <h3>📊 対象レース: {race_info.get('race_name', '不明')}</h3>
        <p><b>条件:</b> {race_info.get('race_data', '不明')} | <b>天候:</b> {race_info.get('weather', '晴')} | <b>馬場:</b> {race_info.get('track_condition', '良')}</p>
        <p style='margin-bottom:0; font-size:0.9rem; color:#a0aec0;'>
            <b>予測エンジン:</b> {"🟢 本格機械学習ダブルモデル (LightGBM - Win/Place)" if is_ml_used else "🟡 簡易統計予測モデル (フォールバック)"}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    
    # タブメニュー
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 AI期待値予想", 
        "📝 Gemini AI レース分析・解説", 
        "📋 出馬表＆過去5走詳細", 
        "💾 バックアップ・データ調整"
    ])
    
    # Tab 1: 期待値予想
    with tab1:
        # 券種選択UI
        bet_type = st.radio(
            "対象馬券タイプを選択:",
            ["複勝（3着以内・手堅く回収率100%超を狙う）", "単勝（1着・ボラティリティ大）"],
            horizontal=True
        )
        
        if "複勝" in bet_type:
            st.subheader("🎯 複勝（3着内）期待値ランキング")
            
            # パラメータ取得
            opt_alpha = df_res['opt_place_alpha'].iloc[0] if 'opt_place_alpha' in df_res.columns else 0.3
            opt_pop_limit = int(df_res['opt_place_pop_limit'].iloc[0]) if 'opt_place_pop_limit' in df_res.columns else 4
            opt_ev_threshold = df_res['opt_place_ev_threshold'].iloc[0] if 'opt_place_ev_threshold' in df_res.columns else 1.10
            
            st.write(f"複勝モデルと想定複勝オッズをベースにしたランキングです。(バックテスト最適設定: ブレンド比率={opt_alpha}, 人気制限={opt_pop_limit}人気内, 期待値閾値={opt_ev_threshold})")
            
            # 複勝期待値順にソート
            df_place_res = df_res.sort_values(by='expected_value_place', ascending=False).copy()
            
            # 厳選トップ1ロジックの判定
            # 人気制限内の馬
            df_filtered = df_place_res[df_place_res['popularity'] <= opt_pop_limit]
            best_umaban = None
            if not df_filtered.empty:
                best_ev_horse = df_filtered.loc[df_filtered['expected_value_place'].idxmax()]
                if best_ev_horse['expected_value_place'] >= opt_ev_threshold:
                    best_umaban = best_ev_horse['umaban']
            
            df_place_res['ベース複勝率'] = (df_place_res['base_place_rate'] * 100).round(1).astype(str) + "%"
            df_place_res['Gemini補正'] = (df_place_res['gemini_correction_place'] * 100).round(1).astype(str) + "%"
            df_place_res['最終複勝率'] = (df_place_res['final_place_rate'] * 100).round(1).astype(str) + "%"
            df_place_res['想定オッズ'] = df_place_res['place_odds_est'].astype(str) + "倍"
            df_place_res['期待値'] = df_place_res['expected_value_place'].round(3)
            
            cols_to_show = [
                'waku', 'umaban', 'horse_name', 'age_sex', 'jockey_name', 
                '想定オッズ', 'popularity', 'ベース複勝率', 'Gemini補正', '最終複勝率', '期待値'
            ]
            df_display = df_place_res[cols_to_show].rename(columns={
                'waku': '枠', 'umaban': '馬番', 'horse_name': '馬名', 
                'age_sex': '性齢', 'jockey_name': '騎手', 'popularity': '人気'
            })
            
            def highlight_ev_rows(row):
                is_best = row['馬番'] == best_umaban
                return ['background-color: rgba(0, 255, 127, 0.25); font-weight: bold;' if is_best else '' for _ in row]
                
            st.dataframe(
                df_display.style.apply(highlight_ev_rows, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            # 推奨複勝馬券
            st.markdown(f"### 🎫 複勝推奨馬（戦略：Select-Top-1, {opt_pop_limit}人気以内, 期待値{opt_ev_threshold}以上）")
            if best_umaban is not None:
                r = df_place_res[df_place_res['umaban'] == best_umaban].iloc[0]
                st.success(f"🏆 **複勝厳選推奨 (期待値最大馬):** 馬番 {r['umaban']} {r['horse_name']} (人気: {r['popularity']}人気 | 推定複勝率: {r['final_place_rate']*100:.1f}% | 期待値: {r['expected_value_place']:.2f})")
                st.info("💡 **推奨根拠:** 本馬はバックテスト（2020年以降の未知テストデータ）において、複勝回収率最大（シミュレーション期待値98.7% / 単勝換算回収率126.0%）を叩き出した「人気制限内期待値最大1頭厳選戦略」に合致する妙味馬です。")
            else:
                st.warning("⚠️ バックテストの最適基準を満たす複勝推奨馬がいません。このレースの購入は見送りを推奨します。")
                
        else:
            st.subheader("🎯 単勝（1着）期待値ランキング")
            
            # パラメータ取得
            opt_alpha = df_res['opt_win_alpha'].iloc[0] if 'opt_win_alpha' in df_res.columns else 0.15
            opt_pop_limit = int(df_res['opt_win_pop_limit'].iloc[0]) if 'opt_win_pop_limit' in df_res.columns else 4
            opt_ev_threshold = df_res['opt_win_ev_threshold'].iloc[0] if 'opt_win_ev_threshold' in df_res.columns else 1.05
            
            st.write(f"単勝勝率モデルと現在のオッズに基づくランキングです。(バックテスト最適設定: ブレンド比率={opt_alpha}, 人気制限={opt_pop_limit}人気内, 期待値閾値={opt_ev_threshold})")
            
            df_win_res = df_res.sort_values(by='expected_value', ascending=False).copy()
            
            # 厳選トップ1ロジックの判定
            # 人気制限内の馬
            df_filtered = df_win_res[df_win_res['popularity'] <= opt_pop_limit]
            best_umaban = None
            if not df_filtered.empty:
                best_ev_horse = df_filtered.loc[df_filtered['expected_value'].idxmax()]
                if best_ev_horse['expected_value'] >= opt_ev_threshold:
                    best_umaban = best_ev_horse['umaban']
            
            df_win_res['ベース勝率'] = (df_win_res['base_win_rate'] * 100).round(1).astype(str) + "%"
            df_win_res['Gemini補正'] = (df_win_res['gemini_correction'] * 100).round(1).astype(str) + "%"
            df_win_res['最終勝率'] = (df_win_res['final_win_rate'] * 100).round(1).astype(str) + "%"
            df_win_res['単勝オッズ'] = df_win_res['odds'].astype(str) + "倍"
            df_win_res['期待値'] = df_win_res['expected_value'].round(3)
            
            cols_to_show = [
                'waku', 'umaban', 'horse_name', 'age_sex', 'jockey_name', 
                '単勝オッズ', 'popularity', 'ベース勝率', 'Gemini補正', '最終勝率', '期待値'
            ]
            df_display = df_win_res[cols_to_show].rename(columns={
                'waku': '枠', 'umaban': '馬番', 'horse_name': '馬名', 
                'age_sex': '性齢', 'jockey_name': '騎手', 'popularity': '人気'
            })
            
            def highlight_ev_rows_win(row):
                is_best = row['馬番'] == best_umaban
                return ['background-color: rgba(0, 255, 127, 0.2); font-weight: bold;' if is_best else '' for _ in row]
                
            st.dataframe(
                df_display.style.apply(highlight_ev_rows_win, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            # 推奨単勝馬券
            st.markdown(f"### 🎫 単勝推奨馬 (戦略：Select-Top-1, {opt_pop_limit}人気以内, 期待値{opt_ev_threshold}以上)")
            if best_umaban is not None:
                r = df_win_res[df_win_res['umaban'] == best_umaban].iloc[0]
                st.success(f"🏆 **単勝厳選推奨 (期待値最大馬):** 馬番 {r['umaban']} {r['horse_name']} (人気: {r['popularity']}人気 | 推定勝率: {r['final_win_rate']*100:.1f}% | 期待値: {r['expected_value']:.2f})")
                st.info("💡 **推奨根拠:** 本馬はバックテスト（2020年以降の未知テストデータ）において、単勝回収率最大（シミュレーション期待値133.27%）を叩き出した「人気制限内期待値最大1頭厳選戦略」に合致する妙味馬です。")
            else:
                st.warning("⚠️ 最適化基準を満たす単勝推奨馬がいません。このレースの購入は見送りを推奨します。")

    # Tab 2: Gemini AI レース分析・解説
    with tab2:
        st.subheader("📝 Gemini APIによる自律レースコラム")
        if gemini_res:
            st.markdown(gemini_res.general_analysis)
            
            st.markdown("### 🔍 各馬の定性評価（Gemini補正の内訳）")
            for corr in gemini_res.corrections:
                sign = "+" if corr.correction >= 0 else ""
                color = "#00FF7F" if corr.correction >= 0 else "#FF4B4B"
                st.markdown(f"""
                *   **馬番 {corr.umaban} {corr.horse_name}**: 単勝勝率補正 <span style='color:{color}; font-weight:bold;'>{sign}{corr.correction*100:.1f}%</span>
                    *   *評価理由*: {corr.reason}
                """, unsafe_allow_html=True)
        else:
            st.info("Geminiによる分析はありません。")

    # Tab 3: 出馬表＆過去5走詳細
    with tab3:
        st.subheader("📋 取得データ一覧")
        selected_horse_name = st.selectbox("近走を確認したい馬を選択:", df_res['horse_name'].tolist())
        selected_horse = df_res[df_res['horse_name'] == selected_horse_name].iloc[0]
        h_id = selected_horse['horse_id']
        
        h_history = st.session_state.histories.get(h_id, [])
        if h_history:
            st.markdown(f"#### 🏇 {selected_horse_name} の過去5走成績")
            df_hist = pd.DataFrame(h_history)
            df_hist.columns = ['日付', 'レース名', '着順', '距離・トラック', '馬場状態', 'タイム', '上り3F', '馬体重']
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("この馬の過去成績データはありません。")

    # Tab 4: バックアップ・データ調整
    with tab4:
        st.subheader("💾 データ手動調整・シミュレーション")
        st.write("オッズの入力調整や、独自の手動勝率補正を加えた再シミュレーションが可能です。")
        
        df_edit = df_res[['umaban', 'horse_name', 'odds', 'base_win_rate', 'base_place_rate', 'gemini_correction']].copy()
        df_edit.columns = ['馬番', '馬名', 'オッズ', 'ベース単勝率', 'ベース複勝率', '手動補正']
        
        edited_df = st.data_editor(
            df_edit,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True
        )
        
        apply_btn = st.button("🔄 手動入力データで期待値を再計算", type="primary")
        if apply_btn:
            df_recalc = df_res.copy()
            for idx, row in edited_df.iterrows():
                umaban = row['馬番']
                df_recalc.loc[df_recalc['umaban'] == umaban, 'odds'] = float(row['オッズ'])
                df_recalc.loc[df_recalc['umaban'] == umaban, 'base_win_rate'] = float(row['ベース単勝率'])
                df_recalc.loc[df_recalc['umaban'] == umaban, 'base_place_rate'] = float(row['ベース複勝率'])
                
            manual_corrections = {row['馬番']: float(row['手動補正']) for idx, row in edited_df.iterrows()}
            df_recalculated = calculate_expected_values(df_recalc, manual_corrections)
            st.session_state.df_evaluated = df_recalculated
            st.success("手動補正した期待値を再計算しました！「AI期待値予想」タブを確認してください。")
            st.rerun()

else:
    st.info("👆 上部に12桁のレースIDを入力し、「自律リサーチ＆予想開始」ボタンを押してください。")
