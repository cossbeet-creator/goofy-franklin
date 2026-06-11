import os
import json
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import google.generativeai as genai

logger = logging.getLogger("horse_gemini")

# 1. 構造化出力のためのPydanticモデル定義
class HorseCorrection(BaseModel):
    umaban: int = Field(description="馬番")
    horse_name: str = Field(description="馬名")
    correction: float = Field(description="勝率の補正値。調教が良い、コース適性がある、前走不利があった場合はプラス（例: 0.01〜0.05）、その逆はマイナス（例: -0.01〜-0.05）とします。範囲は -0.10 から +0.10 の間とします。")
    reason: str = Field(description="この補正を行った具体的な理由。調教、脚質、コース適性、前走の不利などの定性データに基づきます。")

class RaceAnalysis(BaseModel):
    corrections: List[HorseCorrection] = Field(description="各馬の勝率補正リスト")
    general_analysis: str = Field(description="レース全体の展開予想、ペース予想、および期待値が高いと思われる推奨馬の見解などをまとめた詳細な日本語解説コラム（Markdown形式）")

def build_prompt(race_info: Dict[str, Any], df_horses: Any, histories: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    出馬表と近走履歴からGeminiへの分析用プロンプトを構築する
    """
    prompt = f"### レース情報\n"
    prompt += f"レース名: {race_info.get('race_name', '不明')}\n"
    prompt += f"条件: {race_info.get('race_data', '不明')}\n"
    prompt += f"天候/馬場: {race_info.get('weather', '晴')} / {race_info.get('track_condition', '良')}\n\n"
    
    prompt += "### 出走馬および近走成績データ\n"
    for idx, row in df_horses.iterrows():
        h_id = row['horse_id']
        h_name = row['horse_name']
        umaban = row['umaban']
        waku = row['waku']
        age_sex = row['age_sex']
        jockey = row['jockey_name']
        odds = row['odds']
        pop = row['popularity']
        
        prompt += f"--- 馬番 {umaban} (枠{waku}): {h_name} ({age_sex}) ---\n"
        prompt += f" 騎手: {jockey} | 前日/現在のオッズ: {odds}倍 ({pop}人気)\n"
        
        # 近走成績
        history_list = histories.get(h_id, [])
        if history_list:
            prompt += " [近走成績]\n"
            for i, h in enumerate(history_list):
                prompt += f"  前走-{i+1}: {h['date']} | {h['race_name']} | {h['distance_info']} ({h['track_condition']}) | {h['rank']}着 | タイム: {h['time']} | 上り3F: {h['last_3f']} | 馬体重: {h['weight_info']}\n"
        else:
            prompt += " [近走成績] データなし\n"
        prompt += "\n"
        
    prompt += "### 依頼事項\n"
    prompt += "1. 上記の各馬の近走データ、騎手、現在のオッズを分析してください。\n"
    prompt += "2. 各馬について、数値データだけでは測りきれない定性的な要素（『近走は着順が悪いが上りタイムは優秀で展開次第で見直せる』『今回は得意な距離に戻る』『調教コメントが良さそうであると仮定できる要素』など）を考慮し、ベース勝率への加減算としての補正値（-0.10 〜 +0.10）を割り当ててください。\n"
    prompt += "3. レース全体の展開予想（逃げ・先行馬の有利不利、ペース配分など）と、最も期待値が高いと思われる馬の見解について、説得力のある解説文を「general_analysis」に作成してください。\n"
    
    return prompt

def analyze_race_with_gemini(api_key: str, race_info: Dict[str, Any], df_horses: Any, histories: Dict[str, List[Dict[str, Any]]]) -> RaceAnalysis:
    """
    Gemini APIを叩いて、レースの定性分析と期待値補正データ（構造化JSON）を取得する
    """
    if not api_key:
        logger.error("Gemini API Key is not set.")
        # ダミーの補正データを返す
        return generate_default_analysis(df_horses)
        
    try:
        genai.configure(api_key=api_key)
        
        # 安定性とレスポンス向上のため gemini-1.5-flash を使用
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = build_prompt(race_info, df_horses, histories)
        
        logger.info("Sending request to Gemini API...")
        # Structured Outputs (構造化出力) の指定
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RaceAnalysis,
                temperature=0.2 # 決定論的で客観的な分析をさせるために低めに設定
            )
        )
        
        # JSONレスポンスのパース
        res_data = json.loads(response.text)
        
        # Pydanticモデルへパースして返却
        analysis = RaceAnalysis(**res_data)
        logger.info("Successfully parsed Gemini API response.")
        return analysis
        
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        # APIコール失敗時はフォールバックとしてデフォルトの分析を返す
        return generate_default_analysis(df_horses)

def generate_default_analysis(df_horses: Any) -> RaceAnalysis:
    """
    API未設定やエラー時のためのデフォルト分析データを生成する（モック）
    """
    corrections = []
    for idx, row in df_horses.iterrows():
        # デフォルトは補正なし (0.0)
        corrections.append(HorseCorrection(
            umaban=row['umaban'],
            horse_name=row['horse_name'],
            correction=0.0,
            reason="（システム制限またはAPIキー未設定のため、定性補正は適用されていません）"
        ))
        
    general_analysis = """
### レース展開予想 (簡易版)
APIキーが未設定、または呼び出し制限によりGeminiによる詳細な展開予想コラムが生成できませんでした。
現在、数値予測エンジンのみで期待値を計算しています。

**アドバイス:**
* 期待値が1.0以上の馬は、ベースとなる勝率（オッズ支持率＋近走成績）に対して現在のオッズが割高（妙味がある）と判断された馬です。
* 本格的な解説を得るには、サイドバーから **Gemini APIキー** を設定して再度解析を実行してください。
"""
    return RaceAnalysis(corrections=corrections, general_analysis=general_analysis)
