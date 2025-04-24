import streamlit as st
from google.cloud import speech
from google.oauth2 import service_account
import google.generativeai as genai # Gemini API ライブラリをインポート
import json
import io

# ----- アプリのタイトル -----
st.title("🚀 会議文字起こし＆要約 (Google Cloud STT + Gemini)")
st.write("音声ファイルをアップロードし、文字起こし、話者分離整形、AIによる要約を行います。")

# ----- APIキーと認証情報の設定 (Streamlit Secrets から読み込み) -----
try:
    # Google Cloud STT 用の認証情報
    google_credentials_json_str = st.secrets["google_credentials_json"]
    google_credentials_dict = json.loads(google_credentials_json_str)
    credentials = service_account.Credentials.from_service_account_info(google_credentials_dict)
    speech_client = speech.SpeechClient(credentials=credentials)
    st.sidebar.success("Google Cloud STT 認証 OK")

    # Gemini API キー
    gemini_api_key = st.secrets["gemini_api_key"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro') # または gemini-1.5-pro-latest など
    st.sidebar.success("Gemini API 認証 OK")
    can_summarize = True # 要約機能が利用可能かどうかのフラグ

except KeyError as e:
    st.error(f"Streamlit Secrets の設定エラー: '{e}' が見つかりません。")
    st.error("Secrets に `google_credentials_json` と `gemini_api_key` が正しく設定されているか確認してください。")
    st.stop() # エラー時はここで停止
except FileNotFoundError:
     st.error("`.streamlit/secrets.toml` が見つかりません。(ローカル実行時)")
     st.stop()
except json.JSONDecodeError as e: # エラーの種類を具体的に捕捉
    st.error(f"エラー: `google_credentials_json` の形式が正しくありません (JSONDecodeError)。")
    st.error(f"エラー詳細: {e}") # json.loads が出した具体的なエラー理由が表示される
    st.error("上記デバッグエリアに表示された内容を確認し、有効な JSON 形式になっているか（{}の対応、引用符、カンマ、不要な文字やスペース、特殊な改行など）詳細に確認してください。")
    st.stop() # デバッグのためここで停止
except Exception as e:
    st.error(f"認証情報の読み込みまたはクライアント初期化中にエラーが発生しました: {e}")
    st.stop()
    can_summarize = False # 要約機能は利用不可

# ----- 音声ファイルのアップロード -----
uploaded_file = st.file_uploader(
    "文字起こししたい会議音声ファイルを選択してください",
    type=["wav", "flac", "mp3", "ogg", "m4a", "opus", "amr"]
)

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    # 事前に話者数が分かっている場合、ここで設定すると精度が向上する可能性
    # num_speakers = st.number_input("話者数を入力してください (任意)", min_value=1, max_value=10, value=2)

    if st.button("文字起こしと要約を実行"):
        with st.spinner('音声ファイルを処理し、文字起こしと要約を実行中です...'):
            try:
                # --- Google Cloud STT による文字起こし ---
                content = uploaded_file.read()
                audio = speech.RecognitionAudio(content=content)

                # 話者分離を有効にした RecognitionConfig (ネスト構造を使用)
                diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,  # 例: 最小話者数を設定 (任意ですが設定推奨)
                    max_speaker_count=6,  # 例: 最大話者数を設定 (任意ですが設定推奨)
                )

                config = speech.RecognitionConfig(
                    language_code="ja-JP",
                    enable_automatic_punctuation=True,
                    diarization_config=diarization_config, # ★★★ ネストした設定オブジェクトを渡すように変更
                    # model="telephony",
                )

                st.info("Google Cloud STT で文字起こしを実行中...")
                response = speech_client.recognize(config=config, audio=audio)

                if not response.results:
                    st.warning("音声から文字を認識できませんでした。")
                else:
                    st.success("文字起こしが完了しました。")

                    # --- 話者分離に基づいたスクリプト整形 ---
                    st.subheader("🗣️ 話者分離 整形済みスクリプト")
                    transcript_text = ""
                    current_speaker = -1 # 初期化 (話者タグは1から始まることが多い)
                    full_raw_text = "" # 要約用のプレーンテキスト

                    # 最後の結果に含まれる単語リストから話者タグを取得
                    # Note: response.results[-1] に全単語情報が含まれるとは限らない場合があるため、
                    # 本来は全 result を舐めるか、LongRunningRecognize の方が確実
                    # ここでは同期認識の最後の結果を使う簡易的な実装とする
                    if response.results[-1].alternatives[0].words:
                        for word_info in response.results[-1].alternatives[0].words:
                            if word_info.speaker_tag != current_speaker:
                                transcript_text += f"\n\n**話者 {word_info.speaker_tag}:**\n"
                                current_speaker = word_info.speaker_tag
                            transcript_text += word_info.word + " "
                            full_raw_text += word_info.word + " "
                    else: # 単語情報がない場合 (短い音声など) は、単純に結合
                        for result in response.results:
                            transcript_text += result.alternatives[0].transcript + "\n"
                            full_raw_text += result.alternatives[0].transcript + "\n"


                    st.markdown(transcript_text.strip()) # Markdownとして表示

                    # --- Gemini API による要約 ---
                    if can_summarize and full_raw_text:
                        st.subheader("📝 AIによる要約 (Gemini)")
                        st.info("Gemini API で要約を生成中...")

                        # Gemini に渡すプロンプトを作成
                        prompt = f"""
                        以下の会議書き起こしテキストを分析し、主要な議題とそれぞれの内容の要点を箇条書きで簡潔にまとめてください。

                        --- 書き起こしテキスト ---
                        {full_raw_text.strip()}
                        --- 要約 ---
                        """

                        try:
                            # Gemini API を呼び出し
                            gemini_response = gemini_model.generate_content(prompt)

                            # 要約結果を表示
                            st.markdown(gemini_response.text)
                            st.success("要約が完了しました。")

                        except Exception as e:
                            st.error(f"Gemini API での要約生成中にエラーが発生しました: {e}")
                    elif not can_summarize:
                         st.warning("Gemini API の設定に問題があるため、要約機能は利用できません。")


            except Exception as e:
                st.error(f"処理中にエラーが発生しました: {e}")