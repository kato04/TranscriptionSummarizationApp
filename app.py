import streamlit as st
from google.cloud import speech
from google.cloud import storage # GCS クライアントを追加
from google.oauth2 import service_account
import google.generativeai as genai
import json
import io
import os # ファイル名を扱うために追加
import time # 処理待ちのために追加 (より高度な待機方法もある)

# ----- GCS バケット名を設定 -----
# 重要: このバケット名は、あなたの GCP プロジェクト内に実際に存在する GCS バケットの名前に置き換えてください！
GCS_BUCKET_NAME = "transcriptionsummarizationapp"

# ----- アプリのタイトル -----
st.title("🚀 会議文字起こし＆要約アプリ (Google Cloud STT + Gemini)")
st.write("音声ファイルをアップロードし、文字起こし、話者分離整形、AIによる要約を行います。")
st.caption(f"一時ファイルは Google Cloud Storage バケット '{GCS_BUCKET_NAME}' にアップロードされます。")

# ----- APIキーと認証情報の設定 (Streamlit Secrets から読み込み) -----
speech_client = None
storage_client = None
gemini_model = None
can_summarize = False

try:
    # Google Cloud 用の認証情報
    google_credentials_json_str = st.secrets["google_credentials_json"]
    google_credentials_dict = json.loads(google_credentials_json_str)
    credentials = service_account.Credentials.from_service_account_info(google_credentials_dict)

    # Speech-to-Text クライアント
    speech_client = speech.SpeechClient(credentials=credentials)
    st.sidebar.success("Google Cloud STT 認証 OK")

    # Google Cloud Storage クライアント
    storage_client = storage.Client(credentials=credentials)
    st.sidebar.success("Google Cloud Storage 認証 OK")

    # Gemini API キー
    gemini_api_key = st.secrets["gemini_api_key"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')
    st.sidebar.success("Gemini API 認証 OK")
    can_summarize = True

except KeyError as e:
    st.error(f"Streamlit Secrets の設定エラー: '{e}' が見つかりません。")
    st.stop()
except FileNotFoundError:
     st.error("`.streamlit/secrets.toml` が見つかりません。(ローカル実行時)")
     st.stop()
except json.JSONDecodeError as e:
    st.error(f"エラー: `google_credentials_json` の形式が正しくありません (JSONDecodeError)。")
    st.error(f"エラー詳細: {e}")
    st.stop()
except Exception as e:
    st.error(f"認証情報の読み込みまたはクライアント初期化中にエラーが発生しました: {e}")
    st.stop()

# ----- 音声ファイルのアップロード -----
uploaded_file = st.file_uploader(
    "文字起こししたい会議音声ファイルを選択してください (長尺ファイル対応)",
    type=["wav", "flac", "mp3", "ogg", "m4a", "opus", "amr"]
)

if uploaded_file is not None and speech_client and storage_client:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("文字起こしと要約を実行 (長時間対応)"):
        gcs_uri = None
        blob_name = None
        try:
            with st.spinner('音声ファイルを GCS にアップロード中...'):
                # --- GCS へのファイルアップロード ---
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                # ユニークなファイル名を生成 (例: 元のファイル名にタイムスタンプを追加)
                blob_name = f"audio_uploads/{int(time.time())}_{uploaded_file.name}"
                blob = bucket.blob(blob_name)

                # Streamlit の UploadedFile オブジェクトから GCS へアップロード
                # uploaded_file.seek(0) # ファイルポインタを先頭に戻す (必要な場合)
                blob.upload_from_file(uploaded_file)
                gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
                st.info(f"GCS にアップロード完了: {gcs_uri}")

            with st.spinner(f'Google Cloud STT で文字起こしを実行中 (非同期処理)...'):
                # --- Google Cloud STT による文字起こし (longRunningRecognize) ---
                audio = speech.RecognitionAudio(uri=gcs_uri) # GCS URI を指定

                # 話者分離を有効にした RecognitionConfig
                diarization_config = speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=2,
                    max_speaker_count=6,
                )
                config = speech.RecognitionConfig(
                    language_code="ja-JP",
                    enable_automatic_punctuation=True,
                    # diarization_config=diarization_config,
                    # encoding や sample_rate_hertz は GCS 上のファイルから自動判別されることが多い
                )

                # 非同期認識を開始
                operation = speech_client.long_running_recognize(config=config, audio=audio) # ← アンダースコア(_)に変更
                st.info("非同期文字起こし処理を開始しました。完了まで時間がかかります...")

                # オペレーションの完了を待つ (タイムアウトを設定)
                # 長い音声の場合、タイムアウトは十分に長く設定する必要がある (例: 音声の長さの半分〜同程度)
                # ここでは例として 900秒 (15分) を設定
                response = operation.result(timeout=900)
                st.success("文字起こしが完了しました。")

                if not response.results:
                    st.warning("音声から文字を認識できませんでした。")
                else:
                    # --- 話者分離に基づいたスクリプト整形 ---
                    # (前回のコードと同様のロジック - 必要なら調整)
                    st.subheader("🗣️ 話者分離 整形済みスクリプト")
                    transcript_text = ""
                    current_speaker = -1
                    full_raw_text = ""
                    # LongRunningRecognize の場合、全結果が results に含まれるはず
                    for result in response.results:
                         # 単語情報があるかチェック (短い音声だとない場合も)
                        if result.alternatives and result.alternatives[0].words:
                            for word_info in result.alternatives[0].words:
                                if word_info.speaker_tag != current_speaker:
                                    # 最初の発言以外は改行を入れる
                                    if current_speaker != -1:
                                        transcript_text += "\n\n"
                                    transcript_text += f"**話者 {word_info.speaker_tag}:**\n"
                                    current_speaker = word_info.speaker_tag
                                transcript_text += word_info.word + " "
                                full_raw_text += word_info.word + " "
                        elif result.alternatives: # 単語情報がない場合はそのまま連結
                             transcript_text += result.alternatives[0].transcript + "\n"
                             full_raw_text += result.alternatives[0].transcript + "\n"


                    st.markdown(transcript_text.strip())

                    # --- Gemini API による要約 ---
                    if can_summarize and full_raw_text and gemini_model:
                        st.subheader("📝 AIによる要約 (Gemini)")
                        with st.spinner("Gemini API で要約を生成中..."):
                            prompt = f"""
                            以下の会議書き起こしテキストを分析し、主要な議題とそれぞれの内容の要点を箇条書きで簡潔にまとめてください。

                            --- 書き起こしテキスト ---
                            {full_raw_text.strip()}
                            --- 要約 ---
                            """
                            try:
                                gemini_response = gemini_model.generate_content(prompt)
                                st.markdown(gemini_response.text)
                                st.success("要約が完了しました。")
                            except Exception as e:
                                st.error(f"Gemini API での要約生成中にエラーが発生しました: {e}")
                    elif not can_summarize:
                         st.warning("Gemini API の設定に問題があるため、要約機能は利用できません。")

        except Exception as e:
            st.error(f"処理中にエラーが発生しました: {e}")

        finally:
            # --- GCS から一時ファイルを削除 ---
            if blob_name and storage_client:
                try:
                    bucket = storage_client.bucket(GCS_BUCKET_NAME)
                    blob = bucket.blob(blob_name)
                    blob.delete()
                    st.info(f"GCS から一時ファイル {blob_name} を削除しました。")
                except Exception as e:
                    st.warning(f"GCS からの一時ファイル削除中にエラー: {e}")

# ----- (エラーハンドリング部分は省略) -----
# (元のコードのエラーハンドリングを参考にしてください)