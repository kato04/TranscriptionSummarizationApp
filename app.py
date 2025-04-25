import streamlit as st
from google.cloud import speech
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai
from pydub import AudioSegment # pydub をインポート
import json
import io # メモリ上でファイルを扱うためにインポート
import os # ファイル名を扱うためにインポート
import time

# ----- GCS バケット名を設定 -----
# 重要: このバケット名は、あなたの GCP プロジェクト内に実際に存在する GCS バケットの名前に置き換えてください！
GCS_BUCKET_NAME = "transcriptionsummarizationapp" # ← ★★★★★ 必ず書き換えてください ★★★★★

# ----- アプリのタイトル -----
st.title("🚀 会議文字起こし＆要約アプリ (Google Cloud STT + Gemini)")
st.write("音声ファイル (M4Aなども可) をアップロードし、FLACに変換後、文字起こし、話者分離整形、AIによる要約を行います。")
st.caption(f"一時ファイルは Google Cloud Storage バケット '{GCS_BUCKET_NAME}' に FLAC 形式でアップロードされます。")

# ----- APIキーと認証情報の設定 (Streamlit Secrets から読み込み) -----
speech_client = None
storage_client = None
gemini_model = None
can_summarize = False

try:
    # Google Cloud 用の認証情報 (バックスラッシュをエスケープした形式を想定)
    google_credentials_json_str = st.secrets["google_credentials_json"]
    # ここでバックスラッシュのエスケープを元に戻す必要があるかもしれないので注意
    # google_credentials_dict = json.loads(google_credentials_json_str.replace('\\\\', '\\')) # もし \\n で設定した場合
    google_credentials_dict = json.loads(google_credentials_json_str) # \n で設定した場合 (こちらが通常)
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
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest')
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
    st.error("Secrets の内容（特に private_key の改行が \\n になっているか）を確認してください。")
    st.stop()
except Exception as e:
    st.error(f"認証情報の読み込みまたはクライアント初期化中にエラーが発生しました: {e}")
    st.stop()

# ----- 音声ファイルのアップロード -----
uploaded_file = st.file_uploader(
    "文字起こししたい会議音声ファイルを選択してください",
    type=["wav", "flac", "mp3", "ogg", "m4a", "opus", "amr"] # M4Aなども受付
)

if uploaded_file is not None and speech_client and storage_client:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("文字起こしと要約を実行"):
        gcs_uri = None
        blob_name = None
        converted_file_data = None # 変換後のデータ用
        output_format = "flac" # 出力形式

        try:
            # --- Step 1: 音声ファイルをFLACに変換 ---
            st.info(f"アップロードされたファイル ({uploaded_file.name}) を {output_format.upper()} に変換中...")
            try:
                # ファイルポインタを先頭に戻す (重要)
                uploaded_file.seek(0)
                # ファイル名から拡張子を取得 (pydubに形式を伝えるため)
                file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace('.', '')
                if not file_extension: # 拡張子がない場合はタイプから推測を試みる
                    if '/' in uploaded_file.type:
                         file_extension = uploaded_file.type.split('/')[-1]
                    else:
                        # 不明な場合はエラーにするか、デフォルトを試す (ここではエラー)
                        raise ValueError("ファイルの拡張子またはMIMEタイプから形式を特定できません。")

                # pydub で読み込み (ファイルオブジェクトと形式を指定)
                audio = AudioSegment.from_file(uploaded_file, format=file_extension)

                # FLAC形式でメモリ上のバッファにエクスポート
                flac_buffer = io.BytesIO()
                audio.export(flac_buffer, format=output_format)
                flac_buffer.seek(0) # バッファのポインタを先頭に戻す
                converted_file_data = flac_buffer # 変換後のデータを保持
                st.success(f"{output_format.upper()} への変換完了。")

            except Exception as convert_e:
                st.error(f"音声ファイルの形式変換中にエラーが発生しました: {convert_e}")
                st.error(f"対応していない形式か、環境にffmpegがインストールされていない可能性があります。")
                st.stop() # 変換に失敗したらここで停止

            # --- Step 2: 変換後のファイルを GCS にアップロード ---
            if converted_file_data:
                with st.spinner(f'変換された {output_format.upper()} ファイルを GCS にアップロード中...'):
                    bucket = storage_client.bucket(GCS_BUCKET_NAME)
                    # 元のファイル名から拡張子を除き、新しい拡張子とタイムスタンプを付与
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    blob_name = f"audio_uploads/{int(time.time())}_{base_name}.{output_format}"
                    blob = bucket.blob(blob_name)

                    # メモリバッファから GCS へアップロード
                    blob.upload_from_file(converted_file_data, content_type=f'audio/{output_format}')
                    gcs_uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
                    st.info(f"GCS にアップロード完了: {gcs_uri}")

            # --- Step 3: Google Cloud STT (long_running_recognize) ---
            if gcs_uri:
                with st.spinner(f'Google Cloud STT で文字起こしを実行中 (非同期処理)...'):
                    audio_gcs = speech.RecognitionAudio(uri=gcs_uri) # GCS URI を指定

                    # 話者分離を有効にした RecognitionConfig (FLACなのでエンコ―ディング指定不要)
                    diarization_config = speech.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        min_speaker_count=2,
                        max_speaker_count=6,
                    )
                    config = speech.RecognitionConfig(
                        language_code="ja-JP",
                        enable_automatic_punctuation=True,
                        diarization_config=diarization_config, # 話者分離を再度有効化
                    )

                    # 非同期認識を開始
                    operation = speech_client.long_running_recognize(config=config, audio=audio_gcs)
                    st.info("非同期文字起こし処理を開始しました。完了まで時間がかかります...")

                    # オペレーションの完了を待つ (タイムアウトは長めに設定)
                    # 30分程度の音声なら1800秒(30分)くらい見ておく
                    response = operation.result(timeout=1800)
                    st.success("文字起こしが完了しました。")

                    if not response.results:
                        st.warning("音声から文字を認識できませんでした。")
                    else:
                        # --- Step 4: 話者分離に基づいたスクリプト整形 ---
                        st.subheader("🗣️ 話者分離 整形済みスクリプト")
                        transcript_text = ""
                        current_speaker = -1
                        full_raw_text = ""
                        for result in response.results:
                            if result.alternatives and result.alternatives[0].words:
                                for word_info in result.alternatives[0].words:
                                    if word_info.speaker_tag != current_speaker:
                                        if current_speaker != -1:
                                            transcript_text += "\n\n"
                                        transcript_text += f"**話者 {word_info.speaker_tag}:**\n"
                                        current_speaker = word_info.speaker_tag
                                    transcript_text += word_info.word + " "
                                    full_raw_text += word_info.word + " "
                            elif result.alternatives:
                                transcript_text += result.alternatives[0].transcript + "\n"
                                full_raw_text += result.alternatives[0].transcript + "\n"
                        st.markdown(transcript_text.strip())

                        # --- Step 5: Gemini API による要約 ---
                        if can_summarize and full_raw_text and gemini_model:
                            st.subheader("📝 AIによる要約 (Gemini)")
                            with st.spinner("Gemini API で要約を生成中..."):
                                prompt = f"""
                                以下の会議書き起こしテキストを分析し、主要な議題とそれぞれの内容の要点を箇条書きで簡潔にまとめてください。
                                まずはGoogle StTによる書き起こしテキストを日本語に編集しログ形式で出力した後、その内容の要点をまとめてください。
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
            st.error(f"処理中に予期せぬエラーが発生しました: {e}")

        finally:
            # --- Step 6: GCS から一時ファイルを削除 ---
            if blob_name and storage_client:
                try:
                    st.info(f"GCS から一時ファイル {blob_name} の削除を試みます...")
                    bucket = storage_client.bucket(GCS_BUCKET_NAME)
                    blob = bucket.blob(blob_name)
                    blob.delete()
                    st.info(f"GCS から一時ファイル {blob_name} を削除しました。")
                except Exception as e:
                    st.warning(f"GCS からの一時ファイル削除中にエラー: {e}") # 削除失敗は警告にとどめる