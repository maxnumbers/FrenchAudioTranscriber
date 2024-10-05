import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import torch
from transformers import pipeline
import tempfile
import os
import time
import threading
import queue

# Set page title and favicon
st.set_page_config(page_title="French Audio Transcription and Translation", page_icon="🎙️")

# Initialize the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def transcribe_audio(file_path, file_extension, progress_callback):
    recognizer = sr.Recognizer()
    
    if file_extension in ['.mp3', '.wav', '.opus']:
        if file_extension == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                audio.export(temp_wav.name, format='wav')
                with sr.AudioFile(temp_wav.name) as source:
                    audio_data = recognizer.record(source)
        elif file_extension == '.wav':
            with sr.AudioFile(file_path) as source:
                audio_data = recognizer.record(source)
        elif file_extension == '.opus':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                os.system(f"ffmpeg -i {file_path} -acodec pcm_s16le -ar 16000 {temp_wav.name}")
                with sr.AudioFile(temp_wav.name) as source:
                    audio_data = recognizer.record(source)
    elif file_extension == '.mp4':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
            os.system(f"ffmpeg -i {file_path} -acodec pcm_s16le -ar 16000 {temp_wav.name}")
            with sr.AudioFile(temp_wav.name) as source:
                audio_data = recognizer.record(source)
    else:
        raise ValueError("Unsupported file format")

    start_time = time.time()
    max_duration = 60  # 60 seconds timeout

    try:
        while True:
            if time.time() - start_time > max_duration:
                raise TimeoutError("Transcription process timed out")

            try:
                text = recognizer.recognize_google(audio_data, language="fr-FR")
                progress_callback(0.5)  # Update progress to 50% after transcription
                return text
            except sr.UnknownValueError:
                # Retry if the audio is not recognized
                time.sleep(1)
            except sr.RequestError as e:
                raise Exception(f"Could not request results from speech recognition service; {e}")
    except TimeoutError:
        raise Exception("Transcription process timed out. Please try again with a shorter audio file.")

def translate_text(text, progress_callback):
    translation = translator(text, max_length=1000)[0]['translation_text']
    progress_callback(1.0)  # Update progress to 100% after translation
    return translation

def process_audio(file_path, file_extension, progress_queue, cancel_event):
    try:
        progress_queue.put(0.0)
        
        def progress_callback(progress):
            progress_queue.put(progress)
        
        transcription = transcribe_audio(file_path, file_extension, progress_callback)
        if cancel_event.is_set():
            return None, None
        
        translation = translate_text(transcription, progress_callback)
        if cancel_event.is_set():
            return None, None
        
        return transcription, translation
    except Exception as e:
        progress_queue.put((-1, str(e)))
        return None, None

def main():
    st.title("🇫🇷 French Audio Transcription and Translation 🇬🇧")
    st.write("Upload a French audio file to transcribe and translate it to English.")

    uploaded_file = st.file_uploader("Choose a French audio file", type=["mp3", "mp4", "wav", "opus"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Transcribe and Translate"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            cancel_button = st.empty()

            progress_queue = queue.Queue()
            cancel_event = threading.Event()

            def update_progress():
                while True:
                    progress = progress_queue.get()
                    if isinstance(progress, tuple):
                        error_message = progress[1]
                        st.error(f"An error occurred: {error_message}")
                        break
                    elif progress == 1.0:
                        break
                    progress_bar.progress(progress)
                    status_text.text(f"Processing... {progress:.0%}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            progress_thread = threading.Thread(target=update_progress)
            progress_thread.start()

            process_thread = threading.Thread(
                target=process_audio,
                args=(temp_file_path, os.path.splitext(uploaded_file.name)[1], progress_queue, cancel_event)
            )
            process_thread.start()

            if cancel_button.button("Cancel Processing"):
                cancel_event.set()
                st.warning("Processing cancelled by user.")
                process_thread.join()
                progress_thread.join()
                st.stop()

            process_thread.join()
            progress_thread.join()

            transcription, translation = process_thread._target(*process_thread._args)

            if transcription and translation:
                st.subheader("French Transcription:")
                st.write(transcription)

                st.subheader("English Translation:")
                st.write(translation)

            # Clean up temporary file
            os.unlink(temp_file_path)

    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
