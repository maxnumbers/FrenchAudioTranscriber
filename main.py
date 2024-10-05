import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import torch
from transformers import pipeline
import tempfile
import os
from tqdm import tqdm
import time

# Set page title and favicon
st.set_page_config(page_title="French Audio Transcription and Translation", page_icon="🎙️")

# Initialize the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def transcribe_audio(file_path, file_extension):
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
                return text
            except sr.UnknownValueError:
                # Retry if the audio is not recognized
                time.sleep(1)
            except sr.RequestError as e:
                raise Exception(f"Could not request results from speech recognition service; {e}")
    except TimeoutError:
        raise Exception("Transcription process timed out. Please try again with a shorter audio file.")

def translate_text(text):
    translation = translator(text, max_length=1000)[0]['translation_text']
    return translation

def main():
    st.title("🇫🇷 French Audio Transcription and Translation 🇬🇧")
    st.write("Upload a French audio file to transcribe and translate it to English.")

    uploaded_file = st.file_uploader("Choose a French audio file", type=["mp3", "mp4", "wav", "opus"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Transcribe and Translate"):
            with st.spinner("Processing audio..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                try:
                    progress_bar = st.progress(0)
                    for i in tqdm(range(100)):
                        if i == 25:
                            # Transcribe audio
                            transcription = transcribe_audio(temp_file_path, os.path.splitext(uploaded_file.name)[1])
                        elif i == 75:
                            # Translate transcription
                            translation = translate_text(transcription)
                        progress_bar.progress(i + 1)

                    # Display results
                    st.subheader("French Transcription:")
                    st.write(transcription)

                    st.subheader("English Translation:")
                    st.write(translation)

                except TimeoutError:
                    st.error("Transcription process timed out. Please try again with a shorter audio file.")
                except Exception as e:
                    if "Speech recognition could not understand the audio" in str(e):
                        st.error("The audio quality is too low for transcription. Please try a clearer recording.")
                    elif "Could not request results from speech recognition service" in str(e):
                        st.error("There was an issue connecting to the speech recognition service. Please try again later.")
                    else:
                        st.error(f"An unexpected error occurred: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)

        if st.button("Cancel Processing"):
            st.stop()

    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
