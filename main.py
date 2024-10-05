import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import torch
from transformers import pipeline
import tempfile
import os

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

    try:
        text = recognizer.recognize_google(audio_data, language="fr-FR")
        return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from speech recognition service; {e}"

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
                    # Transcribe audio
                    transcription = transcribe_audio(temp_file_path, os.path.splitext(uploaded_file.name)[1])
                    
                    # Translate transcription
                    translation = translate_text(transcription)

                    # Display results
                    st.subheader("French Transcription:")
                    st.write(transcription)

                    st.subheader("English Translation:")
                    st.write(translation)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    # Clean up temporary file
                    os.unlink(temp_file_path)

    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")

if __name__ == "__main__":
    main()
