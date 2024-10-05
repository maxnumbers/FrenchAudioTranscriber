import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import os
import time
import subprocess  

# Initialize the translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")


model_name = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

def convert_to_wav(file_path, file_extension):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        if file_extension == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
            audio.export(temp_wav.name, format='wav')
        elif file_extension in ['.opus', '.mp4']:
            subprocess.run(['ffmpeg', '-i', file_path, '-acodec', 'pcm_s16le', '-ar', '16000', '-y', temp_wav.name], 
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            return file_path  # If it's already a .wav file
        return temp_wav.name

def transcribe_audio(file_path, language="fr-FR", timeout=60):
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)

    start_time = time.time()

    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError("Transcription process timed out")

        try:
            text = recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            time.sleep(1)
        except sr.RequestError as e:
            raise Exception(f"Could not request results from speech recognition service; {e}")

def translate_text(text, translator):
    # Split the text into sentences
    sentences = text.split('.')
    translated_sentences = []

    for sentence in sentences:
        if sentence.strip():
            # Translate each sentence separately
            translation = translator(sentence.strip(), max_length=100)[0]['translation_text']
            translated_sentences.append(translation)

    # Join the translated sentences
    full_translation = '. '.join(translated_sentences)
    return full_translation

def process_audio(file_path, file_extension, translator):
    wav_file = convert_to_wav(file_path, file_extension)
    transcription = transcribe_audio(wav_file)
    translation = translate_text(transcription, translator)
    if wav_file != file_path:
        os.unlink(wav_file)
    return transcription, translation


# Streamlit-specific code
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="French Audio Transcription and Translation",
                       page_icon="🎙️")

    st.title("🇫🇷 French Audio Transcription and Translation 🇬🇧")
    st.write(
        "Upload a French audio file to transcribe and translate it to English."
    )

    uploaded_file = st.file_uploader("Choose a French audio file",
                                     type=["mp3", "mp4", "wav", "opus"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Transcribe and Translate"):
            with st.spinner("Processing audio..."):
                with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(
                            uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                try:
                    transcription, translation = process_audio(
                        temp_file_path,
                        os.path.splitext(uploaded_file.name)[1], translator)

                    st.subheader("French Transcription:")
                    st.write(transcription)

                    st.subheader("English Translation:")
                    st.write(translation)

                except TimeoutError:
                    st.error(
                        "Transcription process timed out. Please try again with a shorter audio file."
                    )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    os.unlink(temp_file_path)

    st.markdown("---")
    st.write("Created with ❤️ using Streamlit")


if __name__ == "__main__":
    run_streamlit_app()
