import unittest
import os
from main import convert_to_wav, transcribe_audio, translate_text, process_audio, translator

class TestAudioProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_file = "test_file.opus"

    def test_convert_to_wav(self):
        wav_file = convert_to_wav(self.test_file, '.opus')
        self.assertTrue(os.path.exists(wav_file))
        self.assertTrue(wav_file.endswith('.wav'))
        os.unlink(wav_file)

    def test_transcribe_audio(self):
        wav_file = convert_to_wav(self.test_file, '.opus')
        transcription = transcribe_audio(wav_file)
        self.assertIsInstance(transcription, str)
        self.assertTrue(len(transcription) > 0)
        print(f"Test Audio Transcription: {transcription}")
        os.unlink(wav_file)

    def test_translate_text(self):
        text = "Bonjour, comment ça va? Je suis très content. C'est une belle journée."
        translation = translate_text(text, translator)
        self.assertIsInstance(translation, str)
        self.assertTrue(len(translation) > 0)
        self.assertNotEqual(text, translation)
        print(f"Test Text Translation:\nOriginal: {text}\nTranslated: {translation}")

    def test_process_audio(self):
        transcription, translation = process_audio(self.test_file, '.opus', translator)
        self.assertIsInstance(transcription, str)
        self.assertIsInstance(translation, str)
        self.assertTrue(len(transcription) > 0)
        self.assertTrue(len(translation) > 0)
        self.assertNotEqual(transcription, translation)
        print(f"Full Audio Test:\nTranscription: {transcription}\nTranslation: {translation}")

if __name__ == '__main__':
    unittest.main()