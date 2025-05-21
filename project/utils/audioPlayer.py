import whisper
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import os
import re
import time
import torch
import chess
from tqdm import tqdm
import beepy as beep

class AudioPlayer:
    def __init__(self):
        self.modelRecorder = whisper.load_model("base")
        tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        self.tacotron2 = tacotron2.to('cuda')
        waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        waveglow = waveglow.remove_weightnorm(waveglow)
        self.waveglow = waveglow.to('cuda')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    
    # def __bip(self, duration=0.2, freq=440, fs=22050):
    #     """Plays a short beep sound."""
    #     t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    #     wave = 0.9 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    #     sd.play(wave, samplerate=fs)
    #     sd.wait()

    def __is_uci_notation(self,move):
        """
        Check if the given move is in UCI notation.
        """
        # Regular expression for UCI notation
        uci_pattern = r"^[a-h][1-8][a-h][1-8][qrbn]?$"
        return bool(re.match(uci_pattern, move))

    def __record_audio(self):
        """
        Records audio from the microphone and saves it to a file.
        """
        fs = 44100  # Sample rate
        seconds = 3  # Duration of recording
        #self.__bip()
        beep.beep(1)
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        # Display a simple timebar/progress bar
        for i in tqdm(np.linspace(0, seconds, int(seconds * 20)), desc="Recording", unit="s", smoothing=0, bar_format='{desc} |{bar}|'):
            time.sleep(1/20)
        sd.wait()  # Wait until recording is finished
        #self.__bip()
        beep.beep(1)
        write('audio.wav', fs, myrecording)

    def __transcribe_audio(self):
        """
        Transcribes the recorded audio using the Whisper model.
        Returns:
            str: The transcribed move in UCI notation.
        """
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        AUDIO_PATH = os.path.join(PROJECT_ROOT, "../audio.wav")
        result = self.modelRecorder.transcribe(AUDIO_PATH, initial_prompt= "Dictated chess moves in UCI format: four characters (letter-digit-letter-digit), optionally a fifth letter for promotion [q, r, b, n]. For example: d2d4, a7a8q, c5d6n. Moves may be spoken in different languages but should be interpreted as UCI moves.")
        # "You are a chess engine. You will receive a chess move in UCI notation. UCI notation is composed of four characters: a letter, a number, another letter and another number. " \
        # "Letters are from a to h and numbers from 1 to 8. Please respond with the move in UCI notation with numbers written in digits and no spaces. For example: [d2d4], [a5b7], etc.. " \
        # "In addition, there could be a fifth character for the promotion of a piece, that is a letter in this set [r,b,n,q]. For example: [d2d4q], [c6e7p]. The input can be said in any language, but the output should be in UCI notation.")
        #remove special characters
        result['text'] = re.sub(r'[^a-zA-Z0-9]', '', result['text'])
        # remove spaces
        result['text'] = result['text'].replace(" ", "")
        # lowercase
        result['text'] = result['text'].lower()
        if self.__is_uci_notation(result['text']):
            return result['text']
        return None
    
    def get_move(self):
        """
        Get a move from the user by recording their voice and transcribing it.
        Returns:
            str: The transcribed move in UCI notation.
        """
        self.__record_audio()
        move = self.__transcribe_audio()
        while move is None:
            text = "I didn't understand. Please try again."
            print(text)
            self.read_text(text)
            self.__record_audio()
            move = self.__transcribe_audio()
        self.read_text("Move accepted: " + move)
        return chess.Move.from_uci(move)
    
    def __produce_spectogram(self, text):
        """
        Produces a mel spectrogram from the given text using Tacotron2 and WaveGlow.
        Returns:
            torch.Tensor: The mel spectrogram.
        """
        sequences, lengths = self.utils.prepare_input_sequence([text])
        with torch.no_grad():
            mel, _, _ = self.tacotron2.infer(sequences, lengths)
        return mel
    
    def __reproduce_audio(self, mel, rate=22050):
        """
        Reproduces audio from the given mel spectrogram using WaveGlow.
        """
        with torch.no_grad():
            audio = self.waveglow.infer(mel)

        audio_numpy = audio[0].data.cpu().numpy()

        # Normalize to avoid clipping
        audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))

        sd.play(audio_numpy, samplerate=rate, blocking=True)

    def read_text(self, text):
        """
        Reads the given text using Tacotron2 and WaveGlow.
        Args:
            text (str): The text to be read.
        """
        mel = self.__produce_spectogram(text)
        self.__reproduce_audio(mel)


if __name__ == "__main__":
    # Example usage
    recorder = AudioPlayer()
    result = recorder.get_move()
    print("Transcribed move:", result)