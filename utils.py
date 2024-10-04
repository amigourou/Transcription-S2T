import os
import glob

import soundfile as sf
import librosa
from pydub import AudioSegment


def convert_m4a_to_wav(input_path):
    # Load the .m4a file using pydub and convert to .wav
    audio = AudioSegment.from_file(input_path, format="m4a")
    
    # Define the temporary .wav file path
    wav_path = input_path.replace(".m4a", ".wav")
    
    # Export audio to .wav format
    audio.export(wav_path, format="wav")
    
    return wav_path

def get_audio_files(input_paths):
    audio_files = []
    for path in input_paths:
        if os.path.isdir(path):  # Check if it's a directory
            audio_files.extend(glob.glob(os.path.join(path, "*.m4a")))  # Adjust the extension as needed
        elif os.path.isfile(path):  # Check if it's a single file
            audio_files.append(path)
    return audio_files

def load_audio(audio_path, target_sample_rate=16000):
    waveform, sample_rate = sf.read(audio_path)
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate