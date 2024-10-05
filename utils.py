import os
import glob
import subprocess
import tempfile

import numpy as np

import soundfile as sf
import librosa
from pydub import AudioSegment

def split_audio(input_file, segment_duration_sec, output_dir=os.path.join(os.path.dirname(__file__),"temp_audio_segments")):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    audio = AudioSegment.from_file(input_file)
    total_duration_ms = len(audio)
    segment_duration_ms = segment_duration_sec * 1000  # Convert to milliseconds
    segments = []

    # Save each segment as a separate .wav file
    for i in range(0, total_duration_ms, segment_duration_ms):
        segment = audio[i:i+segment_duration_ms]
        segment_path = os.path.join(output_dir, f"segment_{i // 1000}_{(i + segment_duration_ms) // 1000}.wav")
        segment.export(segment_path, format="wav")
        segments.append(segment_path)

    return segments


def normalize_audio(audio):
    # If the audio has multiple channels (i.e., shape (n, p)), average the channels to mix down to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Mix down to mono by averaging across channels
    
    # Normalize the waveform to have values between -1 and 1
    audio = audio / np.max(np.abs(audio))
    
    return audio

def convert_audio_to_wav(input_path):
    # Extract the file extension (e.g., .mp3, .m4a) from the input file
    file_extension = os.path.splitext(input_path)[1][1:]  # Get the file extension without the dot
    
    if not "wav" in file_extension:
        # Load the audio file using pydub
        audio = AudioSegment.from_file(input_path, format=file_extension)
        print(audio.duration_seconds)
        # Define the output .wav file path by replacing the original extension
        wav_path = os.path.splitext(input_path)[0] + ".wav"
        
        # Export audio to .wav format
        audio.export(wav_path, format="wav")
        
        return wav_path
    
    return input_path

def convert_audio_to_wav_ffmpeg(input_path):
    # Define the output .wav file path by replacing the original extension
    
    wav_path = os.path.splitext(input_path)[0] + ".wav"
    if not "wav" in input_path:
        print("Converting to .wav...")
        # Use ffmpeg to convert the input file to .wav format
        command = ['ffmpeg', '-i', input_path, wav_path]
        subprocess.run(command, check=True)
        
        return wav_path
    
    return input_path

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