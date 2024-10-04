
import argparse
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

from pyannote.audio import Pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from utils import *

HUGGING_FACE_TOKEN="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"

# Load your Whisper model and processor
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)


def transcribe_audio(input_file_path, output_file_path):
    print("Converting .m4a to .wav...")
    audio_path = convert_m4a_to_wav(input_file_path)
    # Load the diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGING_FACE_TOKEN)

    # Diarize the audio
    diarization = pipeline(audio_path)

    # Load the audio
    audio, sample_rate = load_audio(audio_path)
    audio = audio / max(abs(audio))  # Normalize the waveform

    transcription_segments = []

    # Process each segment
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time, end_time = turn.start, turn.end
        
        # Extract the audio segment for the current speaker
        audio_segment = audio[int(start_time * sample_rate):int(end_time * sample_rate)]
        
        # Convert to the format expected by the processor
        inputs = processor(audio_segment, return_tensors="pt", sampling_rate=sample_rate)

        # Perform transcription
        generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        transcription_segments.append(f"{speaker}: {transcription}")

    # Save the transcription to a file
    with open(output_file_path, "w", encoding="utf-8") as f:
        for segment in transcription_segments:
            f.write(segment + "\n")

    print(f"Transcription saved to {output_file_path}")

if __name__ == "__main__":
    # Example usage

    parser = argparse.ArgumentParser(description="Transcribe audio files to text.")
    parser.add_argument(
        'paths', 
        nargs='+',
        default=r"D:\Deep learning projects\Speech2text\test_conversation.m4a",
        help="File paths or directories containing audio files to transcribe.")
    
    args = parser.parse_args()
    audio_files = get_audio_files(args.paths)

    # input_file_path = r"D:\Deep learning projects\Speech2text\test_conversation.m4a"
    output_file_path = r"D:\Deep learning projects\Speech2text\output_conversation.txt"
    
    for input_file_path in audio_files:
        print(os.path.splitext(os.path.basename(input_file_path))[0])
        output_file_path = os.path.join(os.path.dirname(input_file_path), os.path.splitext(os.path.basename(input_file_path))[0] + "_transcript.txt")
        print(f"====Transcribing conversation {os.path.basename(input_file_path)} ....====")
        transcribe_audio(input_file_path, output_file_path)
