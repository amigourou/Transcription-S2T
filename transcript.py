
import argparse
import os
import logging
import warnings
import torch
# Suppress all warnings
warnings.filterwarnings("ignore")

from pyannote.audio import Pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
from utils import *

def transcribe_audio(input_file_path, output_file_path, processor, num_speakers, language):
    audio_path = convert_audio_to_wav_ffmpeg(input_file_path)

    # Load the diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGING_FACE_TOKEN)
    pipeline.to(device)
    # Diarize the audio
    print("Starting pipeline...")
    logging.info(f"Starting diarization on file: {audio_path}")
    diarization = pipeline(audio_path, num_speakers=num_speakers)
    logging.info("Diarization complete!")

    print("Loading and normalizing the audio...")
    audio, sample_rate = load_audio(audio_path)
    audio = normalize_audio(audio)  # Normalize the waveform

    # Open the output file for writing at the start
    print("Transcripting...")

  

    # Process each segment as it is diarized
    current_speaker = "SPEAKER_00"
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # print("New turn...")
        
        start_time, end_time = turn.start, turn.end

        # Extract the audio segment for the current speaker
        audio_segment = audio[int(start_time * sample_rate):int(end_time * sample_rate)]

        # Convert to the format expected by the processor
        inputs = processor(audio_segment, return_tensors="pt", sampling_rate=sample_rate, language=language)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Perform transcription
        generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Write each transcription segment to the file immediately
        if current_speaker != speaker:
            transcription_segment = f"{speaker}: {transcription}"
        else:
            transcription_segment = f"{transcription}"

        current_speaker = speaker
        
        # print(transcription_segment)  # Optional: print to see live progress
        if os.path.exists(output_file_path):
            mode='a'
        else:
            mode = 'w'
        with open(output_file_path, mode, encoding="utf-8") as f:
            words = transcription_segment.split()  # Split the transcription into words
            for i in range(0, len(words), 15):  # Iterate in chunks of 15 words
                line = ' '.join(words[i:i + 15])  # Join the words back into a single line
                f.write(line + "\n")

    print(f"Transcription saved to {output_file_path}")


if __name__ == "__main__":
    # Example usage

    parser = argparse.ArgumentParser(description="Transcribe audio files to text.")
    parser.add_argument(
        'paths', 
        nargs='+',
        default=r".",
        help="File paths or directories containing audio files to transcribe.")
    
    parser.add_argument(
        '--model',  # change to named argument
        default="medium",  # set default value
        help="Model type (choose from tiny, small, medium, large)"
    )

    parser.add_argument(
        '--language',  # change to named argument
        default="fr",  # set default value
        help="The preferred language"
    )

    parser.add_argument(
        '--num_speakers',  # change to named argument
        type=int,  # ensure it's treated as an integer
        default=2,  # set default value
        help="The number of speakers"
    )
    
    args = parser.parse_args()
    audio_files = get_audio_files(args.paths)
    num_speakers = args.num_speakers
    language = args.language
    model = args.model
    

    HUGGING_FACE_TOKEN= "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"

    model_types = {
        "tiny":"openai/whisper-tiny",
        "small":"openai/whisper-small",
        "medium":"openai/whisper-medium",
        "large":"openai/whisper-large-v3",
    }
    print(model, model_types)
    assert(model in model_types), f"The model type is not supporte, chose from {model_types.keys()}"
    # Load your Whisper model and processor
    model_name = model_types[model]
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for input_file_path in audio_files:
        segments = split_audio(input_file_path, 60)
        output_file_path = os.path.join(os.path.dirname(input_file_path),
                                        os.path.splitext(os.path.basename(input_file_path))[0] + "_transcript.txt")
        print(f"====Transcribing conversation {os.path.basename(input_file_path)} ....====")
        for segment in tqdm(segments) :
            transcribe_audio(segment, output_file_path, processor, num_speakers, language)
