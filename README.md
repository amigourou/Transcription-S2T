# Conversation transcription tool

Small tool to transcript conversations.

## Setup

```
git clone https://github.com/amigourou/Transcription-S2T.git
```
create a python venv or a conda env, activate it and install the required dependencies
```
python3.10 -m venv env_transcription
source env_transcription/bin/activate
pip install -r requirements.txt
```

Then install torch separatly here <https://pytorch.org/get-started/locally>

Install ffmpeg from the website <https://ffmpeg.org/download.html#build-windows>, unzip the build and make sure to add the /bin folder into your PATH environment variable. Check the installation :
```
ffmpeg -version
```

Create an account on HuggingFace <https://huggingface.co>, and generate a new token (Profile/Settings/Access Tokens), READ is OK.
Accept the conditions of use of the several pages : \\
<https://huggingface.co/pyannote/speaker-diarization-3.1> \\
<https://huggingface.co/pyannote/segmentation-3.0>

Add your token in <transcript.py> in the brackets line 112:
```
HUGGING_FACE_TOKEN= "YOUR_TOKEN_HERE"
```

To Add a default path, modify line 83 in transcript.py:

```
    parser = argparse.ArgumentParser(description="Transcribe audio files to text.")
    parser.add_argument(
        '--paths', 
        nargs='+',
        default=r"YOUR_DEFAULT_PATH_HERE",
        help="File paths or directories containing audio files to transcribe.")
```

## Run

```
python3 transcript.py --path YOUR_PATH1 YOUR_PATH2 --language fr --num_speakers 2 --model medium
```

It will run on GPU by default if it's available.

