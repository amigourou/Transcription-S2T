# Conversation transcription tool

Small tool to transcript conversations.

## Setup

```
git clone <>
cd SPEECH2TEXT
```

create a python venv or a conda env, activate it and install the required dependencies
```
python3.10 -m venv env_transcription
source env_transcription/bin/activate
pip install -r requirements.txt
```

Install ffmpeg from the website <https://ffmpeg.org/download.html#build-windows>, unzip the build and make sure to add the /bin folder into your PATH environment variable. Check the installation :
```
ffmpeg -version
```

Create an account on HuggingFace <https://huggingface.co>, and generate a new token (Profile/Settings/Access Tokens), READ is OK.
Accept the conditions of use of the several pages :
<https://huggingface.co/pyannote/speaker-diarization-3.1>
<https://huggingface.co/pyannote/segmentation-3.0>

## Run

```
python3 transcript.py path <your files or folder separated with a space, set default in script> language <default:"fr"> num_speakers <default: 2> model <default: "medium">
```

It will run on gpu by default if it's available.

