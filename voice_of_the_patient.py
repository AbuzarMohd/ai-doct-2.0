import os
from groq import Groq


def transcribe_with_groq(stt_model, audio_filepath, groq_api_key):
    """
    Transcribes patient speech using Groq Whisper model.

    Args:
        stt_model (str): Whisper model name.
        audio_filepath (str): Path to uploaded audio file.
        groq_api_key (str): Groq API key.

    Returns:
        str: Transcribed text from patient speech.
    """

    client = Groq(api_key=groq_api_key)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text
