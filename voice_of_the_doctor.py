import os
from gtts import gTTS
import elevenlabs
from elevenlabs.client import ElevenLabs


def text_to_speech_with_gtts(input_text, output_filepath):
    """
    Convert doctor response text to speech using Google TTS.
    """

    language = "en"

    audio = gTTS(
        text=input_text,
        lang=language,
        slow=False
    )

    audio.save(output_filepath)

    return output_filepath


def text_to_speech_with_elevenlabs(input_text, output_filepath, elevenlabs_api_key):
    """
    Convert doctor response text to speech using ElevenLabs API.
    """

    client = ElevenLabs(api_key=elevenlabs_api_key)

    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )

    elevenlabs.save(audio, output_filepath)

    return output_filepath
