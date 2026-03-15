import streamlit as st
import tempfile
import os

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs


st.set_page_config(
    page_title="AI Doctor with Vision and Voice",
    page_icon="🩺",
    layout="centered"
)


st.title("🩺 AI Doctor with Vision and Voice")
st.write("Upload your voice question and an image for medical observation.")


# Get API keys from Streamlit Secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]


system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose.
What's in this image?. Do you find anything wrong with it medically?
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
Keep your answer concise max two sentences. No preamble start your answer right away please
"""


st.subheader("Upload Patient Voice")

audio_file = st.file_uploader(
    "Upload your voice question",
    type=["mp3", "wav", "m4a"]
)


st.subheader("Upload Image for Diagnosis")

image_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)


if st.button("Analyze"):

    if audio_file is None:
        st.error("Please upload a voice file.")
        st.stop()

    if image_file is None:
        st.error("Please upload an image.")
        st.stop()


    with st.spinner("Processing your request..."):

        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name


        # Speech to Text
        speech_text = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath=audio_path,
            groq_api_key=GROQ_API_KEY
        )


        # Save uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image.write(image_file.read())
            image_path = temp_image.name


        encoded_image = encode_image(image_path)


        # AI Doctor Analysis
        doctor_response = analyze_image_with_query(
            query=system_prompt + " " + speech_text,
            encoded_image=encoded_image,
            model="llama-3.2-11b-vision-preview",
            groq_api_key=GROQ_API_KEY
        )


        # Convert doctor response to speech
        audio_output_path = "doctor_response.mp3"

        text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath=audio_output_path,
            elevenlabs_api_key=ELEVENLABS_API_KEY
        )


    st.success("Analysis Complete")


    st.subheader("Patient Speech (Transcription)")
    st.write(speech_text)


    st.subheader("Doctor's Response")
    st.write(doctor_response)


    st.subheader("Doctor Voice Response")
    st.audio(audio_output_path)
