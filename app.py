import streamlit as st
import tempfile
from streamlit_mic_recorder import mic_recorder

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_elevenlabs


st.set_page_config(
    page_title="AI Doctor with Vision and Voice",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Doctor with Vision and Voice")
st.write("Record your voice question and upload an image for medical observation.")


GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]


system_prompt = """You have to act as a professional doctor. What's in this image?
Do you find anything wrong with it medically? If you make a differential,
suggest some remedies. Answer like a real doctor speaking to a patient.
Keep response concise maximum two sentences."""


st.subheader("🎤 Record Your Voice Question")

audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key="recorder"
)


st.subheader("Upload Image for Diagnosis")

image_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)


analyze_button = st.button("Analyze")


if analyze_button:

    if audio is None:
        st.error("Please record your voice.")
        st.stop()

    if image_file is None:
        st.error("Please upload an image.")
        st.stop()


    with st.spinner("Processing your request..."):

        # Save recorded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio["bytes"])
            audio_path = temp_audio.name


        # Speech to text
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


        # Updated Groq Vision Model
        doctor_response = analyze_image_with_query(
            query=system_prompt + " " + speech_text,
            encoded_image=encoded_image,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=GROQ_API_KEY
        )


        audio_output_path = "doctor_response.mp3"

        text_to_speech_with_elevenlabs(
            input_text=doctor_response,
            output_filepath=audio_output_path,
            elevenlabs_api_key=ELEVENLABS_API_KEY
        )


    st.success("Analysis Complete")


    st.subheader("Patient Speech (Transcription)")
    st.write(speech_text)


    st.subheader("Doctor Response")
    st.write(doctor_response)


    st.subheader("Doctor Voice Response")
    st.audio(audio_output_path)
