import base64
from groq import Groq


def encode_image(image_path):
    """
    Convert an image file to base64 encoding.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_query(query, encoded_image, model, groq_api_key):
    """
    Sends the user query and encoded image to the Groq multimodal model
    and returns the doctor's response.
    """

    client = Groq(api_key=groq_api_key)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content
