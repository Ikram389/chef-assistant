import os
import base64
import mimetypes

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# IMAGE


def image_to_text(image_path: str) -> str:

    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type is None:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    image_data_url = (
        f"data:{mime_type};base64,{encoded_image}"
    )

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Look at this kitchen/refrigerator image.

Identify food ingredients that are visibly present.

Return a simple list.

Do not guess ingredients that cannot reasonably be
identified from the image.

Example:

- chicken
- tomatoes
- eggs
- carrots
- milk
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        },
                    },
                ],
            }
        ],
        temperature=0,
        max_completion_tokens=500,
    )

    return response.choices[0].message.content


# AUDIO


def audio_to_text(audio_path: str) -> str:

    with open(audio_path, "rb") as audio_file:

        transcription = groq_client.audio.transcriptions.create(
            file=(audio_path, audio_file.read()),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0,
        )

    return transcription.text