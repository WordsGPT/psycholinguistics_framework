# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv('apis.env')
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

def generate():


    model = "gemini-2.5-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Hola"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        # no thinking
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_logprobs=True,
        logprobs=5
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    print(response)
    print(response.text)

batches = client.batches.list()
for b in batches:
    print(b.id, b.state)


if __name__ == "__main__":
    generate()
