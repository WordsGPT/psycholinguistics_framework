import openai
import os
from dotenv import load_dotenv
from openai import OpenAI


def openai_login():
    load_dotenv("apis.env")
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

# Configura la API key directamente en la llamada
client = openai_login()

model = "gpt-4o-mini-2024-07-18"
user_prompt = "La edad de adquisición (AoA) de una palabra se refiere a la edad en la que se aprendió una palabra por primera vez. En concreto, cuándo una persona habría entendido por primera vez esa palabra si alguien la hubiera utilizado delante de ella, incluso cuando aún no la hubiera dicho, leído o escrito. Calcule la edad media de adquisición (AoA) de la palabra: \"detestar\" para un hablante nativo de español. Indique solo un número entero. Por favor, limite su respuesta a números."


response = client.chat.completions.create(
  model=model,
  messages=[
    {"role": "user", "content": user_prompt}
  ],
  temperature=0,
  logprobs=True,
  top_logprobs=5
)



print(response)
# with open("response.json", "w") as f:
#     f.write(response.to_json())
# print("Response saved to response.json")