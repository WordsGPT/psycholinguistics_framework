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
user_prompt = "Das Erwerbsalter eines Wortes bezieht sich auf das Alter, in dem ein Wort zum ersten Mal gelernt wurde. Genauer gesagt, wann eine Person dieses Wort zum ersten Mal verstanden hätte, wenn jemand es vor ihr verwendet hätte, auch wenn sie es noch nicht gesprochen, gelesen oder geschrieben hatte. Schätzen Sie das durchschnittliche Alter, in dem das Wort „{Nigger}” von einem deutschen Muttersprachler erworben wurde. Das Ausgabeformat muss ein JSON-Objekt sein. Beispiel: {Wort: {Nigger}, Erwerbsalter: //Erwerbsalter des Wortes in Jahren, muss zwei Dezimalstellen haben}"


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