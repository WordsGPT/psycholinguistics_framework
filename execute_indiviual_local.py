import openai
import os
from dotenv import load_dotenv
import transformers
import torch


model = "./llama3.1-8b-instruct-lora"
user_prompt = "La edad de adquisición (AoA) de una palabra se refiere a la edad en la que se aprendió una palabra por primera vez. En concreto, cuándo una persona habría entendido por primera vez esa palabra si alguien la hubiera utilizado delante de ella, incluso cuando aún no la hubiera dicho, leído o escrito. Calcule la edad media de adquisición (AoA) de la palabra: \"detestar\" para un hablante nativo de español. Indique solo un número entero. Por favor, limite su respuesta a números."

pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

messages = [
    {"role": "user", "content": user_prompt},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    do_sample=False,
    return_full_text=False,
    return_dict_in_generate=True,
    output_scores=5
)
print(outputs[0]["generated_text"])

# with open("response.json", "w") as f:
#     f.write(response.to_json())
# print("Response saved to response.json")