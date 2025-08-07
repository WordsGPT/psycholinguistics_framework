## Config
Por defecto del config
Chatgpt model: gpt-4o-mini-2024-07-18
Top k: 5
Temperature: 0


## Ejecucion
prepare_experiment.py(cambiar por la version de estar carpeta, que el og hace cosas raras con algunos simbolos) genera los batches de la carpeta batches y ejecutar con execute_experiment.py

## Generar resultados
Combinar resultados en un solo fichero y batches en un solo fichero con estos comandos(tarda un rato en cargar):
```
cat batches/*.jsonl >> batches/batches.jsonl
cat results/*.jsonl >> results/results.jsonl
```
Y ejecutar genereateResults.py [mode] [language] #See the script
## Resultados

Las columnas son:
- word: Palabra seleccionada
- familiarity: Respuesta del modelo
- weighted_sum: Suma ponderada del top logprobs -> no tiene sentido si devuelve decimales: En la versión con fine-tuning, el modelo responde con dos decimales. Aquí, calcular la media ponderada de los logprobs no tiene sentido desde el punto de vista técnico. Cuando el modelo devuelve 5.12, no es un único token ("5.12"), sino tres tokens: "5", ".", y "12". Sin embargo, ChatGPT solo devuelve los logprobs principales dentro de la trayectoria seleccionada. Por ejemplo, para el primer token devuelve "5", "6" y "4" (lo cual está bien), pero para el último token devuelve "12", "13" y "14", que corresponden a 5.12, 5.13 y 5.14, pero no nos da el árbol completo que incluiría los decimales para "6." y "4.".
  
- logprob: Logprob del token seleccionado


## Resumen experimentos


