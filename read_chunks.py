import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    if r.status_code != 200:
        print("Ollama Error:", r.text)
        return None

    embedding = r.json()["embeddings"]
    return embedding


# list all jsons

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Processing {json_file} with {len(content['chunks'])} chunks")

    # gathering all the text from the chunks
    texts = [c['text'] for c in content['chunks']]

    # create embeddings in batches
    all_embeddings = []
    batch_size = 150

    for i in range(0, len(texts), batch_size):

        batch = texts[i:i + batch_size]

        print(f"Embedding chunks {i} to {i + len(batch) - 1}")

        embeddings = create_embedding(batch)

        if embeddings is None:
            print("Embedding failed. Stopping program.")
            break

        all_embeddings.extend(embeddings)

    print("Number of chunks:", len(content["chunks"]))
    print("Number of embeddings:", len(all_embeddings))

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = all_embeddings[i]

        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
#  Save the DataFrame to a file using joblib
joblib.dump(df, 'embeddings.joblib')


