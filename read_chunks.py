import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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
        if (i==5): # read only first 5 chunks for testing
            break
    break  # read only first json for testing        

# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

# find similarities of question embedding with all chunk embeddings
#np.vstack creates 2D array from list of 1D arrays, which is required for cosine_similarity function
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
print(similarities)
top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]  # Get indices of top similar chunks
print(max_indx)
new_df = df.loc[max_indx]
print(new_df[["title", "number", "text" ]])

