import requests
import os
import json

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    print("STATUS:", r.status_code)

    embedding = r.json()["embeddings"]
    return embedding

# list all jsons

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)

    # gathering all the text from the chunks to create embeddings in one go
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    print("Number of chunks:", len(content["chunks"]))
    print("Number of embeddings:", len(embeddings))

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
        print(chunk)
    break

# print(my_dicts)


# a = create_embedding(["Chimo is Intelligent and Helpfull person!", "Chimo is a good person!"])
# print(a)