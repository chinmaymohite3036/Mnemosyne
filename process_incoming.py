import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

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

df = joblib.load('embeddings.joblib')

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

