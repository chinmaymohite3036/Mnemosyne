import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"

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

def inference(prompt):
    r = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            }
        )
    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')

incoming_query = input("Ask a Question: ")
print("Thinking...")
question_embedding = create_embedding([incoming_query])[0]

# find similarities of question embedding with all chunk embeddings
#np.vstack creates 2D array from list of 1D arrays, which is required for cosine_similarity function

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

similarity_threshold = 0.55

print("Highest similarity:", similarities.max())

if similarities.max() < similarity_threshold:
    print("I couldn't find this topic in the course material.")
    exit()

top_results = 20 
candidate_indices = similarities.argsort()[::-1][0:top_results]  # Get top 20 similar chunks

# Keep only one best chunk from each video
selected_indices = []
seen_videos = set()

for idx in candidate_indices:

    video_number = df.loc[idx, "number"]

    if video_number not in seen_videos:
        selected_indices.append(idx)
        seen_videos.add(video_number)

    if len(selected_indices) == 5:
        break

new_df = df.loc[selected_indices].copy()

new_df["similarity"] = similarities[selected_indices]

print(
    new_df[
        ["number", "title", "similarity", "text"]
    ]
)

# new_df = df.loc[max_indx]
# new_df = new_df.copy()

new_df["start_time"] = new_df["start"].apply(format_time)
new_df["end_time"] = new_df["end"].apply(format_time)

prompt = f''' I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time

{new_df[["title", "number", "start_time", "end_time", "text"]].to_json(orient="records")}
--------------------------
"{incoming_query}"
User asked  this question related to the video chunks, you have to answer in human way (dont mention the above format, its just for you) where and how much content is taught in which video. Use the provided start_time and end_time values directly when giving timestamps. Do not modify or guess the timestamps and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course.
Instruuctions:
1. You must not use your general knowledge to answer the student's question. Your knowledge about the course comes ONLY from the provided course material.
2. Strictly follow: If the requested topic is not explicitly supported by the provided course material, do not mention other videos as possible sources unless those videos are present in the provided context and actually contain the relevant information.
3. Never assume that a topic was taught in a video based on the video number, title, or general knowledge.
4. If the retrieved course material does not contain the requested topic, simply state that the topic could not be found in the provided course material.
5. Do not recommend a video just because it might be generally related to the question. Recommend a video only when the provided course material contains evidence that the topic is taught there.
6. When the course material does not support an answer, do not attempt to be helpful by filling the missing information from your own knowledge.
'''

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)   

with open("response.txt", "w") as f:
    f.write(response)


# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
    
