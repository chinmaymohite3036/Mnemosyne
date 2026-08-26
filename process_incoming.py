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

print(df.columns)

incoming_query = input("Ask a Question: ")
print("Thinking...")
question_embedding = create_embedding([incoming_query])[0]

# find similarities of question embedding with all chunk embeddings
#np.vstack creates 2D array from list of 1D arrays, which is required for cosine_similarity function

similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()

# Keyword boost for exact topic matches
# query_words = incoming_query.lower().split()

# adjusted_similarities = similarities.copy()

# for i, text in enumerate(df["text"]):
#     text_lower = text.lower()

#     # Give a small boost when important query words appear in the chunk
#     for word in query_words:
#         if len(word) > 2 and word in text_lower:
#             adjusted_similarities[i] += 0.02

similarity_threshold = 0.55

print("Highest similarity:", similarities.max())
print("\nTOP 15 MATCHES:")
for idx in similarities.argsort()[::-1][:15]:
    print(
        f"{similarities[idx]:.3f} | "
        f"Video {df.loc[idx, 'number']} | "
        f"{df.loc[idx, 'start']:.1f}s | "
        f"{df.loc[idx, 'text'][:150]}"
    )

if similarities.max() < similarity_threshold:
    print("I couldn't find this topic in the course material.")
    exit()

top_results = 50

candidate_indices = similarities.argsort()[::-1][:top_results]

selected_indices = []
video_counts = {}

for idx in candidate_indices:

    video_number = df.loc[idx, "number"]

    if video_counts.get(video_number, 0) < 3:
        selected_indices.append(idx)
        video_counts[video_number] = video_counts.get(video_number, 0) + 1

    if len(selected_indices) == 10:
        break

new_df = df.loc[selected_indices].copy()

new_df["similarity"] = similarities[selected_indices]

new_df["start_time"] = new_df["start"].apply(format_time)
new_df["end_time"] = new_df["end"].apply(format_time)

print(
    new_df[
        ["number", "title", "similarity",
         "start_time", "end_time",
         "youtube_url", "text"]
    ].to_string(index=False)
)

# new_df = df.loc[max_indx]
# new_df = new_df.copy()


prompt = f''' 
I am teaching web development in my Sigma web development course.

Here are retrieved video subtitle chunks containing:
- video title
- video number
- start time
- end time
- YouTube timestamp URL
- transcript text

{new_df[["title", "number", "start_time", "end_time", "youtube_url", "text"]].to_json(orient="records")}

--------------------------

Student's question:
"{incoming_query}"

Answer the student's question using ONLY the retrieved course material above.

Your job is to identify where the requested topic is actually taught in the course and provide the most relevant video section(s).

IMPORTANT:
A topic being mentioned is NOT the same as the topic being taught.

A chunk is evidence that a topic is taught only when the instructor actually:
- explains it,
- introduces it,
- demonstrates it,
- gives an example of it,
- or teaches one of its specific concepts.

A chunk that merely mentions a technology, topic, or term is NOT sufficient evidence that the topic is taught.

For every relevant source, provide:
- Video number and title
- Start and end timestamp
- The exact YouTube timestamp URL provided in the retrieved chunk
- A short explanation of what is taught in that section

Use the provided start_time, end_time, and youtube_url values directly.
Do not modify, calculate, guess, or recreate timestamps or YouTube URLs.

When mentioning a source, format it like:

🎥 **Video X — Video Title**
⏱️ **MM:SS – MM:SS**
[Watch this section](youtube_url)

Brief explanation of why this section is relevant.

--------------------------

STRICT RULES:

1. COURSE KNOWLEDGE ONLY
You must not use your general knowledge to answer the student's question.
Your knowledge about the course comes ONLY from the retrieved course material provided above.

2. RETRIEVED EVIDENCE ONLY
Do not claim that a topic is taught unless the retrieved chunks contain evidence supporting that claim.

3. MENTION ≠ TEACHING
If a chunk only mentions a topic, technology, or term without actually explaining, demonstrating, introducing, or teaching it, do NOT recommend that chunk as a source for learning that topic.

4. PREFER DIRECT TEACHING EVIDENCE
When multiple chunks are available, prefer the chunk where the instructor most directly explains, demonstrates, introduces, or gives an example of the requested topic.

5. AVOID OUTRO/CONCLUSION MENTIONS
Do not choose a chunk merely because the topic is mentioned near the end of a video or in a summary/conclusion.
Prefer the section where the topic is actually taught.

6. MULTIPLE CHUNKS FROM THE SAME VIDEO
When multiple chunks from the same video are provided, choose the chunk(s) that provide the strongest direct evidence for the student's question.
You may recommend multiple sections from the same video when they teach different relevant parts of the requested topic.

7. MULTI-TOPIC QUESTIONS
If the student's question contains multiple topics, evaluate each topic separately.

For example, if the question asks about:
- HTML forms
- CSS selectors
- CSS box model

identify which retrieved evidence supports each topic.

Do not ignore a requested topic merely because another topic has stronger similarity.

Only include a topic when the retrieved material actually supports it.

8. MULTIPLE RELEVANT VIDEOS
If different videos contain direct teaching evidence for different parts of the question, you may recommend multiple videos.

Do not force all requested information into one video.

9. UNSUPPORTED TOPICS
If the retrieved course material does not contain direct evidence that the requested topic is taught, say:

"I couldn't find this topic in the provided course material."

Do not guess.
Do not speculate.
Do not recommend a video merely because it appears related.
Do not suggest that a video might contain the topic.

10. NO EXTERNAL RESOURCES
If the requested topic is not supported by the course material, do not recommend external websites, videos, tutorials, documentation, or other resources.

11. DO NOT INFER FROM VIDEO NUMBER OR TITLE
Never assume that a topic was taught based on:
- video number
- video title
- technology name
- general course structure
- your general knowledge

Only the retrieved transcript evidence can establish that a topic was taught.

12. NO GENERAL-KNOWLEDGE FILLING
If the course material does not support an answer, do not fill the missing information using your own knowledge.

13. YOUTUBE URL
When you recommend a source, use the youtube_url belonging to that exact retrieved chunk.

Use the URL exactly as provided.
Do not modify, recreate, or guess the URL.

14. TIMESTAMPS
Use the provided start_time and end_time exactly.
Do not calculate or create different timestamps.

15. RESPONSE FORMAT
For each relevant section, use:

🎥 **Video X — Video Title**
⏱️ **MM:SS – MM:SS**
[Watch this section](youtube_url)

Briefly explain what the instructor teaches in this section and why it answers the student's question.

16. ONLY RECOMMEND SUPPORTED SOURCES
Only provide a YouTube link when the retrieved course material contains direct evidence that the corresponding section is relevant to the student's question.

17. UNRELATED QUESTIONS
If the student asks something unrelated to the Sigma web development course, state that you can only answer questions related to the course.

18. DO NOT OVER-RECOMMEND
Recommend only the strongest relevant sections.
Do not list every retrieved chunk simply because it contains a related word.

19. EVIDENCE OVER SIMILARITY
The similarity score determines which chunks are retrieved, but it does NOT determine which chunk should be recommended.

When selecting the final source, prioritize the actual transcript evidence and how directly the instructor teaches the requested topic.

20. BE HONEST ABOUT UNCERTAINTY
If the retrieved evidence is insufficient to establish that a topic was taught, say that it could not be found rather than making a weak recommendation.

Answer naturally and concisely. 
'''

with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)   

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)


# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])
    
