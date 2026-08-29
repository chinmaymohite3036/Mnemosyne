## 🧠 Mnemosyne

#### AI Teaching Assistant for Course Videos

> **Ask a question. Find where it was taught. Jump straight to the lecture.**

Mnemosyne is a **RAG-based AI teaching assistant** designed to make long course videos easier to search and learn from.

Instead of manually scrubbing through multiple lectures to find a specific concept, a student can ask a natural-language question. Mnemosyne retrieves the most relevant course material, generates an answer grounded in that material, and provides the **exact YouTube timestamp** where the concept was taught.

> **Current focus:** Sigma Web Development course
> **Current status:** RAG backend working 🚧 Frontend in development

---

### 🎯 The Problem

Long course and lecture videos make it difficult to find one specific concept.

You might know **what** you want to learn, but not **where** it was taught.

Manually searching through multiple videos can be slow and frustrating.

Mnemosyne is built around a simple idea:

> **Let the student ask the question instead of searching through the videos.**

---

### 💡 How Mnemosyne Works

The core workflow is:

```text
Student Question
       ↓
Semantic Retrieval
       ↓
Relevant Course Material
       ↓
Grounded Answer
       ↓
Video + Exact Timestamp
```

For example:

```text
Question:
"Where is the <video> tag taught?"

        ↓

Retrieved course content

        ↓

Generated answer

        ↓

Video 10
Timestamp: 02:18

        ↓

Watch directly at that moment on YouTube
```

---

## ⚙️ RAG Pipeline

The current backend follows this pipeline:

```text
Course Videos
      ↓
FFmpeg
      ↓
Audio
      ↓
Whisper large-v2
      ↓
Timestamped Transcripts
      ↓
Contextualized Chunks
      ↓
BGE-M3 Embeddings
      ↓
Cosine Similarity Retrieval
      ↓
Source-Diversified Retrieval
      ↓
Llama 3.2
      ↓
Grounded Answer
      ↓
YouTube Timestamp
```

---

## 🛠️ Tech Stack

| Component         | Technology       |
| ----------------- | ---------------- |
| Language          | Python           |
| Video → Audio     | FFmpeg           |
| Transcription     | Whisper large-v2 |
| Embeddings        | BGE-M3           |
| Embedding Runtime | Ollama           |
| Similarity Search | Scikit-learn     |
| Data Processing   | Pandas, NumPy    |
| Storage           | Joblib           |
| LLM               | Llama 3.2        |
| Video Source      | YouTube          |

---

## 📂 Project Structure

```text
Mnemosyne/
│
├── videos/
│   └── course videos
│
├── mp3/
│   └── extracted audio
│
├── json/
│   └── timestamped transcripts
│
├── video_to_mp3.py
├── mp3_to_json.py
├── preprocess_json.py
├── inference.py
│
├── prompt.txt
├── embeddings.joblib
├── requirements.txt
└── README.md
```

> File names may vary depending on the current version of the project. Check the repository for the latest structure.

---

## 🚀 Running Mnemosyne on Your Own Data

The preprocessing pipeline allows you to build a searchable knowledge base from your own course videos.

### 1. Clone the repository

```bash
git clone https://github.com/chinmaymohite3036/Mnemosyne.git
cd Mnemosyne
```

---

### 2. Install dependencies

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🎥 Step 1 — Add Your Course Videos

Place your course videos inside:

```text
videos/
```

For example:

```text
videos/
├── video1.mp4
├── video2.mp4
├── video3.mp4
└── ...
```

---

## 🎵 Step 2 — Convert Videos to Audio

Run:

```bash
python video_to_mp3.py
```

This extracts the audio from the videos and creates the corresponding MP3 files.

```text
videos/
    ↓
FFmpeg
    ↓
mp3/
```

---

## 📝 Step 3 — Transcribe the Audio

Run:

```bash
python mp3_to_json.py
```

The audio is transcribed using **Whisper large-v2**.

The resulting JSON files preserve timestamp information from the transcripts.

```text
mp3/
    ↓
Whisper large-v2
    ↓
Timestamped JSON transcripts
```

These timestamps are later used to generate direct links to the relevant point in the video.

---

## 🧩 Step 4 — Prepare Chunks and Embeddings

Run:

```bash
python preprocess_json.py
```

This processes the transcript JSON files and prepares the data for semantic retrieval.

The pipeline:

```text
Transcript
    ↓
Contextualized chunks
    ↓
BGE-M3 embeddings
    ↓
Embedding dataframe
    ↓
embeddings.joblib
```

The resulting `embeddings.joblib` file contains the processed data required during retrieval.

---

## 🔎 Step 5 — Ask Questions

Once preprocessing is complete, run the inference program:

```bash
python inference.py
```

Enter a natural-language question about the course.

Mnemosyne will:

1. Convert the question into an embedding.
2. Compare it against the stored course embeddings.
3. Retrieve relevant transcript chunks.
4. Apply source-diversified retrieval.
5. Provide the retrieved context to Llama 3.2.
6. Generate a grounded answer.
7. Provide the relevant video and timestamp.

---

## 🧠 Source-Diversified Retrieval

One of the retrieval improvements made during development was **source-diversified retrieval**.

A simple top-k similarity search can return several highly similar chunks from the same video.

This can cause one video to dominate the context while relevant information from other videos gets pushed out.

Mnemosyne addresses this by:

```text
Retrieve a larger candidate pool
              ↓
Rank candidates by similarity
              ↓
Limit chunks from the same video
              ↓
Build a more diverse final context
```

### Current strategy

```text
Top 50 candidates
        ↓
Maximum 3 chunks per video
        ↓
Maximum 10 final chunks
```

This helps balance **similarity and source diversity** during retrieval.

---

## 🛡️ Grounding & Hallucination Prevention

Retrieval alone is not enough.

The LLM is explicitly instructed to answer using **only the retrieved course material**.

Mnemosyne follows several grounding rules:

* Do not fill gaps using general knowledge.
* Do not invent video numbers, titles, or timestamps.
* A topic being **mentioned** does not necessarily mean it was **taught**.
* A video should only be recommended when the transcript provides evidence that the instructor actually explains or demonstrates the topic.
* Unsupported course questions should be rejected rather than guessed.
* External resources should not be suggested when the requested topic is unsupported by the course material.

The goal is not simply:

> **"Generate an answer."**

It is:

> **"Generate an answer supported by the retrieved course evidence."**

---

## 🧪 Evaluation

Mnemosyne was tested using a **26-question evaluation set** rather than relying only on demonstration questions.

The evaluation includes:

* Direct course questions
* Multi-topic questions
* Unsupported topics
* Topics that are only briefly mentioned
* Low-similarity queries

The original retrieval approach was compared with the improved retrieval strategy to identify weaknesses and improve the system.

### Some observations from testing

* The highest similarity score does not always represent the best final source.
* Semantically related content is not necessarily evidence that a topic was taught.
* Retrieval quality and generation quality are separate problems.
* Multi-topic questions can require evidence from different videos.
* Effective RAG requires **relevance + evidence + diversity**, rather than similarity ranking alone.

---

# 📊 Current Status

### RAG Backend

* [x] Video preprocessing
* [x] Whisper transcription
* [x] Timestamp preservation
* [x] Contextualized transcript chunks
* [x] BGE-M3 embeddings
* [x] Semantic retrieval
* [x] Source-diversified retrieval
* [x] Grounded answer generation
* [x] Evidence-based source selection
* [x] YouTube timestamp links
* [x] Retrieval evaluation

### Frontend

* [ ] Student-facing UI
* [ ] Question → answer interface
* [ ] Video/source cards
* [ ] Integrated timestamp experience

**The RAG backend is currently working end-to-end, while the frontend is under development.**

---

## 🔮 What's Next?

The next phase is turning the working backend into a complete student-facing application.

Planned improvements include:

* Building the frontend
* Connecting the frontend to the RAG backend
* Improving the question → answer → timestamp experience
* Making the overall interaction more intuitive and polished

---

## 📚 What I Learned

Building Mnemosyne taught me that a RAG system is much more than connecting an embedding model to an LLM.

Three lessons stood out:

#### 1. Retrieval quality matters

A good LLM cannot reliably answer a question if the relevant evidence never reaches it.

#### 2. Similarity isn't everything

The highest-scoring chunk isn't automatically the best source. **Relevance, evidence, and source diversity** all matter.

#### 3. Evaluation reveals problems that demos hide

A system can look impressive on a few example questions while still failing on unsupported, ambiguous, or multi-topic queries.

---

## 🧰 Requirements

Before running Mnemosyne, make sure you have:

* Python
* FFmpeg
* Ollama
* Llama 3.2
* BGE-M3
* Required Python dependencies from `requirements.txt`

The exact model/runtime setup may depend on your hardware.

---

## 🤝 Contributing

Contributions, suggestions, and ideas are welcome.

If you find a bug or have an idea for improving retrieval, grounding, preprocessing, or the user experience, feel free to open an issue or submit a pull request.

---

## 📌 Project Status

**Mnemosyne is actively being developed.**

The core RAG pipeline is functional, and the next major milestone is the frontend.

> **Question → Retrieval → Grounded Answer → Relevant Lecture → Exact Timestamp**

---

### 🔗 Repository

**GitHub:**
https://github.com/chinmaymohite3036/Mnemosyne

---

#### Built to make course videos easier to search, understand, and revisit.
