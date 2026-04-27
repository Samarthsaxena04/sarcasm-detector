<div align="center">
  <h1>🕵️‍♂️ Sarcasm Detector</h1>
  <p><b>An AI-powered web app that detects sarcasm in real-time chat conversations.</b></p>
</div>

<br>

## 🚀 Overview

Detecting sarcasm is one of the hardest problems in Natural Language Processing (NLP) because it requires understanding **context**, **tone**, and **intent**—not just grammar. 

This project is an end-to-end Machine Learning pipeline that analyzes multi-turn chat conversations and assigns a real-time "Sarcasm Confidence Score" to each message. 

It was built in multiple phases, starting from a basic grammatical model (HMM) and evolving into a **State-of-the-Art Context-Aware RoBERTa Transformer**, augmented by a custom linguistic post-processing engine.

---

## 🧠 How It Works

Our system uses a two-step approach to achieve **~85% effective accuracy** on real-world conversations:

### 1. The Deep Learning Model (RoBERTa)
At its core, the app uses a **RoBERTa Transformer** fine-tuned on the massive Reddit "Sarcasm" dataset. 
Instead of just reading a single message, we feed the model the **previous message (context) + the current reply**. By using the Transformer's Self-Attention mechanism, it mathematically detects the contradiction between the context and the reply.
*(Raw Model Accuracy on Reddit data: 75.2%)*

### 2. The Inference Post-Processing Engine
Because real-life chat is more nuanced than Reddit comments, we built a **9-rule heuristics pipeline** (using NLTK's VADER sentiment analysis) that runs instantly after the model predicts:

**🔥 Boosting Rules (Catching Missed Sarcasm):**
- **Idiomatic Phrases:** Detects phrases that are almost always sarcastic (*"wouldn't miss it for the world"*).
- **Sentiment Contrast:** Triggers when a reply is highly positive but the context was highly negative.
- **Speaker Contradiction:** Remembers if a speaker was sarcastic earlier in the chat and boosts their score if they use hyperbole again.

**🛡️ Dampening Rules (Eliminating False Positives):**
- **Question Filter:** Simple questions (*"How did it go?"*) are usually genuine.
- **Empathy Filter:** Phrases like *"that's rough"* or *"don't worry"* get their sarcasm scores heavily reduced.
- **Complaint Detection:** Negative first-person statements (*"I had never seen that before"*) are flagged as complaints, not sarcasm.
- **Negative-on-Negative:** When both context and reply are negative, it's genuine empathy, not sarcasm.

---

## 💻 Tech Stack

- **Backend / API:** Python, FastAPI, Uvicorn
- **Machine Learning:** PyTorch, Hugging Face Transformers (RoBERTa)
- **NLP Processing:** NLTK (VADER Sentiment Lexicon)
- **Frontend:** Vanilla HTML, CSS, JavaScript (No frameworks)

---

## 🛠️ Local Setup & Installation

Follow these steps to run the Sarcasm Detector locally:

**1. Clone the repository & enter the directory**
```bash
# Open your terminal and navigate to the project folder
cd "Sarcasm Detecter"
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
*(Make sure you have PyTorch installed for your specific hardware. The app will automatically use a GPU if available, or fallback to CPU).*

**3. Start the Backend API Server**
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**4. Open the App**
Open your web browser and go to:
👉 `http://127.0.0.1:8000/`

---

## 📁 Project Structure

```text
├── api/
│   ├── main.py        # FastAPI server & endpoints
│   └── predict.py     # RoBERTa inference + 9-rule heuristics pipeline
├── models/            # Fine-tuned RoBERTa weights & metrics
├── web/               # Frontend UI
│   ├── index.html     # Main chat interface
│   ├── app.js         # Frontend logic
│   ├── style.css      # Custom styling
│   └── how-it-works.html # Detailed technical walkthrough
├── notebooks/         # Jupyter notebooks for Phase 1-4 model training
├── data/              # Raw and processed datasets
└── requirements.txt   # Python dependencies
```

---

<div align="center">
  <p><i>"Sarcasm is the lowest form of wit, but the highest form of intelligence."</i></p>
</div>
