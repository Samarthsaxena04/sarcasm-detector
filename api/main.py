from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from api.predict import SarcasmPredictor


app = FastAPI(title="Sarcasm Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
predictor = SarcasmPredictor(model_path="huggingsam32/sarcasm-detector-85accuracy")


class Message(BaseModel):
    sender: str
    text: str


class ConversationRequest(BaseModel):
    messages: list[Message]


@app.post("/analyze")
def analyze(req: ConversationRequest):
    """Analyze a conversation for sarcasm."""
    results = []
    total_score = 0
    previous_text = ""          # Immediate previous message (model context)
    speaker_scores = {}         # Per-speaker model sarcasm scores

    for msg in req.messages:
        prior_scores = speaker_scores.get(msg.sender, [])

        pred = predictor.predict(
            text=msg.text,
            context=previous_text,
            speaker_prior_scores=prior_scores
        )
        results.append({
            "sender": msg.sender,
            "text": msg.text,
            "score": pred["score"],
            "label": pred["label"]
        })
        total_score += pred["score"]

        previous_text = msg.text
        speaker_scores.setdefault(msg.sender, []).append(pred["raw_score"])

    # Weighted overall score: considers average, peak, and sarcastic proportion
    n = len(req.messages)
    if n > 0:
        avg_score = total_score / n
        max_score = max(r["score"] for r in results)
        sarcastic_ratio = sum(1 for r in results if r["score"] > 0.55) / n
        overall = round(0.35 * avg_score + 0.35 * max_score + 0.30 * sarcastic_ratio, 4)
    else:
        overall = 0

    return {
        "results": results,
        "overall_score": overall,
        "overall_label": SarcasmPredictor._get_label(overall)
    }


# Serve frontend
app.mount("/static", StaticFiles(directory="web"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.get("/")
def serve_frontend():
    return FileResponse("web/index.html")


@app.get("/how-it-works")
def serve_how_it_works():
    return FileResponse("web/how-it-works.html")
