import re
import torch
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Hyperbolic / idiomatic phrases commonly used in sarcastic replies
HYPERBOLIC_PATTERNS = [
    r"wouldn't miss it", r"for the world", r"nothing i enjoy more",
    r"absolutely love", r"my favorite thing", r"oh i'm sure",
    r"what a surprise", r"how wonderful", r"how delightful",
    r"so thrilled", r"couldn't be happier", r"just wonderful",
    r"oh really", r"oh definitely", r"oh absolutely",
    r"wouldn't dream of", r"nothing better", r"love nothing more",
    r"can hardly wait", r"what could go wrong", r"story of my life",
    r"lucky me", r"just my luck", r"oh joy", r"how exciting",
    r"what a treat", r"how original", r"so original",
    r"oh great", r"yeah right", r"sure thing",
    r"absolutely perfect", r"just perfect", r"couldn't be better",
    r"what a pleasure", r"so helpful", r"thanks a lot",
]

# Patterns indicating genuine empathy/support — used to DAMPEN false positives
GENUINE_MARKERS = [
    r"that must.ve been", r"that must have been",
    r"that's (rough|tough|unfortunate|hard|stressful)",
    r"don't worry", r"happens sometimes", r"sorry to hear",
    r"that (sucks|stinks)", r"hope you", r"i'm sorry",
    r"that's (manageable|okay|understandable|good|great|nice)",
    r"how did .+ go", r"are you (ok|okay|alright)",
    r"hang in there", r"take it easy", r"it'll be",
    r"that's really (unlucky|unfortunate|bad|sad)",
    r"sounds like it didn't",
]


class SarcasmPredictor:
    """Loads the fine-tuned RoBERTa model and predicts sarcasm scores."""

    def __init__(self, model_path: str = "models/roberta_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.sia = SentimentIntensityAnalyzer()

    def predict(self, text: str, context: str = "",
                speaker_prior_scores: list = None) -> dict:
        """Predict sarcasm probability with multi-signal post-processing."""

        # ── 1. Model Inference (unchanged — uses trained format) ──
        if context:
            formatted_input = f"A: {context} [SEP] B: {text}"
        else:
            formatted_input = f"B: {text}"

        inputs = self.tokenizer(
            formatted_input,
            padding="max_length",
            truncation=True,
            max_length=96,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            sarcasm_prob = probs[0][1].item()

        raw_score = sarcasm_prob  # preserve for speaker history

        # ── 2. Boosting Heuristics ────────────────────────────────
        msg_sentiment = self.sia.polarity_scores(text)['compound']
        # Normalize curly quotes to straight so regex patterns match
        text_lower = text.lower().replace('\u2019', "'").replace('\u2018', "'")
        boost = 0.0
        signals = 0

        # Compute context sentiment once (used by both boosting & dampening)
        ctx_sentiment = 0.0
        if context:
            ctx_sentiment = self.sia.polarity_scores(context)['compound']

        # 2a. Sentiment Contrast (strong only — negative ctx → positive reply)
        if context and ctx_sentiment < -0.2 and msg_sentiment > 0.4:
            boost += 0.15
            signals += 1

        # 2b. Hyperbolic / Idiomatic Language Detection
        has_hyperbole = False
        if context:
            for pattern in HYPERBOLIC_PATTERNS:
                if re.search(pattern, text_lower):
                    has_hyperbole = True
                    boost += 0.25
                    signals += 1
                    break

        # 2c. Same-Speaker Contradiction (uses RAW model scores)
        if speaker_prior_scores:
            if has_hyperbole and any(s > 0.5 for s in speaker_prior_scores):
                boost += 0.20
                signals += 1

        # 2d. Compound boost — multiple signals reinforce each other
        if signals >= 2:
            boost += 0.10

        sarcasm_prob = min(0.99, sarcasm_prob + boost)

        # ── 3. Dampening — reduce false positives ─────────────────
        # 3a. Questions without sarcastic patterns are usually genuine
        is_question = text.rstrip().endswith("?")
        if is_question and not has_hyperbole:
            sarcasm_prob *= 0.4

        # 3b. Empathetic / supportive language
        if not has_hyperbole:
            for pattern in GENUINE_MARKERS:
                if re.search(pattern, text_lower):
                    sarcasm_prob *= 0.3
                    break

        # 3c. First-person negative = complaint, not sarcasm
        #     "I had never seen before", "I think I might not get selected"
        if not has_hyperbole and msg_sentiment < 0.1:
            if re.search(r'\b(i|my|me|myself)\b', text_lower):
                sarcasm_prob *= 0.35

        # 3d. Negative-on-negative = empathy/agreement, not sarcasm
        #     Sarcasm needs CONTRAST (positive words in negative context).
        #     When context is negative and reply isn't clearly positive,
        #     it's genuine commiseration, not sarcasm.
        if context and not has_hyperbole:
            if ctx_sentiment < -0.1 and msg_sentiment < 0.2:
                sarcasm_prob *= 0.3

        # 3e. No strong positive words = unlikely sarcasm
        #     Sarcasm almost always involves strong positive words used
        #     ironically ("Amazing", "Perfect", "Great" all score ≥ 2.0
        #     in VADER). If no such word exists, dampen.
        if context and not has_hyperbole:
            words = [w.strip('.,!?;:"\'-…') for w in text_lower.split()]
            has_strong_positive = any(
                self.sia.lexicon.get(w, 0) >= 2.0 for w in words
            )
            if not has_strong_positive:
                sarcasm_prob *= 0.4

        sarcasm_prob = max(0.01, min(0.99, sarcasm_prob))

        return {
            "score": round(sarcasm_prob, 4),
            "raw_score": round(raw_score, 4),
            "label": self._get_label(sarcasm_prob)
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Predict sarcasm for multiple texts."""
        return [self.predict(text) for text in texts]

    @staticmethod
    def _get_label(score: float) -> str:
        if score > 0.55:
            return "sarcastic"
        elif score < 0.35:
            return "genuine"
        return "uncertain"
