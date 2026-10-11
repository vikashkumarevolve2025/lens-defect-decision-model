"""Contact-lens inspection using the OpenAI Decisions API (POST /v1/decisions).

The Decisions API is zero-shot: no training is needed. The Roboflow dataset is used
for (a) optional labelled reference images sent alongside the query image and
(b) offline evaluation (see evaluate.py).
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from PIL import Image, ImageOps

MODEL = os.getenv("OPENAI_DECISION_MODEL", "gpt-6-luna")
PRICE_PER_M_INPUT = 0.10  # USD per 1M input tokens (only input is billed)

# Category values mirror the Roboflow "lens_defect" classes, plus "good" and a fallback.
CATEGORIES: dict[str, tuple[str, str]] = {
    "good": ("Good lens", "Clean lens: no scratch, dig, crack, collapse or foreign material visible."),
    "scratch": ("Scratch", "Thin linear or curved mark on the lens surface."),
    "collapse": ("Collapse", "Lens is deformed, folded, buckled or has lost its round shape / edge."),
    "digs": ("Digs", "Small pits, spots, holes or point-like surface damage."),
    "glass_stick": ("Glass Stick", "Glass shard, fibre or foreign particle stuck on or in the lens."),
    "internal_crack": ("Internal Crack", "Crack or fracture line inside the lens material."),
    "other_defect": ("Other defect", "A visible defect that does not fit the categories above."),
}

SEVERITY_LEVELS = [
    {"label": "None", "description": "No defect visible."},
    {"label": "Cosmetic", "description": "Tiny, faint mark outside the optical centre; acceptable."},
    {"label": "Major", "description": "Clear defect or one in the optical zone; lens should be rejected."},
    {"label": "Critical", "description": "Large, multiple or structural damage (crack, collapse)."},
]

QUESTIONS = [
    {
        "type": "predicate",
        "name": "is_contact_lens",
        "instructions": "The final image (the one to inspect) shows a contact lens, typically a "
        "grayscale top-down inspection photo of a round lens in a holder.",
    },
    {
        "type": "predicate",
        "name": "has_defect",
        "instructions": "The lens in the final image has at least one visible defect such as a "
        "scratch, dig, crack, collapse/deformation or stuck foreign particle. Look carefully at "
        "the whole lens including the edge; defects can be very small.",
    },
    {
        "type": "choice",
        "name": "defect_type",
        "instructions": "Classify the most significant condition of the lens in the final image.",
        "choices": [{"value": k, "description": f"{n}: {d}"} for k, (n, d) in CATEGORIES.items()],
    },
    {
        "type": "score",
        "name": "severity",
        "instructions": "Rate how severe the worst defect on the lens in the final image is.",
        "levels": SEVERITY_LEVELS,
    },
]


@dataclass
class LensResult:
    verdict: str  # GOOD | DEFECTIVE | REVIEW | NOT_A_LENS
    reason: str
    is_lens_prob: float | None = None
    defect_prob: float | None = None
    defect_type: str | None = None
    defect_confidence: float | None = None
    defect_probs: dict[str, float] = field(default_factory=dict)
    severity_score: float | None = None
    severity_label: str | None = None
    severity_confidence: float | None = None
    refusals: list[str] = field(default_factory=list)
    input_tokens: int = 0
    raw: dict = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1_000_000 * PRICE_PER_M_INPUT


def preprocess(img: Image.Image, enhance: bool = True, max_side: int = 1536) -> Image.Image:
    """Normalise orientation/size; optional contrast boost similar to the dataset's equalization."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    if enhance:
        img = ImageOps.autocontrast(img, cutoff=1)
    return img


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_references(ref_dir: str | Path, per_class: int = 2) -> list[tuple[str, Image.Image]]:
    """Load labelled examples from references/<category>/*.png|jpg (category = CATEGORIES key)."""
    refs: list[tuple[str, Image.Image]] = []
    root = Path(ref_dir)
    if not root.is_dir():
        return refs
    for cat in CATEGORIES:
        files = sorted(p for p in (root / cat).glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for p in files[:per_class]:
            refs.append((cat, Image.open(p)))
    return refs


def build_input(img: Image.Image, references, detail: str, enhance: bool) -> list[dict]:
    content: list[dict] = [{
        "type": "input_text",
        "text": "You are a quality inspector for contact lenses on a production line.",
    }]
    if references:
        content.append({"type": "input_text", "text": "Labelled reference examples (close-up crops from other lenses, showing what each defect looks like):"})
        for cat, ref in references:
            name, desc = CATEGORIES[cat]
            content.append({"type": "input_text", "text": f"Reference — {name}: {desc}"})
            content.append({"type": "input_image", "image_url": to_data_url(preprocess(ref, enhance, 768)), "detail": "high"})
    content.append({"type": "input_text", "text": "Final image to inspect:"})
    content.append({"type": "input_image", "image_url": to_data_url(img), "detail": detail})
    return [{"role": "user", "content": content}]


def classify(
    image: Image.Image,
    client: OpenAI | None = None,
    *,
    references=None,
    detail: str = "high",
    enhance: bool = True,
    defect_threshold: float = 0.5,
    review_band: float = 0.15,
    min_confidence: float = 0.5,
) -> LensResult:
    client = client or OpenAI()
    img = preprocess(image, enhance)
    decision = client.decisions.create(
        model=MODEL,
        input=build_input(img, references or [], detail, enhance),
        questions=QUESTIONS,
    )
    return interpret(decision, defect_threshold, review_band, min_confidence)


def interpret(decision, defect_threshold=0.5, review_band=0.15, min_confidence=0.5) -> LensResult:
    a = {ans.name: ans for ans in decision.answers}
    r = LensResult(verdict="REVIEW", reason="", input_tokens=decision.usage.input_tokens,
                   raw=decision.model_dump())
    r.refusals = [n for n, ans in a.items() if ans.type == "refusal"]

    if (x := a.get("is_contact_lens")) and x.type == "predicate":
        r.is_lens_prob = x.probability
    if (x := a.get("has_defect")) and x.type == "predicate":
        r.defect_prob = x.probability
    if (x := a.get("defect_type")) and x.type == "choice":
        r.defect_type, r.defect_confidence = str(x.choice), x.confidence
        r.defect_probs = {str(p.value): p.probability for p in x.probabilities}
    if (x := a.get("severity")) and x.type == "score":
        r.severity_score, r.severity_confidence = x.score, x.confidence
        r.severity_label = SEVERITY_LEVELS[min(round(x.score), len(SEVERITY_LEVELS) - 1)]["label"]

    # --- decision logic -------------------------------------------------------
    if r.is_lens_prob is not None and r.is_lens_prob < 0.5:
        r.verdict, r.reason = "NOT_A_LENS", "Image does not appear to be a contact lens."
    elif r.defect_prob is None or r.defect_type is None:
        r.reason = f"Model refused or did not answer: {', '.join(r.refusals) or 'unknown'}."
    elif abs(r.defect_prob - defect_threshold) < review_band:
        r.reason = f"Defect probability {r.defect_prob:.0%} is too close to the threshold."
    elif (r.defect_prob >= defect_threshold) != (r.defect_type != "good"):
        r.reason = "Defect check and defect type disagree."
    elif r.defect_confidence < min_confidence:
        r.reason = f"Low confidence in category ({r.defect_confidence:.0%})."
    elif r.defect_prob >= defect_threshold:
        r.verdict = "DEFECTIVE"
        r.reason = f"{CATEGORIES.get(r.defect_type, (r.defect_type,))[0]} detected."
    else:
        r.verdict, r.reason = "GOOD", "No defect detected."
    return r
