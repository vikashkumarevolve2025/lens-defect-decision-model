"""Evaluate the classifier on a Roboflow COCO-segmentation export.

Usage:
    python evaluate.py path/to/lens_defect-8/test [--limit 20] [--refs]

Ground truth per image = class with the largest total annotated area ("primary");
images with no annotations count as "good". "Any-match" accuracy also counts a prediction
of any class present in the image (matters only for images with several defect types).
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from lens_classifier import classify, load_references


def norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def ground_truth(split_dir: Path) -> dict[str, tuple[str, set[str]]]:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    cats = {c["id"]: norm(c["name"]) for c in coco["categories"]}
    area = defaultdict(Counter)
    for a in coco["annotations"]:
        area[a["image_id"]][cats[a["category_id"]]] += a.get("area", 1)
    return {
        img["file_name"]: (area[img["id"]].most_common(1)[0][0], set(area[img["id"]]))
        if area[img["id"]] else ("good", {"good"})
        for img in coco["images"]
    }


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refs", action="store_true", help="use ./references few-shot images")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    gt = ground_truth(args.split_dir)
    print("Ground-truth distribution:", dict(Counter(p for p, _ in gt.values())))
    items = list(gt.items())[: args.limit or None]
    client, refs = OpenAI(), (load_references("references") if args.refs else [])

    rows, tokens = [], 0
    for i, (fname, (truth, present)) in enumerate(items, 1):
        r = classify(Image.open(args.split_dir / fname), client, references=refs)
        tokens += r.input_tokens
        rows.append({"file": fname, "truth": truth, "present": "|".join(sorted(present)),
                     "pred_type": r.defect_type, "any_match": r.defect_type in present, "verdict": r.verdict,
                     "defect_prob": r.defect_prob, "confidence": r.defect_confidence})
        print(f"[{i}/{len(items)}] {fname[:40]:40} truth={truth:15} pred={r.defect_type!s:15} {r.verdict}")

    n = len(rows)
    type_acc = sum(r["truth"] == r["pred_type"] for r in rows) / n
    any_acc = sum(r["any_match"] for r in rows) / n
    bin_acc = sum((r["truth"] == "good") == (r["verdict"] == "GOOD") for r in rows) / n
    review = sum(r["verdict"] == "REVIEW" for r in rows) / n
    print(f"\nCategory accuracy (primary): {type_acc:.1%} | Any-match: {any_acc:.1%}\n"
          f"Good-vs-defect accuracy: {bin_acc:.1%} | "
          f"Sent to review: {review:.1%} | Tokens: {tokens:,} (~${tokens / 1e6 * 0.10:.4f})")
    print("Confusion (truth -> pred):", dict(Counter((r["truth"], r["pred_type"]) for r in rows)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
