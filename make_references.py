"""Build few-shot reference crops in ./references/<category>/ from the train split.

For each defect class, picks typical, clearly visible annotations (from different source images)
and saves a square close-up crop around each one.

Usage:
    python make_references.py [data/lens_defect-8/train] [--per-class 2]
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from evaluate import norm

OUT = Path("references")


def source_id(file_name: str) -> str:
    return file_name.split(".rf.")[0]  # Roboflow augmentations share this prefix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir", type=Path, nargs="?", default=Path("data/lens_defect-8/train"))
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    coco = json.loads((args.split_dir / "_annotations.coco.json").read_text())
    cats = {c["id"]: norm(c["name"]) for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}

    by_class: dict[str, list] = {}
    for a in coco["annotations"]:
        by_class.setdefault(cats[a["category_id"]], []).append(a)

    for cat, anns in sorted(by_class.items()):
        out_dir = OUT / cat
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        used: set[str] = set()
        # Annotations near the 75th-percentile area: clearly visible but still typical.
        areas = sorted(a.get("area", 0) for a in anns)
        p75 = areas[int(len(areas) * 0.75)]
        for a in sorted(anns, key=lambda a: abs(a.get("area", 0) - p75)):
            img_info = images[a["image_id"]]
            if source_id(img_info["file_name"]) in used:
                continue
            used.add(source_id(img_info["file_name"]))

            img = Image.open(args.split_dir / img_info["file_name"])
            x, y, w, h = a["bbox"]
            side = min(max(max(w, h) * 2.5 + 64, 320), min(img.size))
            cx, cy = x + w / 2, y + h / 2
            left = min(max(cx - side / 2, 0), img.width - side)
            top = min(max(cy - side / 2, 0), img.height - side)
            crop = img.crop((left, top, left + side, top + side)).resize((512, 512), Image.LANCZOS)
            crop.save(out_dir / f"{cat}_{len(used)}.png")
            if len(used) >= args.per_class:
                break
        print(f"{cat}: {len(used)} reference crop(s)")


if __name__ == "__main__":
    main()
