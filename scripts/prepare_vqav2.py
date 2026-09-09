"""Import a deterministic, image-disjoint subset from official local VQAv2 files."""

import argparse
import json
import random
import shutil
from pathlib import Path


def prepare(questions, annotations, images, output, limit=100, seed=42):
    """Sample unique images and preserve all human answers and source IDs."""
    if limit < 3:
        raise ValueError("Use at least three images to produce all splits")
    qs = {q["question_id"]: q for q in json.loads(Path(questions).read_text())["questions"]}
    anns = sorted(
        json.loads(Path(annotations).read_text())["annotations"], key=lambda a: a["question_id"]
    )
    random.Random(seed).shuffle(anns)
    selected, seen = [], set()
    for annotation in anns:
        if annotation["image_id"] not in seen:
            selected.append(annotation)
            seen.add(annotation["image_id"])
        if len(selected) == limit:
            break
    if len(selected) < limit:
        raise ValueError("Not enough unique images")
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Output manifest already exists; choose a new output directory")
    (output / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, annotation in enumerate(selected):
        image_id = annotation["image_id"]
        matches = list(Path(images).glob(f"*_{image_id:012d}.jpg"))
        if len(matches) != 1:
            raise ValueError(f"Expected one COCO image for {image_id}, found {len(matches)}")
        destination = output / "images" / matches[0].name
        shutil.copyfile(matches[0], destination)
        split = "test" if i % 5 == 0 else "validation" if i % 5 == 1 else "train"
        q = qs[annotation["question_id"]]
        rows.append(
            {
                "id": str(q["question_id"]),
                "image": f"images/{destination.name}",
                "question": q["question"],
                "answers": [a["answer"] for a in annotation["answers"]],
                "category": annotation["question_type"],
                "split": split,
                "source_image_id": image_id,
            }
        )
    manifest = {
        "schema_version": 1,
        "name": "VQAv2-local-subset",
        "synthetic": False,
        "source": "https://visualqa.org/",
        "seed": seed,
        "partition_note": "Project splits from supplied official split; not official test",
        "examples": rows,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("questions", "annotations", "images", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    prepare(**vars(parser.parse_args()))
