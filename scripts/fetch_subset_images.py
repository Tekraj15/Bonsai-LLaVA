"""
Fetch only the COCO images actually referenced by the instruction data.

The full train2017 zip is ~19GB. The dataset references 81,479 unique images, and
resizing them to shortest-side 256 (SigLIP consumes 224) brings the whole corpus
to roughly 3.5-4GB. For a sanity run or subset training you need only a handful.

Usage:
    python scripts/fetch_subset_images.py --limit 64          # sanity-run subset
    python scripts/fetch_subset_images.py                     # all referenced images
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
import yaml
from PIL import Image
from tqdm import tqdm

COCO_URL = "http://images.cocodataset.org/train2017/{}"


def fetch_one(image_name, out_dir, short_side, quality, timeout=20):
    out_path = os.path.join(out_dir, image_name)
    if os.path.exists(out_path):
        return "skipped"
    try:
        resp = requests.get(COCO_URL.format(image_name), timeout=timeout)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")

        w, h = image.size
        scale = short_side / min(w, h)
        if scale < 1.0:
            image = image.resize((round(w * scale), round(h * scale)), Image.BICUBIC)

        image.save(out_path, "JPEG", quality=quality)
        return "downloaded"
    except Exception as e:
        return f"failed: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only fetch images for the first N dataset records.")
    parser.add_argument("--short_side", type=int, default=256)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir = config["image_folder"]
    os.makedirs(out_dir, exist_ok=True)

    with open(config["data_path"]) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]

    names = sorted({item["image"] for item in data if item.get("image")})
    print(f"{len(names)} unique images referenced by {len(data)} records -> {out_dir}")

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_one, n, out_dir, args.short_side, args.quality): n
            for n in names
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            result = fut.result()
            if result.startswith("failed"):
                counts["failed"] += 1
                failures.append((futures[fut], result))
            else:
                counts[result] += 1

    print(f"\ndownloaded={counts['downloaded']} "
          f"already-present={counts['skipped']} failed={counts['failed']}")
    if failures:
        print("First failures:")
        for name, err in failures[:5]:
            print(f"  {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
