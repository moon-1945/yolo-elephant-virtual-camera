import random
import shutil
from pathlib import Path

from datasets import load_dataset, load_dataset_builder
from tqdm import tqdm

OUTPUT = Path("elephant_dataset")
ALL_IMG_DIR = OUTPUT / "all" / "images"
ALL_LBL_DIR = OUTPUT / "all" / "labels"

# Final split ratios (must sum to 1.0)
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
RANDOM_SEED = 42


def collect_all():
    """Phase 1: pull elephant samples from every HF split into one flat folder."""
    builder = load_dataset_builder("vikhyatk/openimages-bbox")
    split_sizes = {name: info.num_examples for name, info in builder.info.splits.items()}
    print(split_sizes)

    ALL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    ALL_LBL_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0

    for hf_split in split_sizes:
        ds = load_dataset("vikhyatk/openimages-bbox", split=hf_split, streaming=True)

        with tqdm(
            ds,
            total=split_sizes[hf_split],
            desc=f"Collecting {hf_split}",
            unit="img",
        ) as pbar:
            for sample in pbar:
                labels = []

                for obj in sample["objects"]:
                    if obj["label"].lower() != "elephant":
                        continue
                    xmin = min(max(obj["xmin"], 0.0), 1.0)
                    ymin = min(max(obj["ymin"], 0.0), 1.0)
                    xmax = min(max(obj["xmax"], 0.0), 1.0)
                    ymax = min(max(obj["ymax"], 0.0), 1.0)

                    w = xmax - xmin
                    h = ymax - ymin
                    if w <= 0 or h <= 0:
                        # degenerate box (annotation edge case) — skip
                        continue

                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2

                    labels.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

                if not labels:
                    continue

                sample["image"].save(ALL_IMG_DIR / f"{saved:08d}.jpg", quality=95)

                with open(ALL_LBL_DIR / f"{saved:08d}.txt", "w") as f:
                    f.write("\n".join(labels))

                saved += 1
                pbar.set_postfix(saved=saved)

    print(f"Collected {saved} images total into {ALL_IMG_DIR}")
    return saved


def split_and_move():
    """Phase 2: shuffle the flat pool and move files into train/validation/test."""
    assert abs(sum(SPLIT_RATIOS.values()) - 1.0) < 1e-6, "SPLIT_RATIOS must sum to 1.0"

    stems = sorted(p.stem for p in ALL_IMG_DIR.glob("*.jpg"))
    if not stems:
        print("No images found to split — did collect_all() run first?")
        return

    random.seed(RANDOM_SEED)
    random.shuffle(stems)

    n = len(stems)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["validation"])
    # test gets the remainder, so rounding never drops a sample
    split_map = {
        "train": stems[:n_train],
        "validation": stems[n_train:n_train + n_val],
        "test": stems[n_train + n_val:],
    }

    for split_name, split_stems in split_map.items():
        img_dir = OUTPUT / "images" / split_name
        lbl_dir = OUTPUT / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in tqdm(split_stems, desc=f"Moving {split_name}", unit="img"):
            shutil.move(str(ALL_IMG_DIR / f"{stem}.jpg"), str(img_dir / f"{stem}.jpg"))
            shutil.move(str(ALL_LBL_DIR / f"{stem}.txt"), str(lbl_dir / f"{stem}.txt"))

        print(f"{split_name}: {len(split_stems)} images")

    # Clean up the now-empty flat pool folder
    shutil.rmtree(OUTPUT / "all", ignore_errors=True)


def write_data_yaml():
    """Writes the data.yaml Ultralytics needs to train/validate against this dataset."""
    content = (
        f"path: {OUTPUT.resolve()}\n"
        "train: images/train\n"
        "val: images/validation\n"
        "test: images/test\n"
        "names:\n"
        "  0: elephant\n"
    )
    (OUTPUT / "data.yaml").write_text(content)
    print(f"Wrote {OUTPUT / 'data.yaml'}")


if __name__ == "__main__":
    collect_all()
    split_and_move()
    write_data_yaml()