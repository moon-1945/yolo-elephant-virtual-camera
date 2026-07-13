import torch
from ultralytics import YOLO


def main():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training will fall back to CPU and be very slow.")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = YOLO("yolo26n.pt")  # nano — smallest model, best fit for 4GB VRAM

    model.train(
        data="elephant_dataset/data.yaml",
        epochs=150,
        imgsz=640,          # standard; drop to 512/480 if you still hit OOM
        batch=2,            # auto-batch: Ultralytics picks the largest batch
                              # that fits ~60% of free VRAM — safest choice
                              # for a fixed 4GB budget. Set a fixed small
                              # int (e.g. 8 or 4) instead if auto-batch
                              # still overshoots on your card.
        amp=True,            # mixed precision — roughly halves memory use
        cache=False,         # don't cache images in RAM/VRAM; keep it low
        workers=4,           # CPU dataloader threads, not GPU-related
        patience=30,         # early stop if val mAP stalls for 30 epochs
        device=0,            # first CUDA GPU; use "cpu" if none available
        project="runs/elephant",
        name="yolo26n",
        val=True,
        plots=True,
    )

    # Quick validation pass on the held-out test split after training
    metrics = model.val(
        data="elephant_dataset/data.yaml",
        split="test",
        project="runs/elephant",
        name="yolo26n_val",
    )
    print(metrics)


if __name__ == "__main__":
    main()