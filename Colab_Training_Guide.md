# Google Colab Training Guide (YOLOv8)

Use this guide to train a high-precision pothole model with strong mAP50 on Google Colab.

**Quick run:** The defaults below use **25 epochs** for the main training so a single Colab session finishes in reasonable time (good enough for demos; bump epochs later if you need every last point of mAP).

## 1) Colab Runtime Setup

1. Open Colab.
2. Go to `Runtime -> Change runtime type`.
3. Set:
   - Hardware accelerator: `GPU`
   - GPU type (if available): `T4` or better

## 2) Install Dependencies

Run in a Colab cell:

```python
!nvidia-smi
!pip install -U ultralytics roboflow
```

## 3) Download Dataset (Roboflow Export: YOLOv8)

Use your own Roboflow workspace/project/version. The export should include `data.yaml`.

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(YOUR_VERSION_NUMBER).download("yolov8")
print("Dataset path:", dataset.location)
```

> If you already downloaded the zip manually, upload/unzip into Colab and note the `data.yaml` path.

## 4) Baseline Training (Strong Starting Point)

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # Try yolov8s.pt if GPU RAM is limited

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,           # reduce to 8 if OOM
    optimizer="AdamW",
    lr0=0.0015,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=2.0,
    cos_lr=True,
    patience=12,        # early stop if val plateaus (fits a 25-epoch budget)
    close_mosaic=8,   # disable mosaic in last N epochs (keep < epochs)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=2.0,
    translate=0.08,
    scale=0.35,
    shear=1.0,
    perspective=0.0005,
    fliplr=0.5,
    mosaic=0.7,
    mixup=0.05,
    copy_paste=0.0,
    erasing=0.2,
    pretrained=True,
    cache=True,
    amp=True,
    workers=2,
    project="runs",
    name="pothole_yolov8m_baseline",
    exist_ok=True
)
```

> **Colab warning:** The `runs/` folder lives on the **ephemeral VM**. If you **disconnect, restart runtime, or open a new notebook**, Section 4’s `best.pt` is **gone**. Run Section 4 again in the **same session** before Section 5, or **upload** a saved `best.pt` and point `BASELINE_PT` to that path.

## 5) Precision-Focused Fine-Tuning

After baseline, optional short fine-tune from `best.pt` with lighter augmentation (add ~10 extra minutes). Skip if you only want one 25-epoch run.

**Always set the checkpoint path explicitly.** If you get `FileNotFoundError`, run the “find `best.pt`” cell below first.

```python
from pathlib import Path
from ultralytics import YOLO

# Default = where Section 4 saves weights (must match name= above)
BASELINE_PT = Path("runs/detect/pothole_yolov8m_baseline/weights/best.pt")

if not BASELINE_PT.is_file():
    # Auto-pick newest best.pt under runs/detect (if you used a different name=)
    candidates = sorted(
        Path("runs/detect").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        BASELINE_PT = candidates[0]
        print("Using:", BASELINE_PT)
    else:
        raise FileNotFoundError(
            "No best.pt found. Run Section 4 in THIS session first, or upload best.pt and set BASELINE_PT = Path('/content/best.pt')"
        )

finetune = YOLO(str(BASELINE_PT))

results_ft = finetune.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01,
    weight_decay=0.0007,
    warmup_epochs=1.0,
    cos_lr=True,
    close_mosaic=0,
    mosaic=0.0,
    mixup=0.0,
    degrees=0.5,
    translate=0.04,
    scale=0.2,
    fliplr=0.3,
    amp=True,
    cache=True,
    patience=5,
    project="runs",
    name="pothole_yolov8m_precision_tune",
    exist_ok=True
)
```

## 6) Validate and Read Metrics

```python
from ultralytics import YOLO

# If you skipped Section 5, use baseline weights instead:
# best_model = YOLO("runs/detect/pothole_yolov8m_baseline/weights/best.pt")
best_model = YOLO("runs/detect/pothole_yolov8m_precision_tune/weights/best.pt")
metrics = best_model.val(data=f"{dataset.location}/data.yaml", split="val", conf=0.25, iou=0.7)
print(metrics)
```

Focus on:
- Precision (`P`) -> your "Verification Precision Score"
- `mAP50`
- `mAP50-95`

## 7) Find Best Inference Threshold for Maximum Precision

Default validation may not give your best operating point. Sweep confidence threshold and pick best precision while keeping recall acceptable.

```python
from ultralytics import YOLO

# Or: YOLO("runs/detect/pothole_yolov8m_baseline/weights/best.pt") if you skipped fine-tuning
best_model = YOLO("runs/detect/pothole_yolov8m_precision_tune/weights/best.pt")

for conf in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
    print(f"\n--- conf={conf} ---")
    best_model.val(data=f"{dataset.location}/data.yaml", split="val", conf=conf, iou=0.7)
```

Typical behavior:
- Higher `conf` -> higher precision, lower recall
- Lower `conf` -> lower precision, higher recall

Pick the threshold that maximizes precision without collapsing recall.

## 8) Train Multiple Seeds (Most Important for Reliable Results)

Run 2–3 trainings with different seeds and compare (each run is 25 epochs).

```python
from ultralytics import YOLO

seeds = [0, 21, 42]
for s in seeds:
    model = YOLO("yolov8m.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        optimizer="AdamW",
        lr0=0.0015,
        cos_lr=True,
        seed=s,
        deterministic=True,
        project="runs",
        name=f"pothole_seed_{s}",
        exist_ok=True
    )
```

Choose the run with best precision + mAP50 balance, not only one metric.

## 9) Export Final Model for Your Streamlit App

```python
from ultralytics import YOLO

final_model = YOLO("runs/detect/pothole_yolov8m_precision_tune/weights/best.pt")  # or baseline best.pt
final_model.export(format="onnx")  # optional
```

For your current app, copy the PyTorch checkpoint:
- `runs/detect/pothole_yolov8m_precision_tune/weights/best.pt` (or `.../pothole_yolov8m_baseline/weights/best.pt` if you skipped Section 5)
- Rename to `pothole_yolov8.pt` and place in project root.

## 10) Practical Tips to Improve Precision and mAP50

- Keep labels clean (wrong boxes strongly hurt precision).
- Avoid extreme augmentations for final fine-tuning.
- Increase image size to `768` if GPU allows (often boosts mAP).
- Use `yolov8l.pt` if you have enough VRAM/time.
- Evaluate on an untouched test set only once at the end.

## Recommended Experiment Grid

Run these and compare (all **25 epochs** unless noted):

1. `yolov8s`, imgsz `640`, epochs `25`
2. `yolov8m`, imgsz `640`, epochs `25`
3. `yolov8m`, imgsz `768`, epochs `25`, batch reduced if needed
4. Best run + optional precision fine-tune (Section 5, +10 epochs)
5. Confidence threshold sweep (Section 7)

For higher ceiling metrics later, repeat Section 4 with `epochs=50` or `100` when you have time.

## Troubleshooting: `FileNotFoundError: ... best.pt`

| Cause | Fix |
|--------|-----|
| **New Colab session / runtime restarted** | `runs/` was wiped. Run **Section 4** again, then Section 5 **in the same session**, or upload a saved `best.pt` to e.g. `/content/best.pt` and set `BASELINE_PT = Path("/content/best.pt")`. |
| **Different `name=` in `train()`** | Weights are under `runs/detect/<that_name>/weights/best.pt`. Use the auto-pick cell in Section 5 or `!find runs -name best.pt` in Colab. |
| **Baseline never finished** | Training must complete at least one epoch with validation; check Section 4 output for errors. |
| **Wrong working directory** | In Colab, `!pwd` should usually be `/content`. Run training from the folder where `runs/` is created. |

After a good training run, **download** `best.pt` immediately (Files panel → right-click) or copy to **Google Drive** so fine-tuning later does not depend on the same runtime.
