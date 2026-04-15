# Google Colab Training(YOLOv8)



## 1) Installation of Dependencies



```python
!nvidia-smi
!pip install -U ultralytics roboflow
```

## 2) Downloaded the Dataset (Roboflow Export: YOLOv8)



```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(YOUR_VERSION_NUMBER).download("yolov8")
print("Dataset path:", dataset.location)
```



## 2) Baseline Training

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,           
    optimizer="AdamW",
    lr0=0.0015,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=2.0,
    cos_lr=True,
    patience=12,        
    close_mosaic=8,   
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



## 3) Precision-Focused Fine-Tuning

After baseline, I fine-tuned  from `best.pt` with lighter augmentation. 



```python
from pathlib import Path
from ultralytics import YOLO

# Baseline model path
BASELINE_PT = Path("runs/detect/pothole_yolov8m_baseline/weights/best.pt")

if not BASELINE_PT.is_file():
    # This line allows autopick the model
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

## 4) Validation of Metrics

```python
from ultralytics import YOLO


best_model = YOLO("runs/detect/pothole_yolov8m_precision_tune/weights/best.pt")
metrics = best_model.val(data=f"{dataset.location}/data.yaml", split="val", conf=0.25, iou=0.7)
print(metrics)
```


- Precision (`P`) -> our "Verification Precision Score"
- `mAP50`
- `mAP50-95`

