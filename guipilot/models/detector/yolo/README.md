
# Yolo for UI recognition

# Run pretrained object detector

---

- Step 1: Download pretrained checkpoint from [this google drive link](https://drive.google.com/file/d/15YUP8aMXC97n-m-U28wP1Sz5Eabw69m-/view?usp=sharing).
- Step 2: Make sure that [`ultralytics`](https://docs.ultralytics.com/quickstart/#install-ultralytics) is installed, run the following code to make inference.
```python
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8m model
model = YOLO("best.pt")

# Display model information (optional)
model.info()

# Run inference, visualize prediction results at viz.png
results = model("path/to/screenshot.jpg")
results[0].save(filename="viz.png")
```

# (Optional) Train model on your own dataset

---

## Prepare custom data for training
- Step 1: Save bounding box annotations to [./datasets/new](../../../../datasets/new).
The folder structure should look like:
```
datasets/
    |_ new/
        |_ app1/
            |_ 1.jpg
            |_ 1.json (this is the corresponding annotations for screenshot 1)
            |_ 2.jpg
            |_ 2.json
            |_ ......
        |_ app2/
            |_ ......
```

- Step 2: Edit the `train` and `val` arguments in `datasets.yaml`, they should be pointing to a directory where you want to save the training and testing images and labels.
E.g. /home/....guipilot/datasets/finetune/images/train/ and /home/....guipilot/datasets/finetune/images/val/. Please use **absolute path** here.
- Step 3: Run dataset splitting, the annotations will be saved in COCO format in json files.
```bash
python -m guipilot.models.detector.yolo.main \
--config guipilot/models/detector/yolo/dataset.yaml \
--prepare_data \
--json_train ./datasets/80app_train_groundtruth.json \
--json_test ./datasets/80app_test_groundtruth.json
```
- Step 4: Verify that your dataset has been successfully created.
```
datasets/
    |_ finetune/
        |_ images/
            |_ train/
                |_ ...jpg
            |_ val/
                |_ ...jpg
        |_ labels/
            |_ train/
                |_ ... txt
            |_ val/
                |_ ... txt
```

## Run training
```bash
python -m guipilot.models.detector.yolo.main \
--config guipilot/models/detector/yolo/dataset.yaml \
--json_train ./datasets/80app_train_groundtruth.json \
--json_test ./datasets/80app_test_groundtruth.json
--do_train --model_name yolov8m \
--epochs 10 --batch_size 16 --lr 1e-4
```


## Run evaluation (Compute mAP, mAR on testing set)
```bash
python -m guipilot.models.detector.yolo.main \
--config guipilot/models/detector/yolo/dataset.yaml \
--do_test
```
