import argparse
import json
import os
import shutil

import numpy as np
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from .shapes import Widget


def create_coco_json(annotation_list, label2id, output_json_path, coco_format):
    # Build a dictionary to hold image filenames and their corresponding ids
    image_dict = {img["file_name"]: img["id"] for img in coco_format["images"]}
    category_dict = {cat["name"]: cat["id"] for cat in coco_format["categories"]}

    # Continue indexing from the last id or start from 1
    image_id = max(image_dict.values(), default=0) + 1
    annotation_id = (
        max((ann["id"] for ann in coco_format["annotations"]), default=0) + 1
    )

    for annotation in annotation_list:
        image_name = annotation["image_name"]
        image_path = image_name
        if not os.path.exists(image_path):
            print(f"{image_path} does not exist")
            continue

        # Ensure each image is only added once
        if image_name not in image_dict:
            image = Image.open(image_path)
            width, height = image.size
            image_dict[image_name] = image_id

            # Add image info
            coco_format["images"].append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": image_name,
                }
            )
            image_id += 1

        current_image_id = image_dict[image_name]

        # Check and add category if not already added
        if annotation["category"] not in category_dict:
            category_id = label2id[annotation["category"]]
            category_dict[annotation["category"]] = category_id
            coco_format["categories"].append(
                {"id": category_id, "name": annotation["category"]}
            )

        # Add annotation info
        coco_format["annotations"].append(
            {
                "id": annotation_id,
                "image_id": current_image_id,
                "category_id": category_dict[annotation["category"]],
                "bbox": annotation["bbox"],  # [x, y, width, height]
                "segmentation": annotation.get("segmentation", []),
                "area": annotation["bbox"][2] * annotation["bbox"][3],
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    # Write the updated COCO data to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(coco_format, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Yolo model")
    parser.add_argument(
        "--config",
        type=str,
        default="./yolo/dataset.yaml",
        help="dataset configuration",
    )
    parser.add_argument(
        "--json_train",
        type=str,
        default="./datasets/80app_train_groundtruth.json",
        help="save the training data as json",
    )
    parser.add_argument(
        "--json_test",
        type=str,
        default="./datasets/80app_test_groundtruth.json",
        help="save the test data as json",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="yolov8m",
        help="which model to use",
        choices=[
            "yolov8m",
            "yolov8l",
            "yolov8n",
            "yolov10s",
            "yolov10m",
            "yolov10l",
            "yolo11n",
            "yolo11s",
            "yolo11m",
            "yolo11l",
        ],
    )
    parser.add_argument("--epochs", type=int, default=10, help="num epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--prepare_data", default=False, action="store_true")
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    args = parser.parse_args()

    # Load the YAML file
    with open(args.config, "r") as file:  # Replace with the actual file path
        data = yaml.safe_load(file)

    # Convert names list to a dictionary
    label2id = {name: idx for idx, name in enumerate(data["names"])}
    ground_truth_dir = data["raw"]

    train_img_dir = data["train"]
    train_label_dir = train_img_dir.replace("images", "labels")
    val_img_dir = data["val"]
    val_label_dir = val_img_dir.replace("images", "labels")

    if args.prepare_data:
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        train_annotation_list = []
        test_annotation_list = []

        file_list = os.listdir(ground_truth_dir)
        sorted_file_list = sorted(file_list, key=lambda s: s.lower())
        num_train = int(data["train_ratio"] * len(file_list))

        for ct, app_dir in enumerate(sorted_file_list):
            for file in os.listdir(os.path.join(ground_truth_dir, app_dir)):
                if file.endswith("json"):
                    json_file_path = os.path.join(ground_truth_dir, app_dir, file)
                    data = json.load(open(json_file_path, encoding="utf-8"))
                    ws = [
                        Widget.from_labelme(d, i) for i, d in enumerate(data["shapes"])
                    ]

                    image_height = data["imageHeight"]
                    image_width = data["imageWidth"]

                    if os.path.exists(json_file_path.replace(".json", ".png")):
                        image_path = file.replace(".json", ".png")
                    elif os.path.exists(json_file_path.replace(".json", ".jpg")):
                        image_path = file.replace(".json", ".jpg")
                    else:
                        image_path = file.replace(".json", ".jpeg")

                    if ct < num_train:  # training
                        Widget.write_yolo_labels(
                            os.path.join(
                                train_label_dir,
                                app_dir + "_" + file.replace("json", "txt"),
                            ),
                            label2id,
                            ws,
                            image_width,
                            image_height,
                        )

                        for shape in data["shapes"]:
                            new_annot = {}
                            new_annot["image_name"] = os.path.join(
                                ground_truth_dir, app_dir, image_path
                            )
                            x1, y1 = shape["points"][0]
                            x2, y2 = shape["points"][1]
                            new_annot["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                            new_annot["category"] = shape["label"]
                            train_annotation_list.append(new_annot)

                        shutil.copy(
                            os.path.join(ground_truth_dir, app_dir, image_path),
                            os.path.join(train_img_dir, app_dir + "_" + image_path),
                        )
                    else:
                        Widget.write_yolo_labels(
                            os.path.join(
                                val_label_dir,
                                app_dir + "_" + file.replace("json", "txt"),
                            ),
                            label2id,
                            ws,
                            image_width,
                            image_height,
                        )

                        for shape in data["shapes"]:
                            new_annot = {}
                            new_annot["image_name"] = os.path.join(
                                ground_truth_dir, app_dir, image_path
                            )
                            x1, y1 = shape["points"][0]
                            x2, y2 = shape["points"][1]
                            new_annot["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                            new_annot["category"] = shape["label"]
                            test_annotation_list.append(new_annot)

                        shutil.copy(
                            os.path.join(ground_truth_dir, app_dir, image_path),
                            os.path.join(val_img_dir, app_dir + "_" + image_path),
                        )

        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": label2id[label], "name": label} for label in label2id
            ],
        }
        create_coco_json(
            annotation_list=train_annotation_list,
            label2id=label2id,
            output_json_path=args.json_train,
            coco_format=coco_format,
        )

        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": label2id[label], "name": label} for label in label2id
            ],
        }
        create_coco_json(
            annotation_list=test_annotation_list,
            label2id=label2id,
            output_json_path=args.json_test,
            coco_format=coco_format,
        )

    if args.do_train:
        model = YOLO(f"./{args.model_name}.pt")
        results = model.train(
            name=f"{args.model_name}",
            data=args.config,
            epochs=args.epochs,
            batch=args.batch_size,
            lr0=args.lr,
            lrf=args.lr,
        )

    if args.do_test:
        model = YOLO(f"./runs/detect/{args.model_name}/weights/best.pt")
        with open(args.json_test, "rb") as handle:
            coco_format = json.load(handle)

        image_dict = {img["file_name"]: img["id"] for img in coco_format["images"]}
        category_dict = {cat["name"]: cat["id"] for cat in coco_format["categories"]}

        predictions = []

        for ct, app_dir in enumerate(sorted(os.listdir(ground_truth_dir))):
            for file in os.listdir(os.path.join(ground_truth_dir, app_dir)):
                if (
                    file.endswith("png")
                    or file.endswith("jpeg")
                    or file.endswith("jpg")
                ):
                    image_name = os.path.join(ground_truth_dir, app_dir, file)

                    if image_name in image_dict:
                        results = model.predict(image_name, max_det=100, conf=0.01)
                        boxes = results[0].boxes
                        scores = boxes.conf.cpu().numpy()
                        classes = [results[0].names[i] for i in boxes.cls.cpu().numpy()]
                        xyxys = boxes.xyxy.cpu().numpy()

                        # Add prediction info
                        for it in range(len(scores)):
                            x1, y1, x2, y2 = map(
                                float, xyxys[it]
                            )  # Convert numpy types to Python floats
                            prediction = {
                                "image_id": image_dict[image_name],
                                "category_id": category_dict[classes[it]],
                                "bbox": [
                                    x1,
                                    y1,
                                    x2 - x1,
                                    y2 - y1,
                                ],  # [x, y, width, height]
                                "score": float(
                                    scores[it]
                                ),  # Ensure score is provided in annotation_list
                            }
                            predictions.append(prediction)

        # Write the prediction data to a JSON file
        with open(
            f"./datasets/80app_test_predict_{args.model_name}.json", "w"
        ) as json_file:
            json.dump(predictions, json_file)

        """Evaluate results"""
        cocoGt = COCO(args.json_test)  # Load the ground truth
        cat_names = {
            cat["id"]: cat["name"] for cat in cocoGt.loadCats(cocoGt.getCatIds())
        }
        cocoDt = cocoGt.loadRes(
            f"./datasets/80app_test_predict_{args.model_name}.json"
        )  # Load your results
        print(f"Number of testing samples = {len(cocoDt.dataset['images'])}")
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.iouThrs = np.asarray([0.5])
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
