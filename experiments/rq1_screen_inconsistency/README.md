»·¾³ÅäÖÃ£º
```
conda env create -f environment.yml
conda activate guipilot
```

Copy the files in rq1 to the GUIPilot path
```
cd experiments/rq1_screen_inconsistency
cp main.py utils.py ../../
cp -r mutate ../../
```
Set the dataset path
```
cd ../../
echo "DATASET_PATH="./datasets"
GUIPILOT_DETECTOR_HF_REPO="code-philia/GUIPilot"
GUIPILOT_DETECTOR_HF_FILE="widget_detector.pt"
GUIPILOT_DETECTOR_DOWNLOAD_URL="https://huggingface.co/code-philia/GUIPilot/resolve/main/widget_detector.pt"" > .env
```
Run the main.py
```
python main.py
```
File evaluation.csv is produced.