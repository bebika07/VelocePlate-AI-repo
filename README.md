# Number Plate Recognition (ANPR)

This repository contains a Flask-based automatic number plate recognition system built with YOLOv8, SORT tracking, and EasyOCR.

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Flask web app:

```bash
python app.py
```

3. Open the browser:

```
http://127.0.0.1:5000
```

4. Upload a supported video in the dashboard and start processing.

## Notes

- The app uses `yolov8n.pt` for vehicle detection.
- `best.pt` may be used if present in the project root for license plate detection.
- The upload page accepts `MP4`, `AVI`, `MOV`, and `MKV` files.
- Generated outputs are stored under `web_results/` and are ignored by git.

## GitHub upload

To upload this project to GitHub:

1. Create a new repository on GitHub.
2. Add the remote locally:
   ```bash
git remote add origin https://github.com/<your-user>/<repo-name>.git
```
3. Push the initial commit:
   ```bash
git push -u origin master
```
