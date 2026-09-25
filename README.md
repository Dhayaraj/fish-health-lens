# Fish Health Lens

Fish disease classification app using a DINOv2 image model, a FastAPI backend,
and a static browser UI. The app returns a possible disease, confidence scores,
experimental severity, and reference treatment information.

## Run the existing app

The trained DINOv2 checkpoint is included in `model/`. You do not need to train
the model just to run the app.

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn api.main:app --reload
```

In a second terminal:

```bash
python3 -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500` in your browser. The API docs are at
`http://127.0.0.1:8000/docs`.

### Windows PowerShell

```bash
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

In a second PowerShell window:

```powershell
cd path\to\fish-health-lens
python -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500`.

## Train again

To download and organize the Kaggle dataset:

```bash
python download_data.py
python check_data.py
```

Then retrain the DINOv2 classifier:

```bash
python train_dino.py
```

Training is optional when using the included checkpoint. Kaggle authentication
may be required for downloading data.

## Current limitations

Severity is an experimental color-feature model based on automatically generated
labels, not human-verified severity labels. Treatment text is reference information
only and must be confirmed by an aquatic veterinarian. YOLO detection and SAM 2
segmentation require additional annotations/checkpoints and are not part of the
working classification flow.

The API runs DINOv2 in an isolated worker process. It uses the current Python
interpreter, so the same code works on Windows and macOS.

## Project layout

```text
api/main.py          FastAPI server
dino_worker.py       Isolated DINOv2 inference process
train_dino.py        DINOv2 training
train_severity.py    Experimental severity training
frontend/             Browser UI
model/                Trained model artifacts
treatment_data.json   Treatment reference data
```

