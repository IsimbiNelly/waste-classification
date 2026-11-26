# Waste Classification: End-to-End ML Pipeline

An image classification app that identifies the type of waste in a photo (cardboard, glass, metal, paper, plastic, trash). It includes a prediction API, a web page, bulk data upload saved to a SQLite database, retraining, Docker deployment, and a Locust load test.

## Links
- **Video demo:** https://youtu.be/ll4yagHTPPw
- **Docker Hub image:** https://hub.docker.com/r/isimbinelly/waste-classification
- **Deployment:** packaged as a Docker image with a web UI (see "Run with Docker" below), running at http://localhost:8000. Free cloud tiers either lack the memory TensorFlow needs or require a paid plan, so the app is delivered as a public Docker image that can run on any container host.

## Project description
- **Data:** the Kaggle "Garbage Classification" dataset (cchangcs), 2,527 images in 6 classes.
- **Model:** MobileNetV2 (pretrained on ImageNet) with data augmentation, dropout, L2 regularization, and early stopping. Class weights handle the imbalance (trash has the fewest images).
- **Test results:** about 85% accuracy at first, and about 90% after retraining (accuracy, loss, precision, recall, F1, and confusion matrix are in the notebook).
- **App features:** predict one image, see uptime, view data charts, upload many images of one class (logged in a SQLite database), and retrain with a button (retraining also starts automatically when 50 or more new images are waiting).
- **Database:** a SQLite file (`data/app.db`, created automatically) logs every uploaded image and every retraining run. See http://localhost:8000/db/summary.

## Project structure
```
waste-classification/
├── README.md
├── notebook/waste_classification.ipynb
├── src/
│   ├── preprocessing.py   (data split script, same logic as prepare_data.py)
│   ├── prepare_data.py
│   ├── model.py
│   ├── prediction.py
│   ├── retrain.py
│   ├── database.py
│   ├── api.py
│   └── static/index.html
├── data/train, data/test
├── models/waste_model.h5, class_names.json, metrics.json
├── results/               (Locust screenshots)
├── Dockerfile, docker-compose.yml, nginx.conf, locustfile.py
└── requirements.txt
```

## Run with Docker (pull from Docker Hub)
The image is public on Docker Hub: https://hub.docker.com/r/isimbinelly/waste-classification

```
docker pull isimbinelly/waste-classification:latest
docker run -p 8000:8000 isimbinelly/waste-classification:latest
```

Then open http://localhost:8000. The first start can take up to a minute while TensorFlow loads the model. No environment variables or accounts are needed.

To build the image yourself instead:
```
docker build -t waste-app .
docker run -p 8000:8000 --name waste1 waste-app
```

## Setup (run locally without Docker)
1. Install Python 3.12 or newer.
2. Clone the repo and open a terminal in its folder.
3. Create and activate a virtual environment, then install the libraries:
```
python -m venv venv
source venv/Scripts/activate      # Windows (Git Bash). On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```
4. (Only if `data/train` and `data/test` are missing) Download the Kaggle dataset into `data/raw` and run `python src/prepare_data.py`.
5. Start the app and open http://localhost:8000:
```
uvicorn api:app --app-dir src --port 8000
```

## Flood request simulation (Locust)
Run several containers behind an nginx load balancer:
```
docker compose up -d --scale app=3
docker compose restart nginx
locust -f locustfile.py --host http://localhost:8000
```
Open http://localhost:8089 and start the test with 10 users and a ramp up of 2, for 1 minute.

Results for `POST /predict` (same settings each time, run on a laptop):

| Containers | Median (ms) | Average (ms) | 95th percentile (ms) | Requests/s | Failures |
|---|---|---|---|---|---|
| 1 | 310 | 571 | 1,200 | 4.5 | 0 |
| 2 | 230 | 329 | 990 | 4.7 | 0 |
| 3 | 190 | 234 | 470 | 4.8 | 0 |

More containers lowered the response time (average down about 59%, 95th percentile down about 61%). Requests per second stayed near 5 because 10 users who wait 1 to 2 seconds between requests cannot send more than that.

Screenshots: see the `results/` folder.

## Retraining
1. Upload several images of one class on the web page, using the matching class in the dropdown. The images are saved to disk and logged in the SQLite database.
2. Click **Retrain model**, or wait for the automatic trigger at 50 or more new images.
3. The saved model is loaded as the starting point, trained on old plus new images, tested on the test set, and saved. The previous model is kept as a backup, the run is logged in the database, and the new scores appear on the page.