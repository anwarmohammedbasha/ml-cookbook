# Customer Churn

## Local setup

From this directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate the synthetic data and train the model:

```bash
python src/data_processor.py
python src/train.py
```

Run the test suite:

```bash
python -m pytest -q
```

Start the API locally:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Docker

Build and run the API container from this directory:

```bash
docker build -t customer-churn-api .
docker run --rm -p 8000:8000 customer-churn-api
```

The image generates training data and the model during its build. The API is available at `http://localhost:8000`.