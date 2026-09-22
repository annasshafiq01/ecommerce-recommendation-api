# E-Commerce Recommendation Engine — API

A KNN-based product recommendation API built with FastAPI. Given a customer ID, it returns products purchased by similar customers that this customer hasn't bought yet.

## How it works

- Trained on the [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (`data.csv`).
- Since the dataset has no explicit customer ratings, an implicit rating (1.00–5.00) is derived per customer-product pair from purchase behavior: total quantity bought plus repeat-order frequency, scaled by percentile rank into a continuous decimal rating (e.g. `4.29`, `3.61`) rather than flat integer buckets.
- If a dataset with a real `Rating` column is used instead, this derivation step is skipped automatically and the real ratings are used.
- Builds a customer × product matrix of these ratings.
- Uses cosine-similarity K-Nearest Neighbors to find the 5 most similar customers.
- Recommends products those neighbors rated highly that the target customer hasn't bought. The returned `score` is the average rating given to that product by similar customers.

## Setup

1. Clone the repo and `cd` into it.
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate it:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set your API key. Copy `.env.example` to `.env` and set a real value:
   ```
   API_KEY=paste-a-long-random-string-here
   ```
   Generate one with:
   ```
   python -c "import secrets; print(secrets.token_hex(32))"
   ```
6. Run the server:
   ```
   uvicorn main:app --reload
   ```
7. Open `http://127.0.0.1:8000/docs` to test it interactively.

## Authentication

Every request to `/recommend/{customer_id}` must include an API key header:

```
X-API-Key: <your key>
```

| Scenario | Response |
|---|---|
| Missing header | `401 Unauthorized` |
| Wrong key | `403 Forbidden` |
| Valid key | `200 OK` with recommendations |

## Endpoints

### `GET /`
Health check. No auth required.

### `GET /recommend/{customer_id}`
Returns up to 5 product recommendations for the given customer ID.

**Headers**
```
X-API-Key: <your key>
```

**Example request**
```
GET /recommend/17850
X-API-Key: your-real-secret-key-here
```

**Example response (200)**
```json
{
  "customer_id": 17850,
  "recommendations": [
    { "product": "3 HOOK PHOTO SHELF ANTIQUE WHITE", "score": 4.82 },
    { "product": "ORGANISER WOOD ANTIQUE WHITE", "score": 4.72 }
  ]
}
```

**Error response (404)** — customer ID not found in the dataset:
```json
{ "detail": "Customer ID 999999.0 not found in the database." }
```

## Known limitations

- Only customers present in the training dataset can get recommendations (no "cold start" support for brand-new customers).
- `data.csv` must be in the same directory as `main.py` when running the server.
