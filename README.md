# E-Commerce Recommendation Engine — API

A KNN-based product recommendation API built with FastAPI. Given a customer ID, it returns products purchased by similar customers that this customer hasn't bought yet, plus a separate endpoint for surfacing never-before-sold products using description-based matching.

## How it works

### Main recommendations (`/recommend/{customer_id}`)
- Trained on the [Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) (`data.csv`).
- Since the dataset has no explicit customer ratings, an implicit rating (1.00–5.00) is derived per customer-product pair from purchase behavior: total quantity bought plus repeat-order frequency, scaled by percentile rank into a continuous decimal rating. If the dataset already has a `Rating` column, real ratings are used instead and this derivation is skipped (or only fills in gaps where a purchase has no rating).
- Builds a customer × product matrix of these ratings, stored as a sparse matrix for memory efficiency.
- Uses cosine-similarity K-Nearest Neighbors (20 neighbors) to find similar customers.
- Ranks unseen products by the **sum** of ratings given by neighbors who bought them (so a product needs both multiple buyers and good ratings to rank highly, not just one high rating from a single neighbor).
- Returns up to 5 recommendations, each with the average neighbor rating and the product's total number of orders.

### New arrivals (`/new-arrivals/{customer_id}`)
- Collaborative filtering (above) can never recommend a product with zero purchase history — it has no data to compare.
- This endpoint instead compares a separate catalog of not-yet-sold products (`new_arrivals.csv`, description text only) against the customer's own purchase history using TF-IDF text similarity.
- Returns up to 5 new/unsold products whose descriptions most closely match what the customer already buys.
- If `new_arrivals.csv` doesn't exist, this endpoint returns an empty list rather than erroring.

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
   **The server will refuse to start if `API_KEY` is not set** — there is no fallback/default key.
6. Set `DATA_PATH` and `NEW_ARRIVALS_PATH` in `.env` to point to the CSV files, which live in the `Datasets/` folder:
   ```
   DATA_PATH=Datasets/data.csv
   NEW_ARRIVALS_PATH=Datasets/new_arrivals.csv
   ```
   (Both default to `data.csv` and `new_arrivals.csv` in the project root if not set — so this step is required given the current folder layout.)
7. Run the server:
   ```
   uvicorn main:app --reload
   ```
8. Open `http://127.0.0.1:8000/docs` to test it interactively.

## Authentication

Every request to `/recommend/{customer_id}` and `/new-arrivals/{customer_id}` must include an API key header:

```
X-API-Key: <your key>
```

| Scenario | Response |
|---|---|
| Missing header | `401 Unauthorized` |
| Wrong key | `403 Forbidden` |
| Valid key | `200 OK` with results |

## Rate limiting

Both endpoints above are limited to **60 requests per minute per caller IP address**. Exceeding this returns a `429 Too Many Requests` response. This protects the server from being overwhelmed by a runaway script or abusive traffic.

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
    { "product": "3 HOOK PHOTO SHELF ANTIQUE WHITE", "rating": 4.82, "orders": 312 },
    { "product": "ORGANISER WOOD ANTIQUE WHITE", "rating": 4.72, "orders": 198 }
  ]
}
```

- `rating`: average rating given to this product by the customer's similar neighbors (1.00–5.00).
- `orders`: total number of separate orders this product has been sold in, across all customers.

**Error response (404)** — customer ID not found in the dataset:
```json
{ "detail": "Customer ID 999999.0 not found in the database." }
```

### `GET /new-arrivals/{customer_id}`
Returns up to 5 not-yet-sold products that best match the customer's existing purchase history, by description similarity.

**Example response (200)**
```json
{
  "customer_id": 17850,
  "new_arrivals": [
    { "product": "YELLOW HANGING HEART T-LIGHT HOLDER", "similarity": 0.77 },
    { "product": "WHITE METAL LANTERN WITH HEART DESIGN", "similarity": 0.74 }
  ]
}
```

- `similarity`: a 0–1 text-similarity score, not a purchase-based rating (these products have never been sold).
- Returns an empty `new_arrivals` list if the customer has no purchase history, or if `new_arrivals.csv` is missing.

## Configuration (environment variables)

| Variable | Required? | Default | Purpose |
|---|---|---|---|
| `API_KEY` | Yes | — (no fallback) | API authentication key. Server refuses to start if missing. |
| `DATA_PATH` | No | `data.csv` | Path to the transaction dataset. Set to `Datasets/data.csv` for this project's layout. |
| `NEW_ARRIVALS_PATH` | No | `new_arrivals.csv` | Path to the new-arrivals product catalog. Set to `Datasets/new_arrivals.csv` for this project's layout. |

## Known limitations

- Only customers present in the training dataset can get `/recommend` results (no "cold start" support for brand-new customers with zero purchase history).
- `/new-arrivals` requires the customer to have at least one prior purchase to compare against — a customer with no history gets an empty list there too.
- Data files must be reachable from wherever the server runs (see `DATA_PATH`/`NEW_ARRIVALS_PATH` above); the server does not auto-refresh if the underlying CSV changes — it needs a restart to pick up new data.
