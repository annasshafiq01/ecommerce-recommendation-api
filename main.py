# E-commerce recommendation engine implemented using FAST API. It authorizes the access of API using API key authenciation.
# Run http://localhost:8000/docs for better demonstration via swagger.
# Read README.md from the repo for help

import os
import secrets
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings('ignore')

# Make an .env file containing API_KEY in the same folder as main.py to use enviromental variable
# Format:
# API_KEY=api_key_123
load_dotenv()

API_KEY = os.environ.get("API_KEY", "test-key-100")
# Fallback key will be used if there is no .env file

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# verify_api_key() is called whenever there is a request for recommendations
def verify_api_key(provided_key: str = Security(api_key_header)):
    if provided_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key.")
    if not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key.")
    return provided_key

# Global variables to be accessed the API endpoint 
matrix = None
knn_model = None

# Lifespan will start whenever the API is loaded and keeps running until the server is turned off.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global matrix, knn_model

    df = pd.read_csv("data.csv", encoding='unicode_escape')

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]

    top_items = df['Description'].value_counts().head(1500).index
    df = df[df['Description'].isin(top_items)]

    matrix = df.pivot_table(
        index='CustomerID', columns='Description', values='Quantity', aggfunc='sum'
    ).fillna(0)
    matrix = (matrix > 0).astype(int)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
    knn_model.fit(matrix.values)

    yield


app = FastAPI(title="E-Commerce Recommendation Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Recommendation(BaseModel):
    product: str
    score: int


class RecommendationResponse(BaseModel):
    customer_id: float
    recommendations: list[Recommendation]

# Default message for whenever "/" i.e., homepage is accessed
@app.get("/")
def root():
    return {"message": "Recommendation API is running. Try /recommend/{customer_id}"}

# Actual request for the recommendations for a specific customer_id that is fetched from the URL
@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
def recommend(customer_id: float, api_key: str = Security(verify_api_key)):
    if customer_id not in matrix.index:
        raise HTTPException(
            status_code=404,
            detail=f"Customer ID {customer_id} not found in the database."
        )

    # Generating recommendations for customer_id
    user_index = matrix.index.get_loc(customer_id)
    user_vector = matrix.iloc[user_index].values.reshape(1, -1)

    distances, indices = knn_model.kneighbors(user_vector)
    neighbor_indices = indices.flatten()[1:]

    neighbor_data = matrix.iloc[neighbor_indices]
    neighbor_item_frequency = neighbor_data.sum(axis=0)

    target_user_history = matrix.iloc[user_index]
    unseen_products = neighbor_item_frequency[target_user_history == 0]

    top_recommendations = unseen_products.sort_values(ascending=False)
    top_recommendations = top_recommendations[top_recommendations > 0].head(5)

    # results list will be returned in .json format that contains recommended products and their score.
    # (score is the number of neighbours that have actually bought the product)
    results = [
        Recommendation(
            product=product,
            score=int(freq)) for 
            product, freq in top_recommendations.items()
    ]

    return RecommendationResponse(customer_id=customer_id, recommendations=results)