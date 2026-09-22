# E-commerce recommendation engine built with FastAPI. Access is authenticated using an API key.
# Run http://localhost:8000/docs for interactive documentation via Swagger UI.
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

# Create a .env file in the same folder as main.py to set the API_KEY environment variable
# Format:
# API_KEY=api_key_123
load_dotenv()

API_KEY = os.environ.get("API_KEY", "test-key-100")
# Fallback key is used if API_KEY is not set via .env or another environment variable

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# verify_api_key() is called whenever there is a request for recommendations
def verify_api_key(provided_key: str = Security(api_key_header)):
    if provided_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key.")
    if not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key.")
    return provided_key

# Global variables to be accessed by the API endpoints
matrix = None
knn_model = None

# Runs once at startup to load data and train the model; pauses here (via yield) until the server shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global matrix, knn_model

    df = pd.read_csv("data.csv", encoding='unicode_escape')

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]

    top_items = df['Description'].value_counts().head(1500).index
    df = df[df['Description'].isin(top_items)]

   # Ratings are derived in this block 
   # If there is no ratings column in the dataset otherwise it'll be skipped.
    if 'Rating' not in df.columns:
        implicit = df.groupby(['CustomerID', 'Description']).agg(
            total_quantity=('Quantity', 'sum'),
            frequency=('InvoiceNo', 'nunique')
        ).reset_index()

        # Frequency has double the weight of quantity of product purchased.
        implicit['raw_score'] = implicit['total_quantity'] + (implicit['frequency'] * 2)

        #Divides the raw_score into 5 groups and a rating from 1-5 is assigned to each product based on the group it is in.
        implicit['Rating'] = pd.qcut(
            implicit['raw_score'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        ).astype(int)

        df = df.merge(implicit[['CustomerID', 'Description', 'Rating']], on=['CustomerID', 'Description'])

    matrix = df.pivot_table(
        index='CustomerID', columns='Description', values='Rating', aggfunc='mean'
    ).fillna(0)

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
    score: float


class RecommendationResponse(BaseModel):
    customer_id: float
    recommendations: list[Recommendation]

# Default message for whenever "/" i.e., homepage is accessed
@app.get("/")
def root():
    return {"message": "Recommendation API is running. Try /recommend/{customer_id}"}

# Returns recommendations for the customer_id provided in the URL path
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
    # Calculating the average rating for each product based on the ratings
    neighbor_avg_rating = neighbor_data.where(neighbor_data > 0).mean(axis=0).fillna(0)

    target_user_history = matrix.iloc[user_index]
    unseen_products = neighbor_avg_rating[target_user_history == 0]

    top_recommendations = unseen_products.sort_values(ascending=False)
    top_recommendations = top_recommendations[top_recommendations > 0].head(5)

    # results list will be returned in JSON format that contains recommended products and their score.
    # (score is the average rating given to this product by similar customers)
    results = [
        Recommendation(
            product=product,
            score=round(float(rating), 2)) for
            product, rating in top_recommendations.items()
    ]

    return RecommendationResponse(customer_id=customer_id, recommendations=results)