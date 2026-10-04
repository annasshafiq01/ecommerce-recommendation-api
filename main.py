# E-commerce recommendation engine built with FastAPI. Access is authenticated using an API key.
# Run http://localhost:8000/docs for interactive documentation via Swagger UI.
# Read README.md from the repo for help

import os
import secrets
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import warnings

warnings.filterwarnings('ignore')

# Create a .env file in the same folder as main.py to set the API_KEY environment variable
# Format:
# API_KEY=api_key_123
load_dotenv()

API_KEY = os.environ.get("API_KEY")

# A runtime error is raised if there is no API_KEY in the env
if not API_KEY:
    raise RuntimeError(
        "API_KEY environment variable is not set. Create a .env file in this folder "
        "containing a line like API_KEY=your-real-secret-key before starting the server."
    )

# The csv path would be read from the env alongwith the API_KEY.
DATA_PATH = os.environ.get("DATA_PATH", "data.csv")

# The products that have not been sold yet (new arrivals) stored in a different csv than
# the data.csv
NEW_ARRIVALS_PATH = os.environ.get("NEW_ARRIVALS_PATH", "new_arrivals.csv")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# verify_api_key() is called whenever there is a request for recommendations
def verify_api_key(provided_key: str = Security(api_key_header)):
    if provided_key is None:
        raise HTTPException(status_code=401, detail="Missing API Key.")
    if not secrets.compare_digest(provided_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API Key.")
    return provided_key

# limiter limits the requests to 60 per minute per a single IP address.
limiter = Limiter(key_func=get_remote_address)

matrix = None
knn_model = None
product_orders = None


# Global variables for creating a sparse matrix
customer_ids = None
product_names = None
customer_index = None

# Global variables for new arrivals product catalog and vectoriser
new_arrivals_catalog = None
new_arrivals_vectorizer = None
new_arrivals_vectors = None

# Runs once at startup to load data and train the model; pauses here (via yield) until the server shuts down.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global matrix, knn_model, product_orders, customer_ids, product_names, customer_index
    global new_arrivals_catalog, new_arrivals_vectorizer, new_arrivals_vectors

    df = pd.read_csv(DATA_PATH, encoding='unicode_escape')

    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]

    top_items = df['Description'].value_counts().head(1500).index
    df = df[df['Description'].isin(top_items)]

    # Ratings are derived in this block if there is no ratings column in the dataset, otherwise it'll be skipped.
    if 'Rating' not in df.columns:
        implicit = df.groupby(['CustomerID', 'Description']).agg(
            total_quantity=('Quantity', 'sum'),
            frequency=('InvoiceNo', 'nunique')
        ).reset_index()

        # Frequency has double the weight of quantity of product purchased.
        implicit['raw_score'] = implicit['total_quantity'] + (implicit['frequency'] * 2)

        percentile = implicit['raw_score'].rank(pct=True)
        implicit['Rating'] = (1 + 4 * percentile).round(2)

        df = df.merge(implicit[['CustomerID', 'Description', 'Rating']], on=['CustomerID', 'Description'])

    # This block generates ratings for products that are bought but do not have a rating. The original 
    # ratings of other products will remain untouched.
    elif df['Rating'].isna().any():
        implicit = df.groupby(['CustomerID', 'Description']).agg(
            total_quantity=('Quantity', 'sum'),
            frequency=('InvoiceNo', 'nunique')
        ).reset_index()

        implicit['raw_score'] = implicit['total_quantity'] + (implicit['frequency'] * 2)

        percentile = implicit['raw_score'].rank(pct=True)
        implicit['derived_rating'] = (1 + 4 * percentile).round(2)

        df = df.merge(implicit[['CustomerID', 'Description', 'derived_rating']], on=['CustomerID', 'Description'])
        df['Rating'] = df['Rating'].fillna(df['derived_rating'])

    product_orders = df.groupby('Description')['InvoiceNo'].nunique()

    temp_matrix = df.pivot_table(
        index='CustomerID', columns='Description', values='Rating', aggfunc='mean'
    ).fillna(0).astype('float32')


    # Creates a sparse matrix i.e., fills only the non zero values while the row and columns indices 
    # are stored in customer_ids and product_names as there is no column and row labels in a sparse matrix
    matrix = csr_matrix(temp_matrix.values)
    customer_ids = temp_matrix.index.to_numpy()
    product_names = temp_matrix.columns.to_numpy()
    customer_index = {cust_id: row for row, cust_id in enumerate(customer_ids)}

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=21)
    knn_model.fit(matrix)

    # checking whether if there is a record of new arrivals 
    if os.path.exists(NEW_ARRIVALS_PATH):
        new_arrivals_catalog = pd.read_csv(NEW_ARRIVALS_PATH)
        # TF-IDF vectorizer is used to compare the product descriptions for generating recommendations
        new_arrivals_vectorizer = TfidfVectorizer()
        new_arrivals_vectors = new_arrivals_vectorizer.fit_transform(new_arrivals_catalog['Description'])
    else:
        new_arrivals_catalog = pd.DataFrame(columns=['Description'])
        new_arrivals_vectorizer = None
        new_arrivals_vectors = None

    yield


app = FastAPI(title="E-Commerce Recommendation Engine", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Recommendation(BaseModel):
    product: str
    rating: float
    orders: int


class RecommendationResponse(BaseModel):
    customer_id: float
    recommendations: list[Recommendation]


class NewArrival(BaseModel):
    product: str
    similarity: float


class NewArrivalResponse(BaseModel):
    customer_id: float
    new_arrivals: list[NewArrival]

# Default message for whenever "/" i.e., homepage is accessed
@app.get("/")
def root():
    return {"message": "Recommendation API is running. Try /recommend/{customer_id}"}

# Returns recommendations for the customer_id provided in the URL path
# @limiter.limit("60/minute") limits max 60 requests per a minute for each IP address
@app.get("/recommend/{customer_id}", response_model=RecommendationResponse)
@limiter.limit("60/minute")
def recommend(request: Request, customer_id: float, api_key: str = Security(verify_api_key)):
    # since "matrix" is now a sparse and no longer has its own row and column labels.
    if customer_id not in customer_index:
        raise HTTPException(
            status_code=404,
            detail=f"Customer ID {customer_id} not found in the database."
        )

    # Generating recommendations for customer_id
    user_index = customer_index[customer_id]
    user_vector = matrix[user_index]

    distances, indices = knn_model.kneighbors(user_vector)
    neighbor_indices = indices.flatten()[1:]

    # only the neighbor data is accessed and converted into a dataframe for saving memory
    neighbor_data = pd.DataFrame(matrix[neighbor_indices].toarray(), columns=product_names)
    # Calculating the average rating for each product based on the ratings
    neighbor_avg_rating = neighbor_data.where(neighbor_data > 0).mean(axis=0).fillna(0)

    # Instead of blindly recommending products best sellers, it sorts the recommendations based on the formula
    # total neighbors that bought the product x average rating 
    neighbor_sum_rating = neighbor_data.sum(axis=0)

    # user purchase data is accessed in the same way as neighbor_data above
    target_user_history = pd.Series(matrix[user_index].toarray().flatten(), index=product_names)
    unseen_products = neighbor_sum_rating[target_user_history == 0]

    top_recommendations = unseen_products.sort_values(ascending=False)

    # top_recommendations contains the sum of rating for each product and the recommendations will be sorted of it basis. 
    top_recommendations = top_recommendations[top_recommendations > 0].head(5)

    # results list will be returned in JSON format that contains recommended products and their score.   
    results = [
        Recommendation(
            product=product,
            rating=round(float(neighbor_avg_rating[product]), 2),
            orders=int(product_orders[product])
        ) 
        for product in top_recommendations.index
    ]

    return RecommendationResponse(customer_id=customer_id, recommendations=results)


# Returns newly arrived products as recommendations for the customer_id provided in the URL path
@app.get("/new-arrivals/{customer_id}", response_model=NewArrivalResponse)
@limiter.limit("60/minute")
def new_arrivals(request: Request, customer_id: float, api_key: str = Security(verify_api_key)):
    if customer_id not in customer_index:
        raise HTTPException(
            status_code=404,
            detail=f"Customer ID {customer_id} not found in the database."
        )

    # checks if there even is a new arrivals catalog or not
    if new_arrivals_vectors is None or len(new_arrivals_catalog) == 0:
        return NewArrivalResponse(customer_id=customer_id, new_arrivals=[])

    # Accessing the customer's previously purchased products record generating recommendations
    user_index = customer_index[customer_id]
    purchased_products = matrix[user_index].toarray().flatten() > 0
    customer_history = list(product_names[purchased_products])

    if not customer_history:
        return NewArrivalResponse(customer_id=customer_id, new_arrivals=[])

    # using the same vectorizer to compare the new arrivals with the customer's history
    history_vectors = new_arrivals_vectorizer.transform(customer_history)

    similarity_scores = cosine_similarity(new_arrivals_vectors, history_vectors).max(axis=1)

    ranked = new_arrivals_catalog.copy()
    ranked['similarity'] = similarity_scores
    ranked = ranked[ranked['similarity'] > 0].sort_values('similarity', ascending=False).head(5)

    results = [
        NewArrival(product=row['Description'], similarity=round(float(row['similarity']), 2))
        for _, row in ranked.iterrows()
    ]

    return NewArrivalResponse(customer_id=customer_id, new_arrivals=results)