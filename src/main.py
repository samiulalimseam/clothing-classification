from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from recommendation_model import get_top_n_recommendations

app = FastAPI()

class RecommendationRequest(BaseModel):
    productId: str
    customerId: str
    domain: str

class RecommendationResponse(BaseModel):
    recommendedProducts: List[str]

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_products(request: RecommendationRequest):
    user_id = int(request.customerId)
    top_n = 5  # Number of recommendations
    recommended_products = get_top_n_recommendations(user_id, n=top_n)
    return RecommendationResponse(recommendedProducts=recommended_products)
