from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .youtube_service import (
    get_channel_stats,
    get_recent_video_ids,
    get_videos_stats,
    get_comments_for_videos,
)
from .scoring import calculate_quantitative_metrics
from .ai_service import analyze_comments_batch
from .brand_fit import calculate_brand_fit_score
from .risk_aggregation import calculate_final_risk_score
from .explanation_service import generate_risk_explanation

app = FastAPI(title="Collab Risk Score Model API")

# Dozvoljava frontend-u (koji ce raditi na drugom portu) da poziva ovaj API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # za razvoj je ok, kasnije mozemo suziti na konkretan frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    channel_handle: str          # npr. "@mkbhd"
    brand_description: str       # opis brenda korisnika


@app.get("/")
def read_root():
    return {"status": "Backend radi ispravno"}


@app.post("/analyze")
def analyze_channel(request: AnalyzeRequest):
    """
    Glavni endpoint - prima naziv kanala i opis brenda,
    vraca kompletnu risk analizu.
    """
    try:
        stats = get_channel_stats(request.channel_handle)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)
    comments = get_comments_for_videos(video_ids, max_per_video=10)

    quant_metrics = calculate_quantitative_metrics(stats, videos)
    sentiment_result = analyze_comments_batch(comments)
    brand_fit_result = calculate_brand_fit_score(request.brand_description, stats, videos)

    final_result = calculate_final_risk_score(quant_metrics, sentiment_result, brand_fit_result)

    explanation = generate_risk_explanation(
        stats["title"], request.brand_description, final_result
    )

    return {
        "channel": {
            "title": stats["title"],
            "subscriber_count": stats["subscriber_count"],
            "video_count": stats["video_count"],
        },
        "quantitative_metrics": quant_metrics,
        "sentiment": sentiment_result,
        "brand_fit": brand_fit_result,
        "risk_assessment": final_result,
        "ai_explanation": explanation,
    }