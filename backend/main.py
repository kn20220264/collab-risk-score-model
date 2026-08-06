"""FastAPI aplikacija - Collab Risk Score API."""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .youtube_service import (
    get_channel_stats,
    get_recent_video_ids,
    get_videos_stats,
    get_comments_for_videos,
)
from .scoring import calculate_quantitative_metrics
from .ai_service import analyze_comments_batch
from .brand_fit import calculate_brand_fit_score
from .risk_aggregation import calculate_final_risk_score, calculate_bulk_risk_scores
from .explanation_service import generate_risk_explanation
from .roc_service import get_module_weights, MODULE_RANK_ORDER, MODULE_LABELS
from .risk_aggregation import RISK_CAP_RULES
from .brand_research_service import research_brand
from .transcript_service import get_transcripts_for_videos
from .content_analysis_service import analyze_creator_content

app = FastAPI(
    title="Collab Risk Score API",
    version="1.0.0",
    description=(
        "Akademski prototip alata za procjenu rizika saradnje brendova "
        "sa YouTube kreatorima. Puna metodologija (AHP tezine, risk cap "
        "pravila) je transparentno dostupna preko /api/v1/methodology, "
        "za razliku od komercijalnih alata poput CreatorScore/HypeAuditor."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    channel_handle: str
    brand_description: str


class BulkCreatorRequest(BaseModel):
    channel_handle: str
    brand_description: Optional[str] = None
    brand_name: Optional[str] = None
    force_refresh_brand: bool = False


class BulkAnalyzeRequest(BaseModel):
    creators: list[BulkCreatorRequest]


def _gather_and_score(
    channel_handle: str,
    brand_description: str = None,
    brand_name: str = None,
    force_refresh_brand: bool = False,
) -> dict:
    if not brand_description and not brand_name:
        raise HTTPException(
            status_code=400,
            detail="Potrebno je unijeti ili 'brand_description' ili 'brand_name'.",
        )

    generated_description = None
    brand_from_cache = None
    if not brand_description and brand_name:
        research_result = research_brand(brand_name, force_refresh=force_refresh_brand)
        brand_description = research_result["generated_description"]
        generated_description = brand_description
        brand_from_cache = research_result.get("from_cache", False)

    try:
        stats = get_channel_stats(channel_handle)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)
    transcripts = get_transcripts_for_videos(video_ids, max_videos=6, max_words_per_video=50)
    comments = get_comments_for_videos(video_ids, max_per_video=10)

    quant_metrics = calculate_quantitative_metrics(stats, videos)
    sentiment_result = analyze_comments_batch(comments)
    brand_fit_result = calculate_brand_fit_score(brand_description, stats, videos, transcripts=transcripts)

    content_analysis = analyze_creator_content(stats, videos, transcripts)

    return {
        "stats": stats,
        "videos": videos,
        "transcripts": transcripts,
        "quant_metrics": quant_metrics,
        "sentiment_result": sentiment_result,
        "brand_fit_result": brand_fit_result,
        "brand_description_used": brand_description,
        "brand_description_auto_generated": generated_description is not None,
        "brand_description_from_cache": brand_from_cache,
        "creator_persona": content_analysis.get("creator_persona"),
        "profanity_analysis": content_analysis.get("profanity_analysis"),
        "brand_partners": content_analysis.get("brand_partners"),
    }


def _tier_from_score(score: float) -> str:
    """Opisna kategorija uz numericki skor (analogno CreatorScore 'tier')."""
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Fair"
    else:
        return "Poor"


def _compute_audience_health(quant_metrics: dict, sentiment_result: dict, authenticity_score: float) -> dict:
    """'Audience Health' sekcija: bot_activity_pct iz authenticity_score, toxic_pct iz sentimenta."""
    bot_activity_pct = round(100 - authenticity_score, 1)

    toxic_pct = round(sentiment_result.get("negative_pct", 0), 1)

    if bot_activity_pct < 5 and toxic_pct < 10:
        health_label = "Excellent"
    elif bot_activity_pct < 15 and toxic_pct < 25:
        health_label = "Good"
    elif bot_activity_pct < 30 and toxic_pct < 40:
        health_label = "Fair"
    else:
        health_label = "Poor"

    return {
        "bot_activity_pct": bot_activity_pct,
        "toxic_pct": toxic_pct,
        "health_label": health_label,
        "sentiment_breakdown": {
            "positive_pct": sentiment_result.get("positive_pct", 0),
            "neutral_pct": sentiment_result.get("neutral_pct", 0),
            "negative_pct": sentiment_result.get("negative_pct", 0),
        },
    }


def _compute_content_analyzed(quant_metrics: dict, sentiment_result: dict) -> dict:
    """'Content Analyzed' sekcija - obim podataka koriscenih u analizi."""
    return {
        "posts_analyzed": quant_metrics.get("sample_size", 0),
        "comments_analyzed": sentiment_result.get("sample_size", 0),
        "platforms": 1,  # samo YouTube
    }


@app.get("/")
def read_root():
    return {"status": "Backend radi ispravno", "version": "1.0.0"}


@app.get("/api/v1/methodology")
def get_methodology():
    """Transparentan prikaz metodologije: AHP tezine (sa CR) i risk cap pravila."""
    roc_result = get_module_weights()

    risk_cap_description = [
        {"name": rule["name"], "cap": rule["cap"]}
        for rule in RISK_CAP_RULES
    ]

    return {
        "weighting_method": "ROC (Rank Order Centroid) - vidi roc_service.py za obrazlozenje metoda",
        "modules": MODULE_LABELS,
        "roc": {
            "rank_order": roc_result["rank_order"],
            "weights": roc_result["weights"],
        },
        "risk_cap_mechanism": {
            "type": "conjunctive / non-compensatory (Einhorn, 1970)",
            "description": (
                "Ako je bilo koja pojedinacna metrika u kriticnoj zoni, "
                "finalni skor se ogranicava nezavisno od ponderisanog "
                "zbira ostalih modula."
            ),
            "rules": risk_cap_description,
        },
        "bulk_scoring_note": (
            "Za analizu vise kreatora odjednom (POST /api/v1/creators/bulk), "
            "koriste se entropijske tezine (Hwang & Yoon, 1981) izracunate "
            "iz tog konkretnog uzorka, kao objektivna alternativa fiksnim "
            "ROC tezinama."
        ),
    }


@app.get("/api/v1/creators/youtube/{handle}")
def get_creator_score(
    handle: str,
    brand_description: str = Query(None, description="Detaljan opis brenda (opciono ako je dat brand_name)"),
    brand_name: str = Query(None, description="Samo naziv brenda - opis se generise automatski istrazivanjem"),
    include_explanation: bool = Query(False, description="Da li ukljuciti AI-generisano tekstualno obrazlozenje"),
    force_refresh_brand: bool = Query(
        False,
        description=(
            "Ako je True, zaobilazi kesiran opis brenda (ako postoji) i "
            "generise nov preko web istrazivanja. Default (False) koristi "
            "kesiran opis radi reproducibilnosti brand-fit skora - vidi "
            "poglavlje 4.3."
        ),
    ),
):
    if not handle.startswith("@"):
        handle = f"@{handle}"

    data = _gather_and_score(
        handle,
        brand_description=brand_description,
        brand_name=brand_name,
        force_refresh_brand=force_refresh_brand,
    )

    primary_niche = (data.get("creator_persona") or {}).get("primary_niche")

    risk_result = calculate_final_risk_score(
        data["quant_metrics"], data["sentiment_result"], data["brand_fit_result"],
        subscriber_count=data["stats"]["subscriber_count"],
        primary_niche=primary_niche,
    )

    audience_health = _compute_audience_health(
        data["quant_metrics"], data["sentiment_result"], risk_result["module_scores"]["authenticity"]
    )
    content_analyzed = _compute_content_analyzed(data["quant_metrics"], data["sentiment_result"])

    response = {
        "status": "scored",
        "creator": {
            "platform": "youtube",
            "handle": handle,
            "display_name": data["stats"]["title"],
            "thumbnail_url": data["stats"].get("thumbnail_url"),
            "subscriber_count": data["stats"]["subscriber_count"],
            "video_count": data["stats"]["video_count"],
            "engagement_rate_pct": data["quant_metrics"]["engagement_rate"],
            "score": risk_result["final_score"],
            "tier": _tier_from_score(risk_result["final_score"]),
            "risk_category": risk_result["risk_category"],
            "modules": risk_result["module_scores"],
            "weights_used": risk_result["weights_used"],
            "weights_source": risk_result["weights_source"],
            "triggered_risk_flags": risk_result["triggered_caps"],
            "subscriber_tier": risk_result.get("subscriber_tier"),
            "brand_description_used": data["brand_description_used"],
            "brand_description_auto_generated": data["brand_description_auto_generated"],
            "brand_description_from_cache": data["brand_description_from_cache"],
            "creator_persona": data.get("creator_persona"),
            "profanity_analysis": data.get("profanity_analysis"),
            "brand_partners": data.get("brand_partners"),
            "audience_health": audience_health,
            "content_analyzed": content_analyzed,
            "brand_fit_warning": data["brand_fit_result"].get("warning"),
            "content_profile_level": data["brand_fit_result"].get("content_profile_level"),
        },
    }

    if include_explanation:
        response["creator"]["ai_explanation"] = generate_risk_explanation(
            data["stats"]["title"], data["brand_description_used"], risk_result
        )

    return response


@app.post("/api/v1/creators/bulk")
def bulk_creator_scores(request: BulkAnalyzeRequest):
    """Analiza vise kreatora odjednom, koristi entropijske tezine izracunate iz cijelog batch-a."""
    if len(request.creators) > 25:
        raise HTTPException(
            status_code=400,
            detail="Maksimalno 25 kreatora po bulk zahtjevu (akademski prototip, nema queue infrastrukturu).",
        )

    raw_data = []
    handles = []
    for creator in request.creators:
        handle = creator.channel_handle
        if not handle.startswith("@"):
            handle = f"@{handle}"
        handles.append(handle)

        data = _gather_and_score(
            handle,
            brand_description=creator.brand_description,
            brand_name=creator.brand_name,
            force_refresh_brand=creator.force_refresh_brand,
        )
        raw_data.append(data)

    bulk_results = calculate_bulk_risk_scores(raw_data)

    results = []
    for handle, data, risk_result in zip(handles, raw_data, bulk_results):
        results.append({
            "platform": "youtube",
            "handle": handle,
            "display_name": data["stats"]["title"],
            "score": risk_result["final_score"],
            "tier": _tier_from_score(risk_result["final_score"]),
            "risk_category": risk_result["risk_category"],
            "modules": risk_result["module_scores"],
            "triggered_risk_flags": risk_result["triggered_caps"],
            "subscriber_tier": risk_result.get("subscriber_tier"),
            "brand_description_used": data["brand_description_used"],
            "brand_description_from_cache": data["brand_description_from_cache"],
            "creator_persona": data.get("creator_persona"),
        })

    weights_source = bulk_results[0]["weights_source"] if bulk_results else None
    weights_used = bulk_results[0]["weights_used"] if bulk_results else None

    return {
        "results": results,
        "summary": {
            "total": len(results),
            "weights_source": weights_source,
            "weights_used": weights_used,
        },
    }


# Stari endpoint (deprecated) - zadrzan radi kompatibilnosti sa postojecim frontend-om

@app.post("/analyze")
def analyze_channel(request: AnalyzeRequest):
    """DEPRECATED - koristiti GET /api/v1/creators/youtube/{handle}."""
    data = _gather_and_score(request.channel_handle, brand_description=request.brand_description)

    primary_niche = (data.get("creator_persona") or {}).get("primary_niche")

    final_result = calculate_final_risk_score(
        data["quant_metrics"], data["sentiment_result"], data["brand_fit_result"],
        subscriber_count=data["stats"]["subscriber_count"],
        primary_niche=primary_niche,
    )

    explanation = generate_risk_explanation(
        data["stats"]["title"], request.brand_description, final_result
    )

    audience_health = _compute_audience_health(
        data["quant_metrics"], data["sentiment_result"], final_result["module_scores"]["authenticity"]
    )
    content_analyzed = _compute_content_analyzed(data["quant_metrics"], data["sentiment_result"])

    return {
        "channel": {
            "title": data["stats"]["title"],
            "subscriber_count": data["stats"]["subscriber_count"],
            "video_count": data["stats"]["video_count"],
        },
        "quantitative_metrics": data["quant_metrics"],
        "sentiment": data["sentiment_result"],
        "brand_fit": data["brand_fit_result"],
        "risk_assessment": final_result,
        "ai_explanation": explanation,
        "creator_persona": data.get("creator_persona"),
        "profanity_analysis": data.get("profanity_analysis"),
        "brand_partners": data.get("brand_partners"),
        "audience_health": audience_health,
        "content_analyzed": content_analyzed,
    }
