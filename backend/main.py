"""
FastAPI aplikacija - Collab Risk Score API.

Struktura endpoint-a je namjerno inspirisana komercijalnim API-jem
CreatorScore (creatorscore.io), radi lakseg poredjenja u poglavlju 3
rada (pregled alata):

- GET /api/v1/creators/youtube/{handle}  - analiza jednog kreatora
  (analogno GET /api/v1/creators/{platform}/{handle} kod CreatorScore)
- POST /api/v1/creators/bulk             - analiza vise kreatora odjednom
  (analogno POST /api/v1/creators/bulk kod CreatorScore), koristi
  entropijske tezine izracunate iz tog batch-a
- GET /api/v1/methodology                - transparentan prikaz cijele
  metodologije (AHP matrica, tezine, CR, risk cap pravila) - ovo je
  namjerna razlika u odnosu na CreatorScore i slicne komercijalne
  alate, ciji algoritmi nisu javno objavljeni ("crna kutija"); ovaj
  endpoint je direktan argument za poglavlje 3 (transparentnost kao
  prednost akademskog/otvorenog pristupa nasuprot proprietary alatima).

Zadrzan je i stari POST /analyze endpoint (deprecated) radi
kompatibilnosti sa postojecim frontend-om dok se ne prebaci na novu
strukturu.

Dodato (Creator Persona / Profanity / Brand Partners / Audience
Health / Content Analyzed): prosirenje po uzoru na dodatne "agente"
komercijalnih alata (vidi content_analysis_service.py za detaljno
metodolosko obrazlozenje).
"""

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


# ---------------------------------------------------------------------
# Pydantic modeli
# ---------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    channel_handle: str
    brand_description: str


class BulkCreatorRequest(BaseModel):
    channel_handle: str
    brand_description: str


class BulkAnalyzeRequest(BaseModel):
    creators: list[BulkCreatorRequest]


# ---------------------------------------------------------------------
# Interna pomocna funkcija - prikuplja podatke i racuna module za JEDAN kanal
# ---------------------------------------------------------------------

def _gather_and_score(channel_handle: str, brand_description: str = None, brand_name: str = None) -> dict:
    if not brand_description and not brand_name:
        raise HTTPException(
            status_code=400,
            detail="Potrebno je unijeti ili 'brand_description' ili 'brand_name'.",
        )

    generated_description = None
    if not brand_description and brand_name:
        research_result = research_brand(brand_name)
        brand_description = research_result["generated_description"]
        generated_description = brand_description

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

    return {
        "stats": stats,
        "videos": videos,
        "transcripts": transcripts,
        "quant_metrics": quant_metrics,
        "sentiment_result": sentiment_result,
        "brand_fit_result": brand_fit_result,
        "brand_description_used": brand_description,
        "brand_description_auto_generated": generated_description is not None,
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


def _compute_audience_health(quant_metrics: dict, sentiment_result: dict) -> dict:
    """
    'Audience Health' sekcija (po uzoru na CreatorScore-ovu istoimenu
    kategoriju) - koristi vec postojece metrike, samo preformulisane
    u citljiviji format:
    - bot_activity_pct: proksi preko subscriber_to_view_ratio (visi
      odnos = veca sumnja na neaktivnu/kupljenu bazu, vidi
      scoring.py/risk_aggregation.py za teorijsko obrazlozenje)
    - toxic_pct: % negativnih komentara iz sentiment analize
    """
    ratio = quant_metrics.get("subscriber_to_view_ratio", 0)
    # USKLADJENO sa normalize_authenticity_score (risk_aggregation.py) -
    # oba mjesta sada koriste isti mnozilac (3) za istu osnovnu ideju
    # (subscriber-to-view ratio kao proksi autenticnosti/bot aktivnosti).
    #
    # METODOLOSKO OGRANICENJE (transparentno, ne skriveno): mnozilac 3
    # je HEURISTIKA, ne empirijski izveden broj. Pravi akademski
    # pristup za ovu vrstu proraciuna (Developing a Multimodal Approach
    # to Channel Characterization on YouTube) NE koristi pretpostavljen
    # mnozilac - umjesto toga, racuna Cohen's d effect size za svaku
    # metriku na osnovu STVARNOG dataseta aktivnih i suspendovanih
    # (potvrdjeno anomalnih) kanala, i koristi taj Cohen's d kao tezinu.
    # Takav pristup zahtijeva label-ovan dataset poznato-suspendovanih
    # kanala, sto nadilazi obim ovog rada - vidi napomenu u
    # normalize_authenticity_score (risk_aggregation.py) i poglavlje
    # "Ogranicenja i buduci rad".
    bot_activity_pct = round(min(ratio * 3, 100), 1)

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
    """
    'Content Analyzed' sekcija - samo pregled obima podataka koriscenih
    u analizi (transparentnost, ne nova metrika).
    """
    return {
        "posts_analyzed": quant_metrics.get("sample_size", 0),
        "comments_analyzed": sentiment_result.get("sample_size", 0),
        "platforms": 1,  # samo YouTube - eksplicitno, za razliku od multi-platform alata
    }


# ---------------------------------------------------------------------
# Endpoint-i
# ---------------------------------------------------------------------

@app.get("/")
def read_root():
    return {"status": "Backend radi ispravno", "version": "1.0.0"}


@app.get("/api/v1/methodology")
def get_methodology():
    """
    Transparentan prikaz cijele metodologije agregacije:
    - AHP pairwise comparison matrica i izracunate tezine (sa CR)
    - Risk cap pravila (IF-THEN mapiranje)

    Namjerna razlika u odnosu na komercijalne alate (CreatorScore,
    HypeAuditor), ciji algoritmi nisu javno objavljeni.
    """
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
):
    if not handle.startswith("@"):
        handle = f"@{handle}"

    data = _gather_and_score(handle, brand_description=brand_description, brand_name=brand_name)

    risk_result = calculate_final_risk_score(
        data["quant_metrics"], data["sentiment_result"], data["brand_fit_result"],
        subscriber_count=data["stats"]["subscriber_count"],
    )

    content_analysis = analyze_creator_content(data["stats"], data["videos"], data["transcripts"])
    audience_health = _compute_audience_health(data["quant_metrics"], data["sentiment_result"])
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
            "brand_description_used": data["brand_description_used"],
            "brand_description_auto_generated": data["brand_description_auto_generated"],
            "creator_persona": content_analysis.get("creator_persona"),
            "profanity_analysis": content_analysis.get("profanity_analysis"),
            "brand_partners": content_analysis.get("brand_partners"),
            "audience_health": audience_health,
            "content_analyzed": content_analyzed,
        },
    }

    if include_explanation:
        response["creator"]["ai_explanation"] = generate_risk_explanation(
            data["stats"]["title"], data["brand_description_used"], risk_result
        )

    return response


@app.post("/api/v1/creators/bulk")
def bulk_creator_scores(request: BulkAnalyzeRequest):
    """
    Analiza vise kreatora odjednom. Koristi entropijske tezine
    izracunate iz cijelog batch-a (objektivna alternativa AHP tezinama
    - vidi entropy_service.py), analogno CreatorScore bulk endpoint-u,
    uz razliku da je metodologija tezinjenja transparentna i
    prilagodjena samom uzorku, a ne fiksna.
    """
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

        data = _gather_and_score(handle, brand_description=creator.brand_description)
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


# ---------------------------------------------------------------------
# Stari endpoint - zadrzan radi kompatibilnosti sa postojecim frontend-om
# ---------------------------------------------------------------------

@app.post("/analyze")
def analyze_channel(request: AnalyzeRequest):
    """
    DEPRECATED - koristiti GET /api/v1/creators/youtube/{handle}.
    Zadrzano dok se frontend ne prebaci na novu strukturu.
    """
    data = _gather_and_score(request.channel_handle, brand_description=request.brand_description)

    final_result = calculate_final_risk_score(
        data["quant_metrics"], data["sentiment_result"], data["brand_fit_result"],
        subscriber_count=data["stats"]["subscriber_count"],
    )

    explanation = generate_risk_explanation(
        data["stats"]["title"], request.brand_description, final_result
    )

    content_analysis = analyze_creator_content(data["stats"], data["videos"], data["transcripts"])
    audience_health = _compute_audience_health(data["quant_metrics"], data["sentiment_result"])
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
        "creator_persona": content_analysis.get("creator_persona"),
        "profanity_analysis": content_analysis.get("profanity_analysis"),
        "brand_partners": content_analysis.get("brand_partners"),
        "audience_health": audience_health,
        "content_analyzed": content_analyzed,
    }