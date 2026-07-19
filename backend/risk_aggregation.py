# Tezine modula u finalnom skoru (moraju sabrati na 1.0)
WEIGHTS = {
    "quantitative": 0.35,   # engagement, konzistentnost, rast
    "authenticity": 0.20,   # subscriber/view ratio
    "sentiment": 0.25,      # sentiment komentara
    "brand_fit": 0.20,      # semanticko poklapanje
}

# Pragovi za risk cap - kriticni nalazi koji ogranicavaju finalni skor
RISK_CAP_RULES = [
    {
        "name": "Izrazito negativan sentiment",
        "condition": lambda m: m["sentiment_score"] < -30,
        "cap": 40,
    },
    {
        "name": "Sumnjiv odnos pretplatnici/pregledi",
        "condition": lambda m: m["subscriber_to_view_ratio"] > 50,
        "cap": 35,
    },
    {
        "name": "Vrlo slabo brand-fit poklapanje",
        "condition": lambda m: m["brand_fit_score"] < 10,
        "cap": 50,
    },
]


def normalize_quantitative_score(quant_metrics: dict) -> float:
    """
    Pretvara sirove kvantitativne metrike u jedinstven skor 0-100.
    Vise engagement-a = bolje. Manja varijacija pregleda = bolje.
    Umjeren, redovan posting = bolje.
    """
    engagement = min(quant_metrics["engagement_rate"] / 10 * 100, 100)  # 10% ER = max skor
    view_stability = max(0, 100 - quant_metrics["view_consistency_cv"])  # manji CV = bolji skor

    return round((engagement + view_stability) / 2, 2)


def normalize_authenticity_score(quant_metrics: dict) -> float:
    """
    Manji subscriber/view ratio = bolja autenticnost (aktivnija publika u odnosu na velicinu).
    Skaliramo tako da ratio 0-10 daje visok skor, a preko 30 nizak.
    """
    ratio = quant_metrics["subscriber_to_view_ratio"]
    score = max(0, 100 - ratio * 3)
    return round(min(score, 100), 2)


def normalize_sentiment_score(sentiment_result: dict) -> float:
    """
    Sentiment score je vec u opsegu -100 do 100, pretvaramo u 0-100.
    """
    raw = sentiment_result["sentiment_score"]
    return round((raw + 100) / 2, 2)


def apply_risk_caps(final_score: float, metrics: dict) -> dict:
    """
    Provjerava sve risk cap pravila i, ako se neko aktivira, ogranicava finalni skor.
    Vraca i listu aktiviranih upozorenja radi transparentnosti (za AI obrazlozenje kasnije).
    """
    triggered = []
    capped_score = final_score

    for rule in RISK_CAP_RULES:
        if rule["condition"](metrics):
            triggered.append(rule["name"])
            capped_score = min(capped_score, rule["cap"])

    return {
        "score_before_cap": final_score,
        "score_after_cap": round(capped_score, 2),
        "triggered_caps": triggered,
    }


def categorize_risk(score: float) -> str:
    if score >= 70:
        return "Nizak rizik"
    elif score >= 45:
        return "Srednji rizik"
    else:
        return "Visok rizik"


def calculate_final_risk_score(quant_metrics: dict, sentiment_result: dict, brand_fit_result: dict) -> dict:
    """
    Objedinjuje sve module u finalni risk score, sa primjenom risk cap mehanizma.
    """
    module_scores = {
        "quantitative": normalize_quantitative_score(quant_metrics),
        "authenticity": normalize_authenticity_score(quant_metrics),
        "sentiment": normalize_sentiment_score(sentiment_result),
        "brand_fit": brand_fit_result["brand_fit_score"],
    }

    weighted_sum = sum(module_scores[key] * WEIGHTS[key] for key in WEIGHTS)

    # Metrike potrebne za provjeru risk cap pravila
    cap_check_metrics = {
        "sentiment_score": sentiment_result["sentiment_score"],
        "subscriber_to_view_ratio": quant_metrics["subscriber_to_view_ratio"],
        "brand_fit_score": brand_fit_result["brand_fit_score"],
    }

    cap_result = apply_risk_caps(weighted_sum, cap_check_metrics)
    final_score = cap_result["score_after_cap"]

    return {
        "module_scores": module_scores,
        "weighted_score_before_cap": round(weighted_sum, 2),
        "final_score": final_score,
        "risk_category": categorize_risk(final_score),
        "triggered_caps": cap_result["triggered_caps"],
    }


# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats, get_comments_for_videos
    from scoring import calculate_quantitative_metrics
    from ai_service import analyze_comments_batch
    from brand_fit import calculate_brand_fit_score

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)
    comments = get_comments_for_videos(video_ids, max_per_video=10)

    quant_metrics = calculate_quantitative_metrics(stats, videos)
    sentiment_result = analyze_comments_batch(comments)

    brand_description = "Tech brand selling premium smartphones, laptops and consumer electronics accessories, focused on innovation and design quality."
    brand_fit_result = calculate_brand_fit_score(brand_description, stats, videos)

    final_result = calculate_final_risk_score(quant_metrics, sentiment_result, brand_fit_result)

    print("=== FINALNI RISK SCORE ===")
    print(f"Kanal: {stats['title']}")
    print(f"Skorovi po modulima: {final_result['module_scores']}")
    print(f"Ponderisani skor (prije cap-a): {final_result['weighted_score_before_cap']}")
    print(f"FINALNI SKOR: {final_result['final_score']}")
    print(f"Kategorija: {final_result['risk_category']}")
    print(f"Aktivirani risk cap-ovi: {final_result['triggered_caps']}")