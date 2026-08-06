"""Agregacija risk skora iz kvantitativnog, authenticity, sentiment i brand-fit modula."""

import json
import os
import statistics

import numpy as np

from .roc_service import get_module_weights
from .entropy_service import compute_entropy_weights, build_decision_matrix

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]


# Stratifikacija referentne distribucije po velicini kanala (subscriber tier),
# da se kanal poredi samo sa slicnim po velicini.
_SUBSCRIBER_TIERS = [
    (0, 250_000, "nano/mikro (<250K)"),
    (250_000, 1_000_000, "mikro/srednji (250K-1M)"),
    (1_000_000, 5_000_000, "makro (1M-5M)"),
    (5_000_000, 15_000_000, "vrlo veliki (5M-15M)"),
    (15_000_000, float("inf"), "velikan/mega-kanal (15M+)"),
]

_REFERENCE_DATA_PATH = os.path.join(os.path.dirname(__file__), "reference_ratios.json")


def _load_reference_data() -> list:
    if os.path.exists(_REFERENCE_DATA_PATH):
        with open(_REFERENCE_DATA_PATH, "r") as f:
            return json.load(f)
    return []


_REFERENCE_DATA = _load_reference_data()


def _get_tier_label(subscriber_count: int) -> str:
    for lo, hi, label in _SUBSCRIBER_TIERS:
        if lo <= subscriber_count < hi:
            return label
    return _SUBSCRIBER_TIERS[-1][2]


_TIER_RATIOS_CACHE = {
    (lo, hi): np.array(sorted([d["ratio"] for d in _REFERENCE_DATA if lo <= d["subs"] < hi]))
    for lo, hi, _ in _SUBSCRIBER_TIERS
}

_FALLBACK_TIER_LO = _SUBSCRIBER_TIERS[-1][0]


def _get_tier_ratios_cached(subscriber_count: int) -> np.ndarray:
    if subscriber_count is None:
        subscriber_count = _FALLBACK_TIER_LO

    for lo, hi, _ in _SUBSCRIBER_TIERS:
        if lo <= subscriber_count < hi:
            return _TIER_RATIOS_CACHE[(lo, hi)]

    last_lo, last_hi, _ = _SUBSCRIBER_TIERS[-1]
    return _TIER_RATIOS_CACHE[(last_lo, last_hi)]


_Z_SCORE_THRESHOLD = 2.0


def _calculate_ratio_z_score(ratio: float, subscriber_count: int) -> float:
    tier_ratios = _get_tier_ratios_cached(subscriber_count)
    if len(tier_ratios) < 2:
        return 0.0
    mean_ratio = float(np.mean(tier_ratios))
    std_ratio = float(np.std(tier_ratios, ddof=1))
    if std_ratio == 0:
        return 0.0
    return (ratio - mean_ratio) / std_ratio


def _percentile_score(ratio: float, reference_sorted: np.ndarray) -> float:
    """Percentil-baziran skor (InCites metod): % referentnih kanala sa gorim ratio-om."""
    n = len(reference_sorted)
    if n < 2:
        return 50.0

    if ratio <= reference_sorted[0]:
        pr_lo, pr_hi = 0.0, 100.0 / n
        cc_lo, cc_hi = reference_sorted[0], reference_sorted[1]
    elif ratio >= reference_sorted[-1]:
        pr_lo, pr_hi = 100.0 * (n - 2) / n, 100.0 * (n - 1) / n
        cc_lo, cc_hi = reference_sorted[-2], reference_sorted[-1]
    else:
        idx = int(np.searchsorted(reference_sorted, ratio))
        cc_lo, cc_hi = reference_sorted[idx - 1], reference_sorted[idx]
        pr_lo, pr_hi = 100.0 * (idx - 1) / n, 100.0 * idx / n

    if cc_hi == cc_lo:
        percentile = pr_lo
    else:
        percentile = pr_lo + (ratio - cc_lo) * (pr_hi - pr_lo) / (cc_hi - cc_lo)

    return max(0.0, min(100.0, percentile))


_SENTIMENT_REFERENCE_PATH = os.path.join(
    os.path.dirname(__file__), "reference_sentiment.json"
)


def _load_sentiment_reference() -> dict:
    """Generise scripts/calibrate_sentiment_reference.py. Prazan dict = cap se ne aktivira."""
    if not os.path.exists(_SENTIMENT_REFERENCE_PATH):
        return {}
    try:
        with open(_SENTIMENT_REFERENCE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


_SENTIMENT_REFERENCE = _load_sentiment_reference()

_SENTIMENT_Z_THRESHOLD = -2.0


def get_sentiment_reference_stats() -> dict:
    dist = _SENTIMENT_REFERENCE.get("distribucija", {})
    params = _SENTIMENT_REFERENCE.get("parametri_izvodjenja", {})
    normality = _SENTIMENT_REFERENCE.get("test_normalnosti", {})

    mean = dist.get("mean")
    std = dist.get("std")
    available = mean is not None and std is not None and std > 0

    return {
        "available": available,
        "mean": mean,
        "std": std,
        "median": dist.get("median"),
        "n_channels": params.get("kanala_u_referentnom_uzorku"),
        "n_comments": params.get("ukupno_komentara"),
        "min_comments_per_channel": params.get("min_comments_per_channel"),
        "cap_threshold": _SENTIMENT_REFERENCE.get("cap_threshold"),
        "z_score_threshold": _SENTIMENT_Z_THRESHOLD,
        "normality_test": normality.get("test"),
        "normality_p_value": normality.get("p_value"),
        "normality_rejected": normality.get("normalnost_odbacena"),
        "skewness": normality.get("skewness"),
        "dataset": _SENTIMENT_REFERENCE.get("dataset", {}),
        "sensitivity_analysis": _SENTIMENT_REFERENCE.get("analiza_osjetljivosti", []),
    }


def _calculate_sentiment_z_score(sentiment_score: float) -> float:
    ref = get_sentiment_reference_stats()
    if not ref["available"]:
        return 0.0
    return (sentiment_score - ref["mean"]) / ref["std"]


def _is_sentiment_anomalous(sentiment_score: float) -> bool:
    if not get_sentiment_reference_stats()["available"]:
        return False
    return _calculate_sentiment_z_score(sentiment_score) < _SENTIMENT_Z_THRESHOLD


# Risk cap pravila: IF-THEN mapiranje, non-compensatory u odnosu na ponderisani zbir.
# metrics dict mora sadrzati "subscriber_count" (za stratifikovan Z-score).
RISK_CAP_RULES = [
    {
        "name": "Statisticki anomalan negativan sentiment (Z-score < -2.0)",
        "condition": lambda m: _is_sentiment_anomalous(m["sentiment_score"]),
        "cap": 40,
    },
    {
        "name": "Statisticki anomalan odnos pretplatnici/pregledi (Z-score > 2.0, u odnosu na kanale slicne velicine)",
        "condition": lambda m: _calculate_ratio_z_score(
            m["subscriber_to_view_ratio"], m["subscriber_count"]
        ) > _Z_SCORE_THRESHOLD,
        "cap": 35,
    },
    {
        "name": "Ekstremno slabo brand-fit poklapanje",
        "condition": lambda m: m["brand_fit_score"] < 10,
        "cap": 30,
    },
    {
        "name": "Vrlo slabo brand-fit poklapanje",
        "condition": lambda m: m["brand_fit_score"] < 25,
        "cap": 50,
    },
]


_LARGE_CHANNEL_CURVE_A = 2415.774517
_LARGE_CHANNEL_CURVE_B = -0.329418
_LARGE_CHANNEL_THRESHOLD = 15_000_000
_LARGE_CHANNEL_MIN_SUBS = 18_400_000
_LARGE_CHANNEL_MAX_SUBS = 35_900_000

_MEGA_CHANNEL_THRESHOLD = 100_000_000
_MEGA_CHANNEL_MIN_BENCHMARK = 1.0

# Referentne vrijednosti intenziteta interakcije po tematskoj kategoriji.
# Izvor: Rieder, Coromina i Matamoros-Fernandez (2020), Tabela 10.
_TABLE10_INTENSITY = {
    "Gaming": 4.7,
    "Lifestyle": 3.9,
    "Society": 3.8,
    "Knowledge": 3.7,
    "Entertainment": 3.5,
    "Sports": 3.2,
    "Music": 2.9,
    "none": 3.1,
}

_CALIBRATION_BASELINE_CATEGORY = "Gaming"

# Mapiranje creator_persona.primary_niche na kategorije iz Tabele 10.
_NICHE_TO_TABLE10_CATEGORY = {
    "tech": "Knowledge",
    "consumer electronics": "Knowledge",
    "education": "Knowledge",
    "science": "Knowledge",
    "finance": "Knowledge",
    "gaming": "Gaming",
    "comedy": "Entertainment",
    "entertainment": "Entertainment",
    "music": "Music",
    "sports": "Sports",
    "fitness": "Sports",
    "news": "Society",
    "politics": "Society",
    "lifestyle": "Lifestyle",
    "beauty": "Lifestyle",
    "fashion": "Lifestyle",
    "food": "Lifestyle",
    "travel": "Lifestyle",
}


def _map_niche_to_table10_category(primary_niche: str) -> str:
    if not primary_niche:
        return "none"
    return _NICHE_TO_TABLE10_CATEGORY.get(primary_niche.strip().lower(), "none")


_TABLE10_NEUTRAL_INTENSITY = round(
    sum(_TABLE10_INTENSITY.values()) / len(_TABLE10_INTENSITY), 3
)


def _genre_adjustment_factor(primary_niche: str) -> float:
    """Mnozilac za engagement benchmark u odnosu na Gaming kao baznu kategoriju."""
    category = _map_niche_to_table10_category(primary_niche)
    baseline = _TABLE10_INTENSITY[_CALIBRATION_BASELINE_CATEGORY]
    return _TABLE10_INTENSITY.get(category, baseline) / baseline


def _absolute_genre_benchmark(primary_niche: str) -> float:
    """Apsolutna ocekivana vrijednost engagement-a (%) za zanr, iz Tabele 10."""
    if not primary_niche:
        return _TABLE10_NEUTRAL_INTENSITY
    category = _map_niche_to_table10_category(primary_niche)
    if category == "none":
        return _TABLE10_NEUTRAL_INTENSITY
    return _TABLE10_INTENSITY[category]


_DEFAULT_INTENSITY = 3.6
_TIER2_ANCHOR = 15_000_000
_TIER2_CEILING = 35_900_000
_TIER2_DECAY_EXPONENT = -0.329


def _engagement_benchmark_for_size(subscriber_count: int, primary_niche: str = None) -> float:
    """Ocekivani engagement rate prilagodjen velicini kanala i zanru.

    Ispod 15M: konstanta po kategoriji (Rieder et al.). Od 15M navise: ista
    vrijednost umanjena po zakonu stepena, bez skoka na granici.
    """
    base = _TABLE10_INTENSITY.get(primary_niche, _DEFAULT_INTENSITY)

    if subscriber_count is None or subscriber_count < _TIER2_ANCHOR:
        return round(base, 3)

    effective = min(subscriber_count, _TIER2_CEILING)
    decay = (effective / _TIER2_ANCHOR) ** _TIER2_DECAY_EXPONENT

    return round(base * decay, 3)


def normalize_quantitative_score(
    quant_metrics: dict, subscriber_count: int = None, primary_niche: str = None
) -> float:
    if subscriber_count is not None:
        benchmark = _engagement_benchmark_for_size(subscriber_count, primary_niche)
    else:
        benchmark = 10.0  # fallback za kompatibilnost unazad

    engagement = min(quant_metrics["engagement_rate"] / benchmark * 100, 100)
    view_stability = max(0, 100 - quant_metrics["view_consistency_cv"])

    return round((engagement + view_stability) / 2, 2)


def normalize_authenticity_score(quant_metrics: dict, subscriber_count: int = None) -> float:
    """Percentil-baziran skor autenticnosti, stratifikovan po velicini kanala."""
    ratio = quant_metrics["subscriber_to_view_ratio"]

    if subscriber_count is None:
        subscriber_count = _FALLBACK_TIER_LO

    tier_ratios = _get_tier_ratios_cached(subscriber_count)
    percentile = _percentile_score(ratio, tier_ratios)

    # Jednosmjerno: ratio na/ispod medijane = pun skor, nizak ratio nije rizik.
    if percentile <= 50:
        return 100.0
    return round(100 - (percentile - 50) * 2, 2)


def normalize_sentiment_score(sentiment_result: dict) -> float:
    raw = sentiment_result["sentiment_score"]
    return round((raw + 100) / 2, 2)


def apply_risk_caps(final_score: float, metrics: dict) -> dict:
    """Najstroziji aktivirani cap odredjuje gornju granicu finalnog skora."""
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


def build_sentiment_diagnostics(sentiment_result: dict) -> dict:
    ref = get_sentiment_reference_stats()
    raw = sentiment_result["sentiment_score"]

    return {
        "raw_net_sentiment": raw,
        "positive_pct": sentiment_result.get("positive_pct"),
        "neutral_pct": sentiment_result.get("neutral_pct"),
        "negative_pct": sentiment_result.get("negative_pct"),
        "z_score": round(_calculate_sentiment_z_score(raw), 3),
        "reference_mean": ref["mean"],
        "reference_std": ref["std"],
        "reference_n_channels": ref["n_channels"],
        "cap_threshold": ref["cap_threshold"],
        "threshold_basis": (
            "z_score_external_reference" if ref["available"] else "unavailable_cap_disabled"
        ),
    }


_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "reference_risk_thresholds.json"
)

# Rezerva ako kalibracioni fajl nedostaje - nije za produkcijsku upotrebu.
_FALLBACK_THRESHOLDS = {"low_threshold": 83.48, "high_threshold": 58.97}

_thresholds_cache = None


def _load_risk_thresholds() -> dict:
    global _thresholds_cache
    if _thresholds_cache is not None:
        return _thresholds_cache

    try:
        with open(_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        granice = data["granice"]
        stvarni = data["stvarni_parovi"]
        ukrsteni = data["ukrsteni_parovi"]

        _thresholds_cache = {
            "low_threshold": granice["low_threshold"],
            "high_threshold": granice["high_threshold"],
            "source": (
                f"empirijski, kontrolna grupa "
                f"(n={stvarni['n']} stvarnih / {ukrsteni['n']} ukrstenih)"
            ),
            "n_stvarnih": stvarni["n"],
            "n_ukrstenih": ukrsteni["n"],
            "cliffs_delta": data.get("validacija", {}).get("cliffs_delta"),
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        _thresholds_cache = {
            **_FALLBACK_THRESHOLDS,
            "source": "rezerva n=5 (kalibracioni fajl nedostupan)",
        }

    return _thresholds_cache


def get_risk_category_thresholds() -> dict:
    return _load_risk_thresholds()


def categorize_risk(score: float) -> str:
    thresholds = get_risk_category_thresholds()
    if score >= thresholds["low_threshold"]:
        return "Nizak rizik"
    elif score >= thresholds["high_threshold"]:
        return "Srednji rizik"
    else:
        return "Visok rizik"


def calculate_final_risk_score(
    quant_metrics: dict,
    sentiment_result: dict,
    brand_fit_result: dict,
    subscriber_count: int = None,
    weights: dict = None,
    weights_source: str = "roc",
    primary_niche: str = None,
) -> dict:
    """Objedinjuje sve module u finalni risk score."""
    if weights is None:
        roc_result = get_module_weights()
        weights = roc_result["weights"]
        weights_source = "roc"

    module_scores = {
        "quantitative": normalize_quantitative_score(quant_metrics, subscriber_count, primary_niche),
        "authenticity": normalize_authenticity_score(quant_metrics, subscriber_count),
        "sentiment": normalize_sentiment_score(sentiment_result),
        "brand_fit": brand_fit_result["brand_fit_score"],
    }

    weighted_sum = sum(module_scores[key] * weights[key] for key in MODULE_LABELS)

    cap_check_metrics = {
        "sentiment_score": sentiment_result["sentiment_score"],
        "subscriber_to_view_ratio": quant_metrics["subscriber_to_view_ratio"],
        "subscriber_count": subscriber_count,
        "brand_fit_score": brand_fit_result["brand_fit_score"],
    }

    cap_result = apply_risk_caps(weighted_sum, cap_check_metrics)
    final_score = cap_result["score_after_cap"]

    return {
        "module_scores": module_scores,
        "weights_used": weights,
        "weights_source": weights_source,
        "weighted_score_before_cap": round(weighted_sum, 2),
        "final_score": final_score,
        "risk_category": categorize_risk(final_score),
        "triggered_caps": cap_result["triggered_caps"],
        "subscriber_tier": _get_tier_label(subscriber_count) if subscriber_count is not None else None,
        "sentiment_diagnostics": build_sentiment_diagnostics(sentiment_result),
    }


def calculate_bulk_risk_scores(creators_raw_data: list) -> list:
    """Risk score za vise kreatora odjednom, sa entropijskim tezinama iz tog batch-a."""
    all_module_scores = []
    for creator in creators_raw_data:
        subscriber_count = creator.get("stats", {}).get("subscriber_count")
        primary_niche = creator.get("creator_persona", {}).get("primary_niche")
        scores = {
            "quantitative": normalize_quantitative_score(creator["quant_metrics"], subscriber_count, primary_niche),
            "authenticity": normalize_authenticity_score(creator["quant_metrics"], subscriber_count),
            "sentiment": normalize_sentiment_score(creator["sentiment_result"]),
            "brand_fit": creator["brand_fit_result"]["brand_fit_score"],
        }
        all_module_scores.append(scores)

    if len(all_module_scores) >= 2:
        decision_matrix = build_decision_matrix(all_module_scores, MODULE_LABELS)
        entropy_result = compute_entropy_weights(decision_matrix, MODULE_LABELS)
        weights = entropy_result["weights"]
        weights_source = "entropy"
    else:
        weights = get_module_weights()["weights"]
        weights_source = "roc"

    results = []
    for creator, module_scores in zip(creators_raw_data, all_module_scores):
        subscriber_count = creator.get("stats", {}).get("subscriber_count")
        weighted_sum = sum(module_scores[key] * weights[key] for key in MODULE_LABELS)

        cap_check_metrics = {
            "sentiment_score": creator["sentiment_result"]["sentiment_score"],
            "subscriber_to_view_ratio": creator["quant_metrics"]["subscriber_to_view_ratio"],
            "subscriber_count": subscriber_count,
            "brand_fit_score": creator["brand_fit_result"]["brand_fit_score"],
        }
        cap_result = apply_risk_caps(weighted_sum, cap_check_metrics)
        final_score = cap_result["score_after_cap"]

        results.append({
            "module_scores": module_scores,
            "weights_used": weights,
            "weights_source": weights_source,
            "weighted_score_before_cap": round(weighted_sum, 2),
            "final_score": final_score,
            "risk_category": categorize_risk(final_score),
            "triggered_caps": cap_result["triggered_caps"],
            "subscriber_tier": _get_tier_label(subscriber_count) if subscriber_count is not None else None,
            "sentiment_diagnostics": build_sentiment_diagnostics(creator["sentiment_result"]),
        })

    return results


if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats, get_comments_for_videos
    from scoring import calculate_quantitative_metrics
    from ai_service import analyze_comments_batch
    from brand_fit import calculate_brand_fit_score

    ref = get_sentiment_reference_stats()
    print("=== REFERENTNA DISTRIBUCIJA SENTIMENTA ===")
    if ref["available"]:
        print(f"Kanala u uzorku : {ref['n_channels']} (iz {ref['n_comments']} komentara)")
        print(f"mean / std      : {ref['mean']} / {ref['std']}")
        print(f"cap prag        : {ref['cap_threshold']}")
        print(f"Normalnost      : {ref['normality_test']}, p={ref['normality_p_value']}, "
              f"odbacena={ref['normality_rejected']}, skewness={ref['skewness']}")
    else:
        print("NEDOSTUPNA - sentiment cap se nece aktivirati.")
        print(f"Ocekivana putanja: {_SENTIMENT_REFERENCE_PATH}")
        print("Pokreni: python scripts/calibrate_sentiment_reference.py <dataset.csv>")
    print()

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)
    comments = get_comments_for_videos(video_ids, max_per_video=10)

    quant_metrics = calculate_quantitative_metrics(stats, videos)
    sentiment_result = analyze_comments_batch(comments)

    brand_description = "Tech brand selling premium smartphones, laptops and consumer electronics accessories, focused on innovation and design quality."
    brand_fit_result = calculate_brand_fit_score(brand_description, stats, videos)

    final_result = calculate_final_risk_score(
        quant_metrics, sentiment_result, brand_fit_result,
        subscriber_count=stats["subscriber_count"],
    )

    print("=== FINALNI RISK SCORE (ROC tezine) ===")
    print(f"Kanal: {stats['title']}")
    print(f"Sloj velicine: {final_result['subscriber_tier']}")
    print(f"Tezine koriscene: {final_result['weights_used']} (izvor: {final_result['weights_source']})")
    print(f"Skorovi po modulima: {final_result['module_scores']}")
    print(f"Ponderisani skor (prije cap-a): {final_result['weighted_score_before_cap']}")
    print(f"FINALNI SKOR: {final_result['final_score']}")
    print(f"Kategorija: {final_result['risk_category']}")
    print(f"Aktivirani risk cap-ovi: {final_result['triggered_caps']}")
    print(f"Sentiment dijagnostika: {final_result['sentiment_diagnostics']}")

