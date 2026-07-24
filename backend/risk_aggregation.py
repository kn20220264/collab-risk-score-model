"""
Modul agregacije risk skora.

Metodoloske izmjene u odnosu na prethodnu verziju:
1. Tezine w1-w4 se vise NE hardkoduju direktno ovdje - uvoze se iz
   ahp_service.py (izracunate iz pairwise comparison matrice preko
   glavnog sopstvenog vektora, Saaty 1980/1987) ili, za bulk analizu
   vise kreatora, iz entropy_service.py (Hwang & Yoon, 1981).
2. Risk cap mehanizam je formalizovan kao IF-THEN mapiranje
   (frekvencija/ozbiljnost -> kategorija rizika), po uzoru na
   Markowski & Mannan (2008), citirano u Duijm (2015, Safety Science),
   umjesto proizvoljnih pragova bez obrazlozenja. Pragovi ostaju
   procjena autora (jer komercijalni alati poput CreatorScore/
   HypeAuditor ne objavljuju svoje), ali je sama STRUKTURA pravila
   (non-compensatory/conjunctive) akademski utemeljena - vidi
   ssrn5468566.pdf (Einhorn, 1970 - conjunctive model) i
   Banihabib et al. (2020) za poredjenje compensatory/
   non-compensatory MCDM pristupa.
3. Kombinacija ponderisanog zbira (compensatory) i risk cap-a
   (non-compensatory) je namjerna hibridna metodologija, ne
   improvizacija - vidi iste izvore.
4. Engagement benchmark (_engagement_benchmark_for_size) koristi
   DVOSLOJNI pristup - vidi detaljnu napomenu iznad funkcije. Ranija
   verzija (jednoslojna power-law regresija na Lopez-Navarrete Tabelu
   4) je napustena jer su ti podaci na nivou POJEDINACNOG VIDEA, ne
   kanala, sto je dovodilo do neopravdano strogog benchmarka za manje/
   srednje kanale (klemovanje na najvisu vrijednost iz seta). Nova
   verzija koristi stvarne CHANNEL-LEVEL podatke (Lopez-Navarrete
   Tabela 1, n=3) za velike kanale, i transparentno obiljezen
   industrijski fallback za manje kanale, gdje pouzdan akademski
   channel-level izvor trenutno ne postoji.
"""

from .ahp_service import get_module_weights
from .entropy_service import compute_entropy_weights, build_decision_matrix

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]

# Risk cap pravila kao formalno IF-THEN mapiranje:
# "AKO je metrika M u kategoriji kriticnog nalaza, ONDA je finalni
# skor ogranicen na najvise 'cap' vrijednost", nezavisno od
# ponderisanog zbira ostalih modula (conjunctive/non-compensatory
# logika, Einhorn 1970).
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
    # GRADUISAN brand-fit cap (dva praga) - struktura sa vise
    # stepenovanih kategorija ozbiljnosti je usaglasena sa formalnom
    # risk-matrix metodologijom (Markowski & Mannan, 2008, citirano u
    # Duijm, 2015, Safety Science). Pragovi < 10 i < 25 potvrdjeni
    # testom na NikkieTutorials + Caterpillar primjeru (brand_fit_score
    # = 13.77 i 4.04 u ponovljenim testovima), MasterChef Srbija +
    # Plazma (brand_fit_score = 49.39, legitiman umjeren fit) ostaje
    # sigurno iznad oba praga.
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


# ---------------------------------------------------------------------
# Engagement benchmark - DVOSLOJNI pristup, jer nijedan pojedinacan
# izvor ne pokriva ceo opseg velicina kanala pouzdano:
#
# SLOJ 1 (>= 15 miliona pretplatnika): power-law kriva fitovana na
# STVARNE CHANNEL-LEVEL podatke (Lopez-Navarrete et al., 2021, Tabela
# 1: ElRubiusOMG 35.9M/7.815%, Vegetta777 27.5M/8.603%, AuronPlay
# 18.4M/9.750%). Ovo je pravi nivo agregacije za nasu svrhu (subs ->
# prosjecan ER kanala), ali samo n=3 tacke, sve u opsegu 18-36M -
# koristi se samo unutar/blizu tog opsega (klemovano).
#   ER = 2415.774517 * subs^-0.329418
#   (regresija: log(ER) = log(a) + b*log(subs), numpy.polyfit)
#
# SLOJ 2 (< 15 miliona pretplatnika): NEMAMO pouzdan akademski
# channel-level izvor za ovaj segment (Lopez-Navarrete Tabela 4 je
# video-level, pogresan nivo agregacije za direktnu primjenu ovdje -
# klemovanje na tim podacima je ranije davalo neopravdano strog
# benchmark ~20% za sve kanale ispod ~600k pretplatnika). Umjesto
# lazne preciznosti, koristi se eksplicitno OBILJEZEN, konzervativan
# industrijski prag: sredina "dobrog" opsega (3-7%) po opste
# prihvacenoj industrijskoj klasifikaciji (YouTube Engagement Rate
# Calculator FAQ, komercijalni izvor - NIJE peer-reviewed, tretirati
# kao privremeni oslonac dok se ne nadje/sprovede pravi channel-level
# akademski izvor za ovaj segment velicina - vidi poglavlje 4.3,
# planirano dalje istrazivanje).
# ---------------------------------------------------------------------

_LARGE_CHANNEL_CURVE_A = 2415.774517
_LARGE_CHANNEL_CURVE_B = -0.329418
_LARGE_CHANNEL_THRESHOLD = 15_000_000
_LARGE_CHANNEL_MIN_SUBS = 18_400_000   # donja granica stvarno mjerenog opsega (Tabela 1)
_LARGE_CHANNEL_MAX_SUBS = 35_900_000   # gornja granica stvarno mjerenog opsega (Tabela 1)

_SMALL_CHANNEL_FALLBACK_ER = 5.0  # sredina "dobrog" opsega 3-7%, industrijski izvor, NIJE akademski potvrdjeno


def _engagement_benchmark_for_size(subscriber_count: int) -> float:
    """
    Vraca ocekivan "odlican" engagement rate (%) za dati broj
    pretplatnika.

    Za velike kanale (>= 15M), koristi krivu fitovanu na stvarne
    channel-level podatke (n=3, Lopez-Navarrete Tabela 1) - kriva se
    dodatno klemuje na stvarno mjeren opseg (18.4M-35.9M) da se
    izbjegne neopravdana ekstrapolacija van njega.

    Za manje kanale (< 15M), gdje nemamo pouzdan akademski channel-
    level izvor, koristi se transparentno obiljezen industrijski
    fallback (5%, sredina "dobrog" opsega 3-7%) - NIJE akademski
    potvrdjeno, treba tretirati kao privremeno rjesenje.
    """
    if subscriber_count >= _LARGE_CHANNEL_THRESHOLD:
        clamped = max(_LARGE_CHANNEL_MIN_SUBS, min(subscriber_count, _LARGE_CHANNEL_MAX_SUBS))
        return round(_LARGE_CHANNEL_CURVE_A * (clamped ** _LARGE_CHANNEL_CURVE_B), 3)
    else:
        return _SMALL_CHANNEL_FALLBACK_ER


def normalize_quantitative_score(quant_metrics: dict, subscriber_count: int = None) -> float:
    """
    Pretvara sirove kvantitativne metrike u jedinstven skor 0-100.

    Engagement komponenta se poredi sa dvoslojnim benchmarkom (vidi
    _engagement_benchmark_for_size) umjesto univerzalnog fiksnog
    praga ili proizvoljnih stepenastih kategorija.
    """
    if subscriber_count is not None:
        benchmark = _engagement_benchmark_for_size(subscriber_count)
    else:
        benchmark = 10.0  # fallback za kompatibilnost unazad

    engagement = min(quant_metrics["engagement_rate"] / benchmark * 100, 100)
    view_stability = max(0, 100 - quant_metrics["view_consistency_cv"])

    return round((engagement + view_stability) / 2, 2)


def normalize_authenticity_score(quant_metrics: dict) -> float:
    """
    Manji subscriber/view ratio = bolja autenticnost.
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
    Provjerava sva IF-THEN risk cap pravila. Ako se neko aktivira,
    finalni skor se ogranicava na min(trenutni_cap, novi_cap) - dakle
    najstroziji aktivirani cap odredjuje gornju granicu (conjunctive
    logika: jedan kriticni nalaz je dovoljan da obori skor, bez obzira
    na ostale module).
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


def calculate_final_risk_score(
    quant_metrics: dict,
    sentiment_result: dict,
    brand_fit_result: dict,
    subscriber_count: int = None,
    weights: dict = None,
    weights_source: str = "ahp",
) -> dict:
    """
    Objedinjuje sve module u finalni risk score.

    subscriber_count: broj pretplatnika kanala, koristi se da
        engagement rate bude ocijenjen u odnosu na benchmark
        prilagodjen velicini kanala.
    """
    if weights is None:
        ahp_result = get_module_weights()
        weights = ahp_result["weights"]
        weights_source = "ahp"

    module_scores = {
        "quantitative": normalize_quantitative_score(quant_metrics, subscriber_count),
        "authenticity": normalize_authenticity_score(quant_metrics),
        "sentiment": normalize_sentiment_score(sentiment_result),
        "brand_fit": brand_fit_result["brand_fit_score"],
    }

    weighted_sum = sum(module_scores[key] * weights[key] for key in MODULE_LABELS)

    cap_check_metrics = {
        "sentiment_score": sentiment_result["sentiment_score"],
        "subscriber_to_view_ratio": quant_metrics["subscriber_to_view_ratio"],
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
    }


def calculate_bulk_risk_scores(creators_raw_data: list) -> list:
    """
    Racuna risk score za vise kreatora odjednom, koristeci entropijske
    tezine izracunate IZ TOG KONKRETNOG BATCH-A.

    creators_raw_data: lista dict-ova, svaki sa kljucevima
        'quant_metrics', 'sentiment_result', 'brand_fit_result', 'stats'
        (stats mora sadrzati 'subscriber_count').
    """
    all_module_scores = []
    for creator in creators_raw_data:
        subscriber_count = creator.get("stats", {}).get("subscriber_count")
        scores = {
            "quantitative": normalize_quantitative_score(creator["quant_metrics"], subscriber_count),
            "authenticity": normalize_authenticity_score(creator["quant_metrics"]),
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
        weights_source = "ahp"

    results = []
    for creator, module_scores in zip(creators_raw_data, all_module_scores):
        weighted_sum = sum(module_scores[key] * weights[key] for key in MODULE_LABELS)

        cap_check_metrics = {
            "sentiment_score": creator["sentiment_result"]["sentiment_score"],
            "subscriber_to_view_ratio": creator["quant_metrics"]["subscriber_to_view_ratio"],
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
        })

    return results


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

    final_result = calculate_final_risk_score(
        quant_metrics, sentiment_result, brand_fit_result,
        subscriber_count=stats["subscriber_count"],
    )

    print("=== FINALNI RISK SCORE (AHP tezine) ===")
    print(f"Kanal: {stats['title']}")
    print(f"Tezine koriscene: {final_result['weights_used']} (izvor: {final_result['weights_source']})")
    print(f"Skorovi po modulima: {final_result['module_scores']}")
    print(f"Ponderisani skor (prije cap-a): {final_result['weighted_score_before_cap']}")
    print(f"FINALNI SKOR: {final_result['final_score']}")
    print(f"Kategorija: {final_result['risk_category']}")
    print(f"Aktivirani risk cap-ovi: {final_result['triggered_caps']}")