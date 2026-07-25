"""
Modul agregacije risk skora.

Metodoloske izmjene u odnosu na prethodnu verziju:
1. Tezine w1-w4 se vise NE hardkoduju direktno ovdje - uvoze se iz
   roc_service.py (Rank Order Centroid - Sidanski princip AHP ankete
   nije bio izvodljiv zbog nedostatka pristupa vecem broju eksperata,
   pa je ROC metod usvojen kao pojednostavljena, ali i dalje akademski
   utemeljena alternativa - zahtijeva samo rangiranje kriterijuma, ne
   puna parna poredjenja) ili, za bulk analizu vise kreatora, iz
   entropy_service.py (Hwang & Yoon, 1981).
2. Risk cap mehanizam je formalizovan kao IF-THEN mapiranje
   (frekvencija/ozbiljnost -> kategorija rizika), po uzoru na
   Markowski & Mannan (2008), citirano u Duijm (2015, Safety Science),
   umjesto proizvoljnih pragova bez obrazlozenja. Struktura pravila
   (non-compensatory/conjunctive) je akademski utemeljena - vidi
   ssrn5468566.pdf (Einhorn, 1970 - conjunctive model) i
   Banihabib et al. (2020) za poredjenje compensatory/
   non-compensatory MCDM pristupa.
3. Kombinacija ponderisanog zbira (compensatory) i risk cap-a
   (non-compensatory) je namjerna hibridna metodologija, ne
   improvizacija - vidi iste izvore.
4. Engagement benchmark (_engagement_benchmark_for_size) koristi
   TROSLOJNI pristup - vidi detaljnu napomenu iznad funkcije.
5. Subscriber-to-view ratio I sentiment risk cap-ovi koriste Z-SCORE
   pristup umjesto fiksnih pragova, po uzoru na Daranda et al. (2025).
   Subscriber-ratio Z-score je kalibrisan na nezavisnom dataset-u
   (n=878, Youtube_Influencer_Analysis_Updated.csv). Sentiment
   Z-score je i dalje kalibrisan na manjem, sopstvenom test-uzorku
   (n=4) - vidi napomene iznad obje funkcije.
6. Kategorije rizika (nizak/srednji/visok) koriste PERCENTIL-BAZIRANE
   granice izvedene iz referentnog uzorka testiranih kanala, umjesto
   pretpostavljenih vrijednosti - po uzoru na Daranda et al. (2025).
"""

import statistics

from .roc_service import get_module_weights
from .entropy_service import compute_entropy_weights, build_decision_matrix

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]


# ---------------------------------------------------------------------
# Z-SCORE PRISTUP za subscriber-to-view ratio anomaliju - kalibrisano
# na nezavisnom, vecem dataset-u (Youtube_Influencer_Analysis_
# Updated.csv - javno dostupan, rucno prikupljen istrazivacki dataset,
# 647 jedinstvenih kanala, raspon 34-305M pretplatnika). Nakon
# ciscenja outliera (uklonjeni redovi sa ER>100% kao vjerovatne
# greske u podacima, i gornji 1. percentil ratio vrijednosti), n=878
# validnih video-kanal parova.
#
# Mean=13.30, std=46.37 (desno-asimetricna distribucija - ocekivano
# za ovu vrstu omjera, koji prirodno ima dugi rep). Prag izracunat po
# istoj metodologiji kao Daranda et al. (2025) - mean + 2*std.
#
# NAPOMENA (transparentno navedeno): distribucija je desno-asimetricna
# (mean >> median), sto znaci da mean+2*std nije savrseno kalibrisan
# prag za ovakvu raspodjelu - robusnija alternativa (npr. medijan +
# MAD) bi bila metodoloski cistija, ali mean+2*std je zadrzan radi
# konzistentnosti sa Daranda et al. (2025) metodologijom.
# ---------------------------------------------------------------------

_REFERENCE_RATIO_MEAN = 13.3028
_REFERENCE_RATIO_STD = 46.3733
_Z_SCORE_THRESHOLD = 2.0  # Daranda et al. (2025) - 95% interval povjerenja


def _calculate_ratio_z_score(ratio: float) -> float:
    """
    Z-score za subscriber_to_view_ratio u odnosu na referentnu
    distribuciju (n=878, Youtube_Influencer_Analysis_Updated.csv).
    """
    if _REFERENCE_RATIO_STD == 0:
        return 0.0
    return (ratio - _REFERENCE_RATIO_MEAN) / _REFERENCE_RATIO_STD


# ---------------------------------------------------------------------
# Z-SCORE PRISTUP za sentiment anomaliju - ista metodologija kao
# subscriber-ratio Z-score (Daranda et al., 2025), obrnut smjer
# (nizak sentiment = rizik, ne visok).
#
# NAPOMENA (transparentno navedeno): ovo je STATICKI Z-score na
# jednom "presjeku" sentimenta, ne volatility-baziran pristup koji
# Alipour et al. (2025, preprint, ssrn7102482) pokazuju kao jaci
# prediktor reputacionog rizika (AUC 0.70-0.78 na stvarnim podacima).
# Volatility pristup zahtijeva pracenje sentimenta kroz vise
# vremenskih tacaka, sto trenutni pipeline (jedan uzorak komentara po
# analizi) ne podrzava - navedeno kao pravac buduceg istrazivanja.
#
# Referentni uzorak (n=4, testirani kanali - poglavlje 4.3): MKBHD,
# Simon Wilson, AN NA, NikkieTutorials. Mali uzorak (n<30) - navedeno
# kao ogranicenje, isti tip privremene procjene kao i ranija verzija
# subscriber-ratio Z-score prije pristupa vecem dataset-u.
# ---------------------------------------------------------------------

_REFERENCE_SENTIMENT_SCORES = [
    10,     # MKBHD
    24.5,   # Simon Wilson
    48.5,   # AN NA
    20.5,   # NikkieTutorials
]

_SENTIMENT_Z_THRESHOLD = -2.0  # Daranda et al. (2025), isti princip, obrnut smjer


def _calculate_sentiment_z_score(sentiment_score: float) -> float:
    """
    Z-score za sentiment_score u odnosu na referentnu distribuciju
    testiranih kanala, po istoj metodologiji kao
    _calculate_ratio_z_score.
    """
    if len(_REFERENCE_SENTIMENT_SCORES) < 2:
        return 0.0
    mean_s = statistics.mean(_REFERENCE_SENTIMENT_SCORES)
    std_s = statistics.stdev(_REFERENCE_SENTIMENT_SCORES)
    if std_s == 0:
        return 0.0
    return (sentiment_score - mean_s) / std_s


# Risk cap pravila kao formalno IF-THEN mapiranje:
# "AKO je metrika M u kategoriji kriticnog nalaza, ONDA je finalni
# skor ogranicen na najvise 'cap' vrijednost", nezavisno od
# ponderisanog zbira ostalih modula (conjunctive/non-compensatory
# logika, Einhorn 1970).
RISK_CAP_RULES = [
    {
        "name": "Statisticki anomalan negativan sentiment (Z-score < -2.0)",
        "condition": lambda m: _calculate_sentiment_z_score(m["sentiment_score"]) < _SENTIMENT_Z_THRESHOLD,
        "cap": 40,
    },
    {
        "name": "Statisticki anomalan odnos pretplatnici/pregledi (Z-score > 2.0)",
        "condition": lambda m: _calculate_ratio_z_score(m["subscriber_to_view_ratio"]) > _Z_SCORE_THRESHOLD,
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
# Engagement benchmark - TROSLOJNI pristup, jer nijedan pojedinacan
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
# channel-level izvor za ovaj segment. Umjesto lazne preciznosti,
# koristi se eksplicitno OBILJEZEN, konzervativan industrijski prag:
# sredina "dobrog" opsega (3-7%) po opste prihvacenoj industrijskoj
# klasifikaciji (YouTube Engagement Rate Calculator FAQ, komercijalni
# izvor - NIJE peer-reviewed). Pokusaj kalibracije na nezavisnom
# dataset-u (Youtube_Influencer_Analysis_Updated.csv, n=883) NIJE
# uspio - power-law regresija subs->ER dala je R^2=0.004 (statisticki
# beznacajno), cak i nakon stratifikacije po jeziku sadrzaja
# (R^2=0.002-0.019) - navedeno kao negativan, ali vrijedan nalaz u
# poglavlju 4.3/5, ne kao razlog za zamjenu postojeceg izvora.
#
# SLOJ 3 (>= 100 miliona pretplatnika, mega-kanali): REVIDIRANO nakon
# pronalaska Wies, Bleier & Edeling (2022), "Finding Goldilocks
# Influencers", Journal of Marketing 87(3), 383-405, koji na 802
# Instagram kampanje (1700+ influensera) utvrdjuju OBRNUTO U-OBLIKOVAN
# odnos izmedju broja pratilaca i engagement-a. Umjesto proizvoljnog
# fiksnog broja, koristi se KONTINUIRANA EKSTRAPOLACIJA vec fitovane
# krive (bez gornjeg klema), sa apsolutnim podom od 1% - smjer
# potkrijepljen Wies et al. (2022), magnituda NIJE empirijski
# potvrdjena za ovaj opseg.
# ---------------------------------------------------------------------

_LARGE_CHANNEL_CURVE_A = 2415.774517
_LARGE_CHANNEL_CURVE_B = -0.329418
_LARGE_CHANNEL_THRESHOLD = 15_000_000
_LARGE_CHANNEL_MIN_SUBS = 18_400_000   # donja granica stvarno mjerenog opsega (Tabela 1)
_LARGE_CHANNEL_MAX_SUBS = 35_900_000   # gornja granica stvarno mjerenog opsega (Tabela 1)

_SMALL_CHANNEL_FALLBACK_ER = 5.0  # sredina "dobrog" opsega 3-7%, industrijski izvor, NIJE akademski potvrdjeno

_MEGA_CHANNEL_THRESHOLD = 100_000_000
_MEGA_CHANNEL_MIN_BENCHMARK = 1.0  # apsolutni pod - sprjecava nerealisticno nisku vrijednost za astronomske brojeve pretplatnika


def _engagement_benchmark_for_size(subscriber_count: int) -> float:
    """
    Vraca ocekivan "odlican" engagement rate (%) za dati broj
    pretplatnika. TROSLOJNI pristup:

    SLOJ 1 - mega-kanali (>= 100M): kontinuirana ekstrapolacija
    fitovane krive (BEZ gornjeg klema), sa apsolutnim podom od 1%.
    Smjer opadanja potkrijepljen Wies et al. (2022, Journal of
    Marketing), magnituda NIJE empirijski potvrdjena za ovaj opseg -
    tretirati kao najbolju trenutno dostupnu procjenu, ne konacan broj.

    SLOJ 2 - veliki kanali (15M-100M): kriva klemovana na stvarno
    mjeren opseg (18.4M-35.9M, Lopez-Navarrete Tabela 1).

    SLOJ 3 - manji kanali (< 15M): industrijski fallback (5%).
    """
    if subscriber_count >= _MEGA_CHANNEL_THRESHOLD:
        extrapolated = _LARGE_CHANNEL_CURVE_A * (subscriber_count ** _LARGE_CHANNEL_CURVE_B)
        return round(max(_MEGA_CHANNEL_MIN_BENCHMARK, extrapolated), 3)
    elif subscriber_count >= _LARGE_CHANNEL_THRESHOLD:
        clamped = max(_LARGE_CHANNEL_MIN_SUBS, min(subscriber_count, _LARGE_CHANNEL_MAX_SUBS))
        return round(_LARGE_CHANNEL_CURVE_A * (clamped ** _LARGE_CHANNEL_CURVE_B), 3)
    else:
        return _SMALL_CHANNEL_FALLBACK_ER


def normalize_quantitative_score(quant_metrics: dict, subscriber_count: int = None) -> float:
    """
    Pretvara sirove kvantitativne metrike u jedinstven skor 0-100.

    Engagement komponenta se poredi sa troslojnim benchmarkom (vidi
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

    METODOLOSKO OGRANICENJE (transparentno, ne skriveno): mnozilac 3
    je HEURISTIKA, uskladjena sa istom logikom u
    main.py::_compute_audience_health (koja koristi isti odnos za
    "bot activity" prikaz). Ovaj konkretan broj NIJE empirijski izveden.

    Pravi akademski pristup za ovu vrstu proracuna - vidi Developing a
    Multimodal Approach to Channel Characterization on YouTube - ne
    pretpostavlja mnozilac, nego racuna Cohen's d effect size (Cohen,
    1988) za svaku relacionu metriku (ukljucujuci views-to-subscriber)
    na osnovu STVARNOG label-ovanog dataseta aktivnih i potvrdjeno
    suspendovanih kanala (u njihovom slucaju: 71 kanal, 7 suspendovanih),
    i koristi taj Cohen's d kao tezinu u ponderisanom zbiru
    (S = sum(d_i * x_i)). Takav pristup zahtijeva prikupljanje
    sopstvenog label-ovanog dataseta poznato-problematicnih kanala kroz
    vrijeme, sto nadilazi obim ovog rada. Trenutni mnozilac (3) treba
    tretirati kao pocetnu, nepotvrdjenu procjenu - eksplicitno navedeno
    kao ogranicenje i pravac buduceg istrazivanja u poglavlju 5.
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


# ---------------------------------------------------------------------
# PERCENTIL-BAZIRANE GRANICE za kategorije rizika - zamjenjuju ranije
# pretpostavljene granice (70/45) empirijski izvedenim vrijednostima
# iz sopstvenog testiranog uzorka, po uzoru na Daranda et al. (2025),
# koji definisu "severity" kategorije kao procentile iz sopstvene
# distribucije podataka (npr. "High = gornjih 5%").
#
# Ovdje se koristi tercil-podjela (donja/gornja granica na 33./67.
# percentilu) umjesto Daranda-inog asimetricnog razbijanja (5%/15%/
# 80%), jer je njihov pristup dizajniran za RIJETKE anomalije, dok
# ovaj model treba da podijeli CIJEL uzorak na tri priblizno jednaka
# dijela (nizak/srednji/visok rizik) - metodoloska adaptacija istog
# principa (percentil-bazirano umjesto pretpostavljeno), ne identicna
# replikacija.
#
# VAZNO: _REFERENCE_FINAL_SCORES MORA biti popunjen stvarnim finalnim
# skorovima testiranih kanala (POD TRENUTNOM verzijom koda - ROC
# tezine + nova brand-fit kalibracija) prije upotrebe. Dok uzorak nije
# dovoljan (n<5), koristi se fallback na ranije pretpostavljene
# vrijednosti (70/45).
# ---------------------------------------------------------------------

_REFERENCE_FINAL_SCORES = [
    52.21,  # MKBHD + NYX Professional Makeup
    75.85,  # MKBHD + Apple
    62.44,  # Simon Wilson + Skyscanner
    83.98,  # AN NA + Booking.com
    60.57,  # NikkieTutorials + Prada
]


def get_risk_category_thresholds() -> dict:
    """
    Racuna granice kategorija rizika (nizak/srednji/visok) kao 33. i
    67. percentil referentnog uzorka finalnih skorova. Ako uzorak nije
    popunjen, vraca prethodne pretpostavljene vrijednosti (70/45) kao
    fallback.
    """
    if len(_REFERENCE_FINAL_SCORES) < 5:
        return {"low_threshold": 70, "high_threshold": 45, "source": "pretpostavljeno (nedovoljno podataka)"}

    sorted_scores = sorted(_REFERENCE_FINAL_SCORES)
    percentiles = statistics.quantiles(sorted_scores, n=3)  # [33. percentil, 67. percentil]

    return {
        "low_threshold": round(percentiles[1], 2),   # 67. percentil - granica za "Nizak rizik"
        "high_threshold": round(percentiles[0], 2),  # 33. percentil - granica za "Visok rizik"
        "source": f"empirijski, n={len(_REFERENCE_FINAL_SCORES)}",
    }


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
) -> dict:
    """
    Objedinjuje sve module u finalni risk score.

    subscriber_count: broj pretplatnika kanala, koristi se da
        engagement rate bude ocijenjen u odnosu na benchmark
        prilagodjen velicini kanala.
    """
    if weights is None:
        roc_result = get_module_weights()
        weights = roc_result["weights"]
        weights_source = "roc"

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
        weights_source = "roc"

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

    print("=== FINALNI RISK SCORE (ROC tezine) ===")
    print(f"Kanal: {stats['title']}")
    print(f"Tezine koriscene: {final_result['weights_used']} (izvor: {final_result['weights_source']})")
    print(f"Skorovi po modulima: {final_result['module_scores']}")
    print(f"Ponderisani skor (prije cap-a): {final_result['weighted_score_before_cap']}")
    print(f"FINALNI SKOR: {final_result['final_score']}")
    print(f"Kategorija: {final_result['risk_category']}")
    print(f"Aktivirani risk cap-ovi: {final_result['triggered_caps']}")