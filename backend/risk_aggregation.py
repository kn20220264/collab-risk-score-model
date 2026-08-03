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
5. Subscriber-to-view ratio koristi PERCENTIL-BAZIRAN kontinuirani
   skor (InCites metod, Bornmann & Williams, 2020) UMJESTO ranijeg
   heuristickog mnozioca (ratio*3), plus Z-SCORE risk cap po uzoru
   na Daranda et al. (2026). Referentna distribucija je
   STRATIFIKOVANA po velicini kanala (subscriber tier) - vidi detaljnu
   napomenu iznad _SUBSCRIBER_TIERS. Ovo rjesava otkriveni problem
   testiranjem (poglavlje 4.3): jedinstvena, nestratifikovana
   distribucija sistematski je kaznjavala velike, legitimne kanale
   (npr. MKBHD), jer prirodno imaju visi subscriber-to-view ratio bez
   veze sa autenticnoscu.
6. Kategorije rizika (nizak/srednji/visok) koriste PERCENTIL-BAZIRANE
   granice izvedene iz referentnog uzorka testiranih kanala, umjesto
   pretpostavljenih vrijednosti - po uzoru na Daranda et al. (2026).
7. NOVO: sentiment risk cap koristi Z-score u odnosu na SPOLJASNJU
   referentnu distribuciju od 57 stvarnih YouTube kanala (45.005
   komentara), umjesto ranijeg uzorka od n=4 sopstvenih testova. Time
   sentiment cap prelazi iz statusa "privremena procjena" u status
   "izvedeno dokumentovanim postupkom iz spoljasnjih podataka" - isti
   nivo utemeljenja koji vec ima subscriber-ratio cap. Vidi detaljnu
   napomenu iznad _load_sentiment_reference.
"""

import json
import os
import statistics

import numpy as np

from .roc_service import get_module_weights
from .entropy_service import compute_entropy_weights, build_decision_matrix

MODULE_LABELS = ["quantitative", "authenticity", "sentiment", "brand_fit"]


# ---------------------------------------------------------------------
# STRATIFIKACIJA REFERENTNE DISTRIBUCIJE PO VELICINI KANALA
#
# PROBLEM KOJI OVO RJESAVA (otkriveno testiranjem MKBHD + Samsung,
# poglavlje 4.3): jedinstvena (nestratifikovana) percentil distribucija
# sistematski je davala NIZAK skor autenticnosti velikim, ocigledno
# legitimnim kanalima (npr. MKBHD, 21M+ pretplatnika, ratio=7.08 ->
# authenticity_score=20.79 pod starim nestratifikovanim pristupom).
# Uzrok: referentni dataset (n=874) dominiraju MANJI kreatori (medijana
# ratio-a svega 1.17), pa veliki kanali prirodno ispadaju "ekstremni" u
# odnosu na cijeli skup - iako visi ratio kod velikih kanala odrazava
# normalnu dinamiku (ne gleda svaki pretplatnik svaki video), ne
# vjestacki naduvanu publiku.
#
# RJESENJE: kanal se poredi SAMO sa referentnim kanalima SLICNE
# velicine (isti "sloj"/tier), ne sa cijelim skupom.
#
# GRANICE SLOJEVA - kombinuju dva nezavisna izvora:
#   - 15M i 100M: ISTE granice kao u kvantitativnom modulu
#     (_engagement_benchmark_for_size), radi konzistentnosti strukture
#     modela kroz razlicite module (izvedene iz dostupnosti akademskih
#     channel-level podataka - Lopez-Navarrete Tabela 1 - i granice
#     pouzdane ekstrapolacije - Wies et al., 2023).
#   - 250K, 1M, 5M: NOVE granice, uvedene na osnovu nalaza Rajendran,
#     Creusy & Garnes (2024), koji na uzorku od 250 YouTube kreatora
#     pokazuju da se ponasanje kanala statisticki znacajno mijenja tek
#     iznad 5 miliona pretplatnika (Wilcoxon signed-rank test,
#     p<0.05), dok razlike izmedju manjih kategorija velicine (<250K
#     do 1M-5M) nisu statisticki znacajne
#     [IZVOR: Rajendran, P. T., Creusy, K. & Garnes, V. (2024),
#     "Shorts on the Rise: Assessing the Effects of YouTube Shorts on
#     Long-Form Video Content", arXiv:2402.18208 - fajl:
#     2402_18208v2.pdf].
#
# DRUGA VAZNA ISPRAVKA (poglavlje 4.3 - "recency bias" problem):
# referentni dataset je PONOVO IZGRADJEN nakon otkrivenog problema da
# je stari ratio (subs / prosjek POSLEDNJIH N videa) sistematski
# neuporediv sa referentnim CSV-om, ciji videi imaju medijanu starosti
# ~1600 dana (4.4 godine) - "mladi" video jos nije akumulirao preglede
# koje ce na kraju imati, sto je vjestacki naduvavalo ratio SVAKOG
# kanala analiziranog uzivo, nezavisno od velicine. Vidi detaljno
# obrazlozenje u scoring.py::calculate_subscriber_to_view_ratio.
#
# REKONSTRUKCIJA: CSV je na nivou VIDEA (ne kanala) - isti kanal se
# ponavlja u vise redova sa identicnim "Total Chanel Views"/"No of
# Videos the Channel" vrijednostima. Podaci su DEDUPLICIRANI po
# kanalu (Channel URL) PRIJE racunanja ratio-a, cime se dobija n=647
# jedinstvenih kanala (isti broj pominjan u ranijoj analizi
# kvantitativnog modula - vidi 4.1.2). Ratio je zatim izracunat kao
# subs / (Total Chanel Views / No of Videos the Channel) - CJELOZIVOTNI
# prosjek, direktno uporediv sa cjelozivotnim prosjekom koji sada
# racuna i nas sopstveni pipeline (scoring.py).
#
# PROVJERA VELICINE UZORKA PO SLOJU (n=625 nakon ciscenja outliera):
# <250K: 155, 250K-1M: 120, 1M-5M: 177, 5M-15M: 90, 15M+: 83 (sloj
# >100M spojen sa 15M-100M zbog premalog uzorka - n=4 - kad je
# posmatran zasebno; kvantitativni modul i dalje koristi zasebnu
# granicu na 100M, iz DRUGOG razloga - vidi napomenu kod
# _engagement_benchmark_for_size). Medijane ratio-a po sloju sada
# rastu blago i monotono sa velicinom (1.22 / 2.65 / 4.98 / 5.60 /
# 4.67), sto je realno, ocekivano ponasanje - za razliku od ranije
# verzije (medijana 1.17 za CIJEL skup, bez obzira na velicinu).
# ---------------------------------------------------------------------

_SUBSCRIBER_TIERS = [
    (0, 250_000, "nano/mikro (<250K)"),
    (250_000, 1_000_000, "mikro/srednji (250K-1M)"),
    (1_000_000, 5_000_000, "makro (1M-5M)"),
    (5_000_000, 15_000_000, "vrlo veliki (5M-15M)"),
    (15_000_000, float("inf"), "velikan/mega-kanal (15M+)"),
]

_REFERENCE_DATA_PATH = os.path.join(os.path.dirname(__file__), "reference_ratios.json")


def _load_reference_data() -> list:
    """
    Ucitava referentne podatke (subs + ratio parovi) sa diska.

    Format fajla: JSON lista dict-ova {"subs": int, "ratio": float},
    REPRODUKOVANA direktno iz Youtube_Influencer_Analysis_-_Updated.csv
    (javno dostupan istrazivacki dataset) primjenom sljedeceg postupka
    ciscenja:
      1. Uklonjeni redovi sa nedostajucim subs/views/likes/comments
      2. Uklonjeni redovi sa views<=100 ili subs<=100 (vjerovatne greske)
      3. Izracunat engagement_rate = (likes+comments)/views*100;
         uklonjeni redovi sa ER>100% (vjerovatne greske u podacima) ili
         ER<=0
      4. Izracunat subscriber_to_view_ratio = subs/views
      5. Uklonjen gornji 1% najekstremnijih ratio vrijednosti (outlieri)

    NAPOMENA O REPRODUKCIJI (transparentno navedeno): ponovnim
    izvodjenjem ovog postupka nad izvornim CSV-om dobijeno je n=874,
    globalni mean=13.3622, std=46.4710 - blago drugacije od ranije
    citiranih vrijednosti (n=878, mean=13.3028, std=46.3733) iz
    prethodne verzije ovog rada. Razlika je zanemarljiva (<1% u
    mean/std, 4 reda razlike u n) i pripisuje se sitnoj varijaciji u
    redoslijedu operacija izmedju sesija razvoja; OVDJE navedene,
    upravo reprodukovane vrijednosti koriste se kao autoritativne, jer
    su direktno verifikovane nad izvornim podacima.

    Ako fajl ne postoji (npr. tek klonirana instalacija bez ovog data
    fajla), vraca praznu listu - svi mehanizmi koji zavise od ovih
    podataka (percentil skor, Z-score cap) imaju bezbjedan fallback za
    taj slucaj.
    """
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


# Precomputed sortiran niz PO SLOJU, da se ne racuna iznova pri svakom
# pozivu (performansa) - kljuc je (lo, hi) tuple iz _SUBSCRIBER_TIERS.
_TIER_RATIOS_CACHE = {
    (lo, hi): np.array(sorted([d["ratio"] for d in _REFERENCE_DATA if lo <= d["subs"] < hi]))
    for lo, hi, _ in _SUBSCRIBER_TIERS
}

# Fallback sloj kada subscriber_count nije poznat.
#
# ISPRAVKA (u odnosu na prethodnu verziju): ranije se koristio
# _SUBSCRIBER_TIERS[-2][0] (= 5.000.000), sto je donja granica sloja
# "vrlo veliki (5M-15M)", iako je prateci komentar tvrdio da se kanal
# tretira kao "velikan" sloj. Sada eksplicitno pokazuje na posljednji
# sloj (15M+), sto odgovara deklarisanoj namjeri: bez poznate
# velicine, kanal se poredi sa slojem u kojem su ratio vrijednosti
# prirodno najvise, pa je procjena konzervativna (ne kaznjava
# nepotrebno).
_FALLBACK_TIER_LO = _SUBSCRIBER_TIERS[-1][0]


def _get_tier_ratios_cached(subscriber_count: int) -> np.ndarray:
    """
    Vraca sortirane referentne ratio vrijednosti za sloj kojem kanal
    pripada.

    ISPRAVKA: dodata zastita od subscriber_count=None. Prethodna
    verzija je u tom slucaju bacala TypeError ("'<=' not supported
    between instances of 'int' and 'NoneType'"), jer se None mogao
    proslijediti kroz cap_check_metrics kad se
    calculate_final_risk_score pozove bez subscriber_count - sto
    potpis funkcije izricito dozvoljava. Sada se koristi fallback sloj.
    """
    if subscriber_count is None:
        subscriber_count = _FALLBACK_TIER_LO

    for lo, hi, _ in _SUBSCRIBER_TIERS:
        if lo <= subscriber_count < hi:
            return _TIER_RATIOS_CACHE[(lo, hi)]

    last_lo, last_hi, _ = _SUBSCRIBER_TIERS[-1]
    return _TIER_RATIOS_CACHE[(last_lo, last_hi)]


_Z_SCORE_THRESHOLD = 2.0  # Daranda, Kankeviciene & Daranda (2026) - 95% interval povjerenja


def _calculate_ratio_z_score(ratio: float, subscriber_count: int) -> float:
    """
    Z-score za subscriber_to_view_ratio u odnosu na referentnu
    distribuciju ISTOG SLOJA velicine kanala (ne cijelog n=874 skupa -
    vidi napomenu o stratifikaciji iznad).
    """
    tier_ratios = _get_tier_ratios_cached(subscriber_count)
    if len(tier_ratios) < 2:
        return 0.0
    mean_ratio = float(np.mean(tier_ratios))
    std_ratio = float(np.std(tier_ratios, ddof=1))
    if std_ratio == 0:
        return 0.0
    return (ratio - mean_ratio) / std_ratio


def _percentile_score(ratio: float, reference_sorted: np.ndarray) -> float:
    """
    Percentil-baziran skor autenticnosti (InCites metod, Bornmann &
    Williams, 2020, poglavlje "Cumulative frequencies... optimized
    percentile approach") - zamjenjuje raniji heuristicki mnozilac
    (ratio*3). Vraca procenat referentnih kanala sa GORIM (visim)
    ratio-om od posmatranog (visi rezultat = bolja autenticnost).

    Za ratio unutar opsega referentnog skupa: direktan empirijski
    percentilni rang. Za ratio izvan opsega: linearna interpolacija
    izmedju dvije najblize poznate tacke (Bornmann & Williams, 2020,
    jednacine 10-11), umjesto proizvoljnog "clamp-a" na 0/100.

    NAPOMENA: reference_sorted je NIZ UNUTAR ISTOG SLOJA velicine
    kanala (vidi _get_tier_ratios_cached), ne cijeli n=874 skup.
    """
    n = len(reference_sorted)
    if n < 2:
        return 50.0  # nema referentnih podataka za ovaj sloj - neutralan fallback

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


# ---------------------------------------------------------------------
# Z-SCORE PRISTUP za sentiment anomaliju - ista metodologija kao
# subscriber-ratio Z-score (Daranda et al., 2026), obrnut smjer
# (nizak sentiment = rizik, ne visok).
#
# IZMJENA U ODNOSU NA PRETHODNU VERZIJU: referentna distribucija vise
# NIJE sopstveni uzorak od n=4 testirana kanala (MKBHD, Simon Wilson,
# AN NA, NikkieTutorials), nego SPOLJASNJI dataset od 45.005 YouTube
# komentara sa 66 kanala. Nakon primjene praga od najmanje 50
# komentara po kanalu (radi stabilnosti procjene procenata), u
# referentnom uzorku ostaje n=57 kanala:
#
#   mean = 31.3211
#   std  = 17.4948
#   prag = mean - 2.0 * std = -3.6684
#
# Time sentiment cap prelazi preko standardnog minimuma n>=30 za
# pouzdanu procjenu mu i sigma, i vise ne predstavlja "privremenu
# procjenu" - isti status utemeljenja koji vec ima subscriber-ratio
# cap.
#
# PROVJERA PRETPOSTAVKE NORMALNOSTI: Z-score kao metod pretpostavlja
# priblizno normalnu distribuciju referentne populacije. Ova
# pretpostavka je EKSPLICITNO testirana, ne preutkana:
# Shapiro-Wilk p=0.1484 (normalnost se NE odbacuje), skewness=0.0153
# (prakticno simetricna distribucija). Odgovara Koraku 3
# (Multivariate analysis) iz OECD/JRC prirucnika [Nardo, M., Saisana,
# M., Saltelli, A., Tarantola, S., Hoffmann, A. & Giovannini, E.
# (2005), "Handbook on Constructing Composite Indicators", OECD
# Statistics Working Papers 2005/03, STD/DOC(2005)3 - fajl:
# 533411815016.pdf].
#
# ANALIZA OSJETLJIVOSTI (Korak 7 istog prirucnika): prag zavisi od
# izbora minimalnog broja komentara po kanalu. Testirano:
#   >=30  komentara -> n=58, prag -3.45
#   >=50  komentara -> n=57, prag -3.67   (usvojeno)
#   >=100 komentara -> n=41, prag -6.01
#   >=200 komentara -> n=29, prag -3.80
# Prag ostaje u uskom opsegu (-3.45 do -6.01) kroz sve varijante,
# dakle rezultat nije artefakt jedne metodoloske odluke. Puna analiza
# je zapisana u reference_sentiment.json i reprodukuje se skriptom
# scripts/calibrate_sentiment_reference.py.
#
# EKSTERNA VALIDACIJA PRAGA: Cruickshank, Ng & Soofi (2024), DIVERSE
# dataset (arXiv:2403.03334v3, Tabela III) izvjestavaju raspodjelu
# stavova na 215.509 komentara sa zvanicnog YouTube kanala americke
# vojske: neto -3.96 (stav prema kanalu) i -6.97 (stav prema videu).
# Oba aktiviraju cap, a autori taj kanal i sami opisuju kao "cause
# for concern" iz ugla brenda. Istovremeno, nijedan od cetiri kanala
# testirana tokom razvoja (neto 10.0 do 48.5) ne aktivira cap. Prag
# je dakle validiran sa OBJE strane - ne "puca" na normalne slucajeve,
# a hvata dokumentovan negativan slucaj.
# OGRANICENJE: DIVERSE vrijednosti dolaze iz stance-oznacavanja (weak
# supervision + LLM ansambl), ne iz sentiment klasifikacije istim
# instrumentom kao nas pipeline - koristiti kao indikativan eksterni
# test, ne kao mjerenje na istoj skali.
#
# TEORIJSKO OPRAVDANJE ZA NEKOMPENZATORNI CAP (a ne samo ~6% udjela u
# ponderisanom zbiru): negativan sentiment ne djeluje linearno niti se
# ponistava pozitivnim. Negativne informacije percipiraju se
# intenzivnije od pozitivnih (Pratto & John, 1991, "automatic
# vigilance"), pa i mali broj izrazito negativnih komentara moze imati
# nesrazmjeran reputacioni efekat (Rieger, Schmitt & Frischlich,
# 2018) - oboje citirano u Döring, N. & Mohseni, M. R. (2020),
# "Gendered hate speech in YouTube and YouNow comments", Studies in
# Communication and Media, 9(1), 62-88.
#
# KONVERGENTNA VALIDNOST: u istom referentnom datasetu, korelacija
# izmedju neto sentimenta i procenta toksicnih komentara po kanalu
# iznosi r = -0.634. Dvije nezavisno oznacene dimenzije koreliraju u
# smjeru koji teorija predvidja, sto potkrepljuje da mjera hvata ono
# sto tvrdi da hvata. (Toksicnost i sentiment su srodni, ali razliciti
# konstrukti - Obadimu, Mead & Agarwal izricito navode da je "toxicity
# analysis different from sentiment analysis".)
#
# PREOSTALO OGRANICENJE 1: ovo je STATICKI Z-score na jednom
# "presjeku" sentimenta, ne volatility-baziran pristup koji Alipour
# et al. (2025, preprint, ssrn7102482) pokazuju kao jaci prediktor
# reputacionog rizika (AUC 0.70-0.78 na stvarnim podacima).
# Volatility pristup zahtijeva pracenje sentimenta kroz vise
# vremenskih tacaka, sto trenutni pipeline (jedan uzorak komentara po
# analizi) ne podrzava - navedeno kao pravac buduceg istrazivanja.
#
# PREOSTALO OGRANICENJE 2: referentna distribucija NIJE stratifikovana
# po zanru, iako podaci pokazuju da bi to bilo opravdano (kanali sa
# vijestima: mean 18.85, n=15; ostali kanali: mean 35.78, n=42 -
# razlika od skoro 17 poena). Stratifikacija nije uvedena jer bi
# zahtijevala rucnu, nedokumentovanu klasifikaciju kanala po zanru,
# cime bi se uveo NOVI neutemeljen parametar umjesto postojeceg. Za
# buduci rad: koristiti YouTube kategoriju kanala preko API-ja kao
# objektivan kriterijum stratifikacije.
#
# PREOSTALO OGRANICENJE 3: instrument oznacavanja u referentnom
# datasetu nije isti kao nas (nlptown/bert-base-multilingual-uncased-
# sentiment). Vrijednosti su uporedive po skali i kategorijama, ali ne
# i po instrumentu - navedeno kao ogranicenje.
# ---------------------------------------------------------------------

_SENTIMENT_REFERENCE_PATH = os.path.join(
    os.path.dirname(__file__), "reference_sentiment.json"
)


def _load_sentiment_reference() -> dict:
    """
    Ucitava referentnu distribuciju neto sentimenta sa diska.

    Fajl se generise skriptom scripts/calibrate_sentiment_reference.py
    iz spoljasnjeg dataseta YouTube komentara. Vrijednosti mu i sigma
    se dakle NE upisuju rucno u kod, nego se IZVODE dokumentovanim i
    ponovljivim postupkom - u skladu sa zahtjevom OECD/JRC prirucnika
    da cijeli postupak konstrukcije kompozitnog indikatora bude
    transparentan ("The absence of an 'objective' way of determining
    weights and aggregation methods does not necessarily lead to
    rejection of the validity of composite indicators, as long as the
    entire process is transparent").

    Ako fajl ne postoji ili je neispravan, vraca prazan dict - cap se
    tada NE aktivira (bezbjedan fallback: alat radije nece kazniti
    kanal nego sto ce ga kazniti na osnovu nepostojecih referentnih
    podataka).
    """
    if not os.path.exists(_SENTIMENT_REFERENCE_PATH):
        return {}
    try:
        with open(_SENTIMENT_REFERENCE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


_SENTIMENT_REFERENCE = _load_sentiment_reference()

_SENTIMENT_Z_THRESHOLD = -2.0  # Daranda et al. (2026), isti princip, obrnut smjer


def get_sentiment_reference_stats() -> dict:
    """
    Vraca parametre referentne distribucije sentimenta i njihovu
    provenijenciju.

    Izlaze se kroz API radi transparentnosti - korisnik alata (i
    citalac rada) moze vidjeti NA OSNOVU CEGA je sentiment ocijenjen,
    a ne samo koji je broj ispao.
    """
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
    """
    Z-score za sentiment_score u odnosu na spoljasnju referentnu
    distribuciju (n=57 kanala), po istoj metodologiji kao
    _calculate_ratio_z_score.

    Vraca 0.0 ako referentni podaci nisu dostupni, cime se cap ne
    aktivira.
    """
    ref = get_sentiment_reference_stats()
    if not ref["available"]:
        return 0.0
    return (sentiment_score - ref["mean"]) / ref["std"]


def _is_sentiment_anomalous(sentiment_score: float) -> bool:
    """
    True ako je neto sentiment statisticki anomalno nizak (Z < -2.0).
    Ako referentni podaci nisu dostupni, vraca False.
    """
    if not get_sentiment_reference_stats()["available"]:
        return False
    return _calculate_sentiment_z_score(sentiment_score) < _SENTIMENT_Z_THRESHOLD


# Risk cap pravila kao formalno IF-THEN mapiranje:
# "AKO je metrika M u kategoriji kriticnog nalaza, ONDA je finalni
# skor ogranicen na najvise 'cap' vrijednost", nezavisno od
# ponderisanog zbira ostalih modula (conjunctive/non-compensatory
# logika, Einhorn 1970).
#
# NAPOMENA: uslov za subscriber-ratio cap zahtijeva i
# subscriber_count (za stratifikovan Z-score) - metrics dict koji se
# prosljedjuje u apply_risk_caps mora sadrzati "subscriber_count".
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
# dataset-u (Youtube_Influencer_Analysis_-_Updated.csv, n=874 nakon
# ciscenja) NIJE uspio - power-law regresija subs->ER dala je
# R^2=0.004 (statisticki beznacajno), cak i nakon stratifikacije po
# jeziku sadrzaja (R^2=0.002-0.019) - navedeno kao negativan, ali
# vrijedan nalaz u poglavlju 4.3/5, ne kao razlog za zamjenu
# postojeceg izvora.
#
# SLOJ 3 (>= 100 miliona pretplatnika, mega-kanali): REVIDIRANO nakon
# pronalaska Wies, Bleier & Edeling (2023), "Finding Goldilocks
# Influencers", Journal of Marketing 87(3), 383-405, koji na 802
# Instagram kampanje (1700+ influensera) utvrdjuju OBRNUTO U-OBLIKOVAN
# odnos izmedju broja pratilaca i engagement-a. Umjesto proizvoljnog
# fiksnog broja, koristi se KONTINUIRANA EKSTRAPOLACIJA vec fitovane
# krive (bez gornjeg klema), sa apsolutnim podom od 1% - smjer
# potkrijepljen Wies et al. (2023), magnituda NIJE empirijski
# potvrdjena za ovaj opseg.
#
# NAPOMENA O KONZISTENTNOSTI: granice 15M i 100M koriste se i ovdje i
# u stratifikaciji autenticnosti (_SUBSCRIBER_TIERS) - namjerno, radi
# jedinstvene strukture "rezova" velicine kanala kroz cijeli model.
# ---------------------------------------------------------------------

_LARGE_CHANNEL_CURVE_A = 2415.774517
_LARGE_CHANNEL_CURVE_B = -0.329418
_LARGE_CHANNEL_THRESHOLD = 15_000_000
_LARGE_CHANNEL_MIN_SUBS = 18_400_000   # donja granica stvarno mjerenog opsega (Tabela 1)
_LARGE_CHANNEL_MAX_SUBS = 35_900_000   # gornja granica stvarno mjerenog opsega (Tabela 1)

_SMALL_CHANNEL_FALLBACK_ER = 5.0  # sredina "dobrog" opsega 3-7%, industrijski izvor, NIJE akademski potvrdjeno

_MEGA_CHANNEL_THRESHOLD = 100_000_000
_MEGA_CHANNEL_MIN_BENCHMARK = 1.0  # apsolutni pod - sprjecava nerealisticno nisku vrijednost za astronomske brojeve pretplatnika


# ---------------------------------------------------------------------
# ZANROVSKA KOREKCIJA ENGAGEMENT BENCHMARKA (poglavlje 4.3 - "zanrovski
# confound" problem, otkriven testiranjem MKBHD + Samsung).
#
# PROBLEM: benchmark iznad je kalibrisan ISKLJUCIVO na kanalima
# gaming/entertainment zanra (Lopez-Navarrete et al., 2021, Tabela 1:
# ElRubiusOMG, Vegetta777, AuronPlay). Ovaj zanr ima strukturno visok
# engagement (brze, emotivne reakcije publike) u odnosu na npr. tech
# recenzije, vijesti ili edukativan sadrzaj, gdje publika manje
# "reaguje" cak i kad je sadrzaj kvalitetan i gledan. Primjena
# gaming-kalibrisanog benchmarka univerzalno na SVE zanrove
# sistematski potcjenjuje kreatore van gaming/entertainment niše.
#
# IZVOR ZA KOREKCIJU: Rieder, B., Coromina, O. & Matamoros-Fernandez,
# A. (2020), "Mapping YouTube: A quantitative exploration of a
# platformed media system", First Monday, 25(8) - fajl:
# View_of_Mapping_YouTube_First_Monday.pdf, Tabela 10 ("Per-video
# averages for top-level categories, 100k+ tier"). Tabela daje
# "intensity" metriku (zbir lajkova+dislajkova+komentara / pregledi *
# 100) po zanrovskoj kategoriji, za elitne (100k+ pretplatnika) kanale:
#
#   Gaming: 4.7  |  Lifestyle: 3.9  |  Society: 3.8  |  Knowledge: 3.7
#   Entertainment: 3.5  |  Sports: 3.2  |  none: 3.1  |  Music: 2.9
#
# METODOLOSKO OGRANICENJE (transparentno navedeno, status 🟡 ne 🟢):
# 1. "Intensity" ukljucuje dislajkove (mi ih ne mozemo mjeriti - vidi
#    ogranicenje kod engagement_rate formule) i racuna se kao prosjek
#    POJEDINACNIH video-omjera, ne omjer zbirova kao nas engagement_rate
#    - korekcioni faktor je stoga PRIBLIZNA, ne matematicki ekvivalentna
#    procjena razlike izmedju zanrova.
# 2. Tabela 10 daje samo JEDNU agregiranu vrijednost po zanru (nema
#    raspodjele, nema broja uzorka po kategoriji) - za razliku od
#    stratifikacije autenticnosti (n=625 sirovih tacaka), ovdje je
#    moguc samo grub MNOZILAC, ne pravi percentil unutar zanra.
# 3. Mapiranje nasih kreator-persona niša (Tech, Comedy, Education...)
#    na 8 kategorija iz Tabele 10 je AUTORSKA interpretacija (vidi
#    _NICHE_TO_TABLE10_CATEGORY), ne direktno iz izvora.
#
# BAZNA KATEGORIJA: Gaming (4.7), jer su Lopez-Navarrete kalibracioni
# kanali (ElRubiusOMG, Vegetta777, AuronPlay) najbliži tom zanru -
# korekcioni faktor je odnos ciljne kategorije prema Gaming intenzitetu.
# ---------------------------------------------------------------------

_TABLE10_INTENSITY = {
    "Entertainment": 3.5,
    "Gaming": 4.7,
    "Knowledge": 3.7,
    "Lifestyle": 3.9,
    "Music": 2.9,
    "Society": 3.8,
    "Sports": 3.2,
    "none": 3.1,
}

_CALIBRATION_BASELINE_CATEGORY = "Gaming"

# Mapiranje nasih creator_persona.primary_niche vrijednosti (vidi
# content_analysis_service.py prompt) na 8 kategorija iz Tabele 10.
# AUTORSKA INTERPRETACIJA - nije iz izvora, vidi napomenu iznad.
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


def _genre_adjustment_factor(primary_niche: str) -> float:
    """
    Vraca mnozilac za engagement benchmark na osnovu zanra kreatora,
    relativno na Gaming (bazna kalibraciona kategorija). Vrijednost
    <1.0 znaci da zanr ima strukturno nizi ocekivan engagement od
    gaming-a (npr. Knowledge/tech: 3.7/4.7 ≈ 0.787), pa se benchmark
    ublazava; >1.0 bi znacilo visi ocekivan engagement (npr. da
    postoji zanr iznad Gaming-a, sto ovdje nije slucaj).
    """
    category = _map_niche_to_table10_category(primary_niche)
    baseline = _TABLE10_INTENSITY[_CALIBRATION_BASELINE_CATEGORY]
    return _TABLE10_INTENSITY.get(category, baseline) / baseline


def _engagement_benchmark_for_size(subscriber_count: int, primary_niche: str = None) -> float:
    """
    Vraca ocekivan "odlican" engagement rate (%) za dati broj
    pretplatnika, PRILAGODJEN i zanru kreatora ako je poznat.
    TROSLOJNI pristup po velicini (nepromijenjen), sa dodatnim
    zanrovskim mnoziocem primijenjenim na kraju:

    SLOJ 1 - mega-kanali (>= 100M): kontinuirana ekstrapolacija
    fitovane krive (BEZ gornjeg klema), sa apsolutnim podom od 1%.
    Smjer opadanja potkrijepljen Wies et al. (2023, Journal of
    Marketing), magnituda NIJE empirijski potvrdjena za ovaj opseg -
    tretirati kao najbolju trenutno dostupnu procjenu, ne konacan broj.

    SLOJ 2 - veliki kanali (15M-100M): kriva klemovana na stvarno
    mjeren opseg (18.4M-35.9M, Lopez-Navarrete Tabela 1).

    SLOJ 3 - manji kanali (< 15M): industrijski fallback (5%).

    ZANROVSKA KOREKCIJA (poglavlje 4.3): mnozi se sa
    _genre_adjustment_factor(primary_niche) - vidi napomenu iznad za
    izvor i ogranicenja. Ako primary_niche nije prosljedjen
    (kompatibilnost unazad), korekcija se ne primjenjuje (faktor=1.0).
    """
    if subscriber_count >= _MEGA_CHANNEL_THRESHOLD:
        extrapolated = _LARGE_CHANNEL_CURVE_A * (subscriber_count ** _LARGE_CHANNEL_CURVE_B)
        base = max(_MEGA_CHANNEL_MIN_BENCHMARK, extrapolated)
    elif subscriber_count >= _LARGE_CHANNEL_THRESHOLD:
        clamped = max(_LARGE_CHANNEL_MIN_SUBS, min(subscriber_count, _LARGE_CHANNEL_MAX_SUBS))
        base = _LARGE_CHANNEL_CURVE_A * (clamped ** _LARGE_CHANNEL_CURVE_B)
    else:
        base = _SMALL_CHANNEL_FALLBACK_ER

    genre_factor = _genre_adjustment_factor(primary_niche) if primary_niche else 1.0
    return round(base * genre_factor, 3)


def normalize_quantitative_score(
    quant_metrics: dict, subscriber_count: int = None, primary_niche: str = None
) -> float:
    """
    Pretvara sirove kvantitativne metrike u jedinstven skor 0-100.

    Engagement komponenta se poredi sa troslojnim benchmarkom
    prilagodjenim i velicini i zanru kanala (vidi
    _engagement_benchmark_for_size) umjesto univerzalnog fiksnog
    praga ili proizvoljnih stepenastih kategorija.

    primary_niche: creator_persona.primary_niche iz
        content_analysis_service.py (npr. "Tech") - opciono, ako nije
        poznat, zanrovska korekcija se ne primjenjuje.
    """
    if subscriber_count is not None:
        benchmark = _engagement_benchmark_for_size(subscriber_count, primary_niche)
    else:
        benchmark = 10.0  # fallback za kompatibilnost unazad

    engagement = min(quant_metrics["engagement_rate"] / benchmark * 100, 100)
    view_stability = max(0, 100 - quant_metrics["view_consistency_cv"])

    return round((engagement + view_stability) / 2, 2)


def normalize_authenticity_score(quant_metrics: dict, subscriber_count: int = None) -> float:
    """
    Percentil-baziran skor autenticnosti, STRATIFIKOVAN po velicini
    kanala (vidi 4.1.3 i napomenu iznad _SUBSCRIBER_TIERS) - manji
    subscriber/view ratio, U ODNOSU NA REFERENTNE KANALE SLICNE
    VELICINE, znaci bolju autenticnost.

    ZAMIJENJEN RANIJI PRISTUP (dvije uzastopne izmjene):
    1. Prvobitno: heuristicki mnozilac (score = 100 - ratio*3), bez
       empirijskog opravdanja.
    2. Zamijenjen NESTRATIFIKOVANIM percentil pristupom (InCites
       metod, Bornmann & Williams, 2020) - matematicki ispravnije, ali
       testiranje (MKBHD + Samsung, poglavlje 4.3) otkrilo je da
       jedinstvena referentna distribucija sistematski kaznjava velike
       kanale, jer dataset dominiraju manji kreatori.
    3. TRENUTNA VERZIJA: percentil pristup i dalje, ali sada
       STRATIFIKOVAN po velicini kanala - kanal se poredi samo sa
       referentnim kanalima slicnog obima pretplatnika, cime se
       otklanja pristrasnost prema velikim kanalima.

    Ako subscriber_count nije prosljedjen (kompatibilnost unazad),
    koristi se najveci sloj (15M+) kao razuman fallback - ali funkcija
    bi UVIJEK trebalo da dobije subscriber_count u praksi (vidi pozive
    u calculate_final_risk_score i calculate_bulk_risk_scores).

    Preostalo metodolosko ogranicenje (transparentno navedeno): pravi
    "zlatni standard" pristup bi racunao Cohen's d effect size na
    OSNOVU LABEL-OVANOG dataseta (potvrdjeno legitimni vs. potvrdjeno
    suspendovani/anomalni kanali), po uzoru na Shajari & Agarwal
    (2025). Ovaj rad umjesto toga koristi NElabel-ovanu referentnu
    distribuciju (bez oznake "dobar"/"los" kanal) - percentil rang
    govori "koliko je ovaj kanal neuobicajen u odnosu na kanale slicne
    velicine", ne "koliko je vjerovatno da je ovaj kanal lazan/
    botovan". Ova razlika je eksplicitno navedena kao ogranicenje i
    pravac buduceg istrazivanja (poglavlje 5).
    """
    ratio = quant_metrics["subscriber_to_view_ratio"]

    # ISPRAVKA: ranije je fallback bio _SUBSCRIBER_TIERS[-2][0]
    # (= 5.000.000, sloj "vrlo veliki 5M-15M"), iako je prateci
    # komentar tvrdio da tretira kanal kao "velikan" sloj. Sada se
    # koristi _FALLBACK_TIER_LO, koji pokazuje na posljednji sloj (15M+).
    if subscriber_count is None:
        subscriber_count = _FALLBACK_TIER_LO

    tier_ratios = _get_tier_ratios_cached(subscriber_count)
    percentile = _percentile_score(ratio, tier_ratios)

    # JEDNOSMJERNA transformacija (poglavlje 4.3): kanal NA ILI ISPOD
    # medijane (percentile<=50) dobija pun skor (100) - nizak ratio
    # nikad nije signal rizika. Skor opada SAMO za kanale IZNAD
    # medijane, gdje visi ratio ukazuje na moguce neaktivnu/kupljenu
    # bazu. Uskladjeno sa jednosmjernim Z-score cap mehanizmom
    # (_calculate_ratio_z_score aktivira se samo kad je ratio VISI od
    # praga) i istim principom kod sentiment Z-score-a (samo prevelika
    # negativnost je rizik).
    if percentile <= 50:
        return 100.0
    return round(100 - (percentile - 50) * 2, 2)


def normalize_sentiment_score(sentiment_result: dict) -> float:
    """
    Sentiment score je vec u opsegu -100 do 100, pretvaramo u 0-100.

    Cista linearna transformacija bez slobodnih parametara - ne
    zahtijeva poseban akademski izvor van osnovne aritmetike.
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

    NAPOMENA: metrics dict MORA sadrzati "subscriber_count" (pored
    postojecih sentiment_score, subscriber_to_view_ratio,
    brand_fit_score), zbog stratifikovanog Z-score cap-a. Ako je
    vrijednost None, koristi se fallback sloj (vidi
    _get_tier_ratios_cached) umjesto da se baci izuzetak.
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


def build_sentiment_diagnostics(sentiment_result: dict) -> dict:
    """
    Dijagnostika sentiment modula za API odgovor.

    Izlaze se sirovi neto sentiment, sva tri udjela (positive/neutral/
    negative), Z-score i parametri referentne distribucije.

    Prikaz UDJELA uz neto vrijednost je metodoloski bitan, ne
    kozmeticki: neto sentiment sam po sebi kolapsira dvije razlicite
    situacije u istu vrijednost - kanal sa 100% neutralnih komentara i
    kanal sa 50% pozitivnih i 50% negativnih oba daju neto 0, iako
    prvi ukazuje na ravnodusnu, a drugi na polarizovanu publiku.
    Istovremeni prikaz udjela omogucava razlikovanje ta dva slucaja.
    """
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


# GRANICE KATEGORIJA RIZIKA
#
# Raniji pristup: tercili (33./67. percentil) uzorka od pet rucno
# izabranih parova. Napusten iz dva razloga.
#
# Prvo, kruznost: parove je birao autor, bodovao ih je autorov model, pa
# su se iz tih bodova izvodile kategorije tog istog modela - model je
# tako sam definisao sta je za njega normalno.
#
# Drugo, nestabilnost: sa n=5 tercil-granice su bukvalno drugi i cetvrti
# par u nizu, pa je svaka izmjena bilo kog modula pomjerala granice za
# desetine poena. To se u toku razvoja dogodilo dva puta (prelazak na
# stratifikovanu autenticnost, pa ispravka brand-fit kalibracije).
#
# Sadasnji pristup koristi isti skup parova na kojem je kalibrisan i
# brand-fit modul (Semeradova & Weinlich, 2023, Tabela 2): 34 stvarna,
# dokumentovana para i 102 ukrstena, kontrolna para. Postojanje
# kontrolne grupe mijenja prirodu granica - one vise nisu podjela
# proizvoljnog uzorka na tri dijela, nego tacke razdvajanja dviju
# poznatih populacija:
#
#   low_threshold  - Youdenov indeks: tacka koja maksimizuje
#                    J = osjetljivost + specificnost - 1, dakle
#                    najbolje razdvaja dokumentovane saradnje od
#                    nasumicno sastavljenih parova. Tri nezavisna
#                    postupka (75. percentil ukrstenih, tercil svih
#                    parova, Youden) daju vrijednosti unutar dva
#                    poena - granica nije osjetljiva na izbor metoda,
#                    sto je samo po sebi rezultat analize osjetljivosti.
#
#   high_threshold - NE izvodi se iz percentila donjeg repa. Risk cap
#                    pravila obaraju skor na fiksne vrijednosti (30,
#                    35, 40, 50), pa se u donjem dijelu skale mjerenja
#                    gomilaju na tim konstantama - percentil bi ocitao
#                    vrijednost capa, a ne distribuciju. Umjesto toga,
#                    granica se poravnava sa samim cap sistemom:
#                    postavlja se iznad najstrozih capova i ispod
#                    najblazeg, tako da svaki ozbiljan aktiviran cap
#                    automatski povlaci kategoriju "Visok rizik".
#                    Nekompenzatorna logika i kategorizacija time
#                    govore isto, umjesto da budu dva nezavisna
#                    mehanizma.
#
# Granica prihvatljivosti je, u krajnjoj liniji, izraz tolerancije
# donosioca odluke na rizik, a ne empirijska cinjenica - Duijm (2015)
# to iznosi eksplicitno, tvrdeci da bojenje matrice rizika predstavlja
# definiciju rizika samo po sebi. Empirijske granice ovdje sluze kao
# sidro, ne kao dokaz. Zato uz njih ide i analiza osjetljivosti
# (poglavlje 4.3).
#
# Vrijednosti se ucitavaju iz backend/reference_risk_thresholds.json,
# koji generise scripts/calibrate_risk_thresholds.py. Skriptu ponovo
# pokrenuti nakon svake izmjene koja utice na bilo koji modul - time
# kalibracija postaje reproducibilna procedura umjesto seta brojeva
# prekucanih u kod.
# ---------------------------------------------------------------------
 
_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "reference_risk_thresholds.json"
)
 
# Granice iz uzorka od pet parova, zadrzane iskljucivo kao rezerva ako
# kalibracioni fajl nedostaje (npr. pri prvom pokretanju skripte, koja
# i sama poziva calculate_final_risk_score). Nisu za produkcijsku
# upotrebu - source polje to jasno oznacava.
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
    """
    Vraca granice kategorija rizika izvedene iz kalibracionog skupa sa
    kontrolnom grupom. Vidi obrazlozenje iznad.
    """
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
    """
    Objedinjuje sve module u finalni risk score.

    subscriber_count: broj pretplatnika kanala - koristi se za
        engagement benchmark PRILAGODJEN velicini kanala (vidi
        normalize_quantitative_score), za STRATIFIKOVAN authenticity
        percentil skor (vidi normalize_authenticity_score), i za
        stratifikovan Z-score risk cap (vidi apply_risk_caps).
    primary_niche: creator_persona.primary_niche (npr. "Tech") - ako je
        poznat, koristi se za ZANROVSKU KOREKCIJU engagement benchmarka
        (vidi _engagement_benchmark_for_size i napomenu iznad
        _TABLE10_INTENSITY). Opciono - ako nije prosljedjen, koristi se
        samo velicinski prilagodjen benchmark (bez zanrovske korekcije).
    """
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


# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats, get_comments_for_videos
    from scoring import calculate_quantitative_metrics
    from ai_service import analyze_comments_batch
    from brand_fit import calculate_brand_fit_score

    # Provjera referentne distribucije sentimenta prije analize
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