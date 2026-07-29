"""
Kalibracija referentne distribucije neto sentimenta po kanalu.

SVRHA
-----
Ovaj skript izvodi parametre (mu, sigma) referentne distribucije neto
sentimenta iz spoljasnjeg dataseta YouTube komentara. Time se prag za
sentiment risk cap NE postavlja kao proizvoljna konstanta, nego se
IZVODI dokumentovanim, ponovljivim postupkom - u skladu sa zahtjevom
iz OECD/JRC prirucnika za kompozitne indikatore (Nardo et al., 2005,
STD/DOC(2005)3, Korak 7: Robustness and sensitivity), koji trazi da
budu "dokumentovane i objasnjene analize osjetljivosti i njihovi
rezultati".

ULAZ
----
CSV sa najmanje sljedecim kolonama:
    channel_username  - identifikator kanala
    label_sentiment   - jedna od: positive / neutral / negative

VAZNO - PROVENIJENCIJA: prije citiranja rezultata u radu OBAVEZNO
dopuniti sekciju DATASET_METADATA nize (izvor, autor, godina, licenca,
metod oznacavanja). Bez toga rezultati nisu citabilni.

IZLAZ
-----
backend/reference_sentiment.json

POKRETANJE
----------
    python scripts/calibrate_sentiment_reference.py path/do/dataset.csv
"""

import json
import statistics
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------
# METAPODACI O DATASETU - POPUNITI PRIJE CITIRANJA U RADU
# ---------------------------------------------------------------------
DATASET_METADATA = {
    "naziv": "YouTube Comments Dataset with Sentiment, Toxicity and Spam Labels",
    "izvor_url": "POPUNITI",
    "autor": "POPUNITI",
    "godina": "POPUNITI",
    "licenca": "POPUNITI",
    "metod_oznacavanja": "POPUNITI - ljudski anotatori ili automatski model?",
    "napomena": (
        "Ako su oznake generisane automatski, to je ogranicenje koje "
        "mora biti navedeno u radu (Korak 7, analiza nesigurnosti)."
    ),
}

# Minimalan broj komentara po kanalu da bi kanal usao u referentni uzorak.
# Vrijednost 50 nije proizvoljna: pri manje od ~50 komentara procenti se
# pomjeraju u koracima od preko 2 procentna poena po komentaru, sto unosi
# vjestacku varijansu u procjenu sigma. Osjetljivost na ovaj izbor se
# testira u funkciji sensitivity_analysis() nize i izvjestava u JSON-u.
MIN_COMMENTS_PER_CHANNEL = 50

# Z-score prag preuzet iz Daranda et al. (2026), fajl: 14_2_01_Daranda.pdf,
# Tabela 2: "Z-score threshold 2.0 - 95% confidence interval; 11% false
# positive rate". Isti prag se vec koristi za subscriber-to-view ratio,
# cime je pristup konzistentan kroz cijeli model.
Z_SCORE_THRESHOLD = 2.0


def compute_channel_net_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Racuna neto sentiment po kanalu: %pozitivnih - %negativnih.

    Formula je identicna onoj u backend/ai_service.py, cime je osigurano
    da su referentne vrijednosti i vrijednosti koje alat proizvodi
    mjerene na istoj skali.
    """
    counts = (
        df.groupby("channel_username")["label_sentiment"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for col in ("positive", "neutral", "negative"):
        if col not in counts.columns:
            counts[col] = 0

    counts["n_comments"] = counts[["positive", "neutral", "negative"]].sum(axis=1)
    counts["positive_pct"] = counts["positive"] / counts["n_comments"] * 100
    counts["neutral_pct"] = counts["neutral"] / counts["n_comments"] * 100
    counts["negative_pct"] = counts["negative"] / counts["n_comments"] * 100
    counts["net_sentiment"] = counts["positive_pct"] - counts["negative_pct"]
    return counts


def test_normality(values) -> dict:
    """
    Shapiro-Wilk test normalnosti.

    Z-score kao metod pretpostavlja priblizno normalnu distribuciju
    referentne populacije. Ova pretpostavka se ovdje EKSPLICITNO
    testira, umjesto da se preuti - odgovara Koraku 3 (Multivariate
    analysis) iz OECD/JRC prirucnika.
    """
    statistic, p_value = stats.shapiro(values)
    return {
        "test": "Shapiro-Wilk",
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "normalnost_odbacena": bool(p_value < 0.05),
        "skewness": round(float(stats.skew(values)), 4),
        "interpretacija": (
            "p >= 0.05 znaci da se hipoteza o normalnosti NE odbacuje, "
            "cime je upotreba Z-score pristupa parametarski opravdana."
        ),
    }


def sensitivity_analysis(counts: pd.DataFrame) -> list:
    """
    Analiza osjetljivosti praga na izbor MIN_COMMENTS_PER_CHANNEL.

    Trazeno Korakom 7 OECD/JRC prirucnika: pokazuje koliko izvedeni
    prag zavisi od jedne metodoloske odluke konstruktora.
    """
    results = []
    for min_n in (1, 30, 50, 100, 200):
        subset = counts[counts["n_comments"] >= min_n]["net_sentiment"]
        if len(subset) < 3:
            continue
        mean = statistics.mean(subset)
        std = statistics.stdev(subset)
        results.append(
            {
                "min_comments_per_channel": min_n,
                "n_channels": len(subset),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "cap_threshold": round(mean - Z_SCORE_THRESHOLD * std, 4),
            }
        )
    return results


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    required = {"channel_username", "label_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV-u nedostaju obavezne kolone: {missing}")

    counts = compute_channel_net_sentiment(df)
    eligible = counts[counts["n_comments"] >= MIN_COMMENTS_PER_CHANNEL]

    if len(eligible) < 30:
        print(
            f"UPOZORENJE: samo {len(eligible)} kanala prolazi prag od "
            f"{MIN_COMMENTS_PER_CHANNEL} komentara. Uzorak ispod n=30 "
            f"treba tretirati kao privremenu procjenu."
        )

    net = eligible["net_sentiment"]
    mean = statistics.mean(net)
    std = statistics.stdev(net)
    threshold = mean - Z_SCORE_THRESHOLD * std

    payload = {
        "_opis": (
            "Referentna distribucija neto sentimenta po kanalu, izvedena "
            "iz spoljasnjeg dataseta. Generisano skriptom "
            "scripts/calibrate_sentiment_reference.py - ne uredjivati rucno."
        ),
        "dataset": DATASET_METADATA,
        "parametri_izvodjenja": {
            "min_comments_per_channel": MIN_COMMENTS_PER_CHANNEL,
            "z_score_threshold": Z_SCORE_THRESHOLD,
            "z_score_izvor": "Daranda et al. (2026), fajl: 14_2_01_Daranda.pdf, Tabela 2",
            "ukupno_komentara": int(df.shape[0]),
            "ukupno_kanala_u_datasetu": int(counts.shape[0]),
            "kanala_u_referentnom_uzorku": int(len(eligible)),
        },
        "distribucija": {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "median": round(float(net.median()), 4),
            "min": round(float(net.min()), 4),
            "max": round(float(net.max()), 4),
            "p05": round(float(net.quantile(0.05)), 4),
            "p25": round(float(net.quantile(0.25)), 4),
            "p75": round(float(net.quantile(0.75)), 4),
            "p95": round(float(net.quantile(0.95)), 4),
        },
        "test_normalnosti": test_normality(net),
        "cap_threshold": round(threshold, 4),
        "analiza_osjetljivosti": sensitivity_analysis(counts),
        "kanali": [
            {
                "channel": str(idx),
                "n_comments": int(row["n_comments"]),
                "positive_pct": round(float(row["positive_pct"]), 2),
                "neutral_pct": round(float(row["neutral_pct"]), 2),
                "negative_pct": round(float(row["negative_pct"]), 2),
                "net_sentiment": round(float(row["net_sentiment"]), 2),
            }
            for idx, row in eligible.sort_values("net_sentiment").iterrows()
        ],
    }

    out_path = Path(__file__).resolve().parent.parent / "backend" / "reference_sentiment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Kanala u referentnom uzorku: {len(eligible)}")
    print(f"mean = {mean:.4f}")
    print(f"std  = {std:.4f}")
    print(f"cap_threshold (Z < -{Z_SCORE_THRESHOLD}) = {threshold:.4f}")
    print(f"Zapisano u: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Upotreba: python scripts/calibrate_sentiment_reference.py <dataset.csv>")
    main(sys.argv[1])