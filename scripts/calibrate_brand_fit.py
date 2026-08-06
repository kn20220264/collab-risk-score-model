"""Kalibracija brand-fit opsega [min, max] na dokumentovanim parovima brend-kreator (Semeradova & Weinlich, 2023)."""

import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.youtube_service import (
    get_channel_stats,
    get_recent_video_ids,
    get_videos_stats,
)
from backend.transcript_service import get_transcripts_for_videos
from backend.brand_fit import (
    build_channel_content_summary,
    get_embedding,
    cosine_similarity,
)

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = ROOT / "scripts" / "brand_fit_pairs.json"
CACHE_PATH = ROOT / "scripts" / ".brand_fit_cache.json"
OUT_PATH = ROOT / "backend" / "reference_brand_fit.json"

# Isti parametri kao u produkcijskom pozivu, radi uporedivosti.
MAX_VIDEOS = 20
MAX_TRANSCRIPTS = 6
MAX_WORDS_PER_TRANSCRIPT = 50


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def fetch_channel_embedding(handle: str, cache: dict) -> list:
    """Vraca (kesirani) embedding sadrzajnog profila kanala."""
    if handle in cache:
        return cache[handle]

    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=MAX_VIDEOS)
    videos = get_videos_stats(video_ids)

    try:
        transcripts = get_transcripts_for_videos(
            video_ids,
            max_videos=MAX_TRANSCRIPTS,
            max_words_per_video=MAX_WORDS_PER_TRANSCRIPT,
        )
    except Exception as exc:
        print(f"    upozorenje: transkripti nedostupni ({exc}) - koristi se samo Nivo 1")
        transcripts = None

    profile = build_channel_content_summary(stats, videos, transcripts=transcripts)
    embedding = get_embedding(profile)

    cache[handle] = embedding
    _save_cache(cache)
    return embedding


def summarize(values: list) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else None,
        "median": round(statistics.median(values), 4),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
        "p05": round(ordered[max(0, int(0.05 * len(ordered)) - 1)], 4),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
    }


def main() -> None:
    config = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    brands = config["brendovi"]
    pairs = config["parovi"]

    aktivni = [p for p in pairs if p.get("handle", "").strip()]
    if not aktivni:
        raise SystemExit(
            "Nijedan par nema popunjen 'handle'. Popuniti scripts/brand_fit_pairs.json."
        )
    print(f"Parova sa popunjenim handle-om: {len(aktivni)} od {len(pairs)}\n")

    # Embeddinzi brendova - cetiri poziva ukupno.
    print("Racunam embeddinge brendova...")
    brand_embeddings = {ime: get_embedding(b["opis"]) for ime, b in brands.items()}

    # Embeddinzi kanala - jedan poziv po kanalu, kesirano.
    cache = _load_cache()
    channel_embeddings = {}
    neuspjeli = []

    for i, par in enumerate(aktivni, 1):
        handle = par["handle"].strip()
        print(f"[{i}/{len(aktivni)}] {par['kanal']} ({handle})")
        try:
            channel_embeddings[handle] = fetch_channel_embedding(handle, cache)
        except Exception as exc:
            print(f"    GRESKA: {exc}")
            neuspjeli.append({"kanal": par["kanal"], "handle": handle, "greska": str(exc)})
            continue
        time.sleep(0.3)  # blaga pauza prema API-ju

    print()

    # Sve kombinacije: svaki kanal protiv svakog brenda.
    rezultati = []
    for par in aktivni:
        handle = par["handle"].strip()
        if handle not in channel_embeddings:
            continue
        for ime_brenda, brand_emb in brand_embeddings.items():
            sim = cosine_similarity(brand_emb, channel_embeddings[handle])
            rezultati.append({
                "kanal": par["kanal"],
                "handle": handle,
                "brend": ime_brenda,
                "stvarni_par": ime_brenda == par["brend"],
                "cosine_similarity": round(sim, 4),
            })

    stvarni = [r["cosine_similarity"] for r in rezultati if r["stvarni_par"]]
    ukrsteni = [r["cosine_similarity"] for r in rezultati if not r["stvarni_par"]]

    # donja = medijana ukrstenih, gornja = p95 stvarnih parova
    donja = round(statistics.median(ukrsteni), 4) if ukrsteni else None
    gornja = summarize(stvarni).get("p95")

    po_brendu = {}
    for ime in brands:
        s = [r["cosine_similarity"] for r in rezultati if r["stvarni_par"] and r["brend"] == ime]
        u = [r["cosine_similarity"] for r in rezultati if not r["stvarni_par"] and r["brend"] == ime]
        po_brendu[ime] = {"stvarni": summarize(s), "ukrsteni": summarize(u)}

    payload = {
        "_opis": "Kalibracija brand-fit modula. Generisano skriptom scripts/calibrate_brand_fit.py - ne uredjivati rucno.",
        "izvor_parova": config["_izvor_parova"],
        "parametri": {
            "max_videos": MAX_VIDEOS,
            "max_transcripts": MAX_TRANSCRIPTS,
            "parova_pokusano": len(aktivni),
            "parova_uspjesno": len(channel_embeddings),
            "neuspjeli_kanali": neuspjeli,
        },
        "stvarni_parovi": summarize(stvarni),
        "ukrsteni_parovi": summarize(ukrsteni),
        "kalibracioni_opseg": {
            "min_expected_similarity": donja,
            "max_expected_similarity": gornja,
            "obrazlozenje": (
                "Donja granica = medijana ukrstenih parova (tipicna vrijednost "
                "pri nepoklapanju). Gornja granica = 95. percentil stvarnih "
                "parova (vrlo dobro poklapanje, otporno na pojedinacni ekstrem)."
            ),
        },
        "po_brendu": po_brendu,
        "sve_kombinacije": sorted(rezultati, key=lambda r: -r["cosine_similarity"]),
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 62)
    print("REZULTAT")
    print("=" * 62)
    s, u = summarize(stvarni), summarize(ukrsteni)
    print(f"Stvarni parovi   n={s['n']:<4} mean={s['mean']:.4f}  sd={s['std']}  raspon {s['min']}-{s['max']}")
    print(f"Ukrsteni parovi  n={u['n']:<4} mean={u['mean']:.4f}  sd={u['std']}  raspon {u['min']}-{u['max']}")
    print(f"Razlika sredina: {s['mean'] - u['mean']:.4f}")
    print()
    print(f"PREDLOZENI KALIBRACIONI OPSEG: [{donja}, {gornja}]")
    print(f"(trenutno u kodu: [0.15, 0.45])")
    print()
    print("Po brendu (samo stvarni parovi):")
    for ime, d in po_brendu.items():
        if d["stvarni"]:
            print(f"  {ime:<20} n={d['stvarni']['n']}  mean={d['stvarni']['mean']:.4f}")
    print()
    print(f"Zapisano u: {OUT_PATH}")
    print()
    print("SLJEDECI KORAK: statisticki test razlike izmedju stvarnih i")
    print("ukrstenih parova (Mann-Whitney U) - vidi analyze_brand_fit.py")


if __name__ == "__main__":
    main()