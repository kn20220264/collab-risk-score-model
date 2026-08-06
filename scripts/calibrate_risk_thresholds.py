"""Kalibracija granica kategorija rizika (nizak/srednji/visok) na parovima sa kontrolnom grupom."""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.youtube_service import (
    get_channel_stats,
    get_recent_video_ids,
    get_videos_stats,
    get_comments_for_videos,
)
from backend.scoring import calculate_quantitative_metrics
from backend.ai_service import analyze_comments_batch
from backend.brand_fit import MIN_EXPECTED_SIMILARITY, MAX_EXPECTED_SIMILARITY
from backend.risk_aggregation import calculate_final_risk_score
from backend.risk_aggregation import calculate_final_risk_score, RISK_CAP_RULES

ROOT = Path(__file__).resolve().parent.parent
PAIRS_PATH = ROOT / "backend" / "reference_two_component.json"
HANDLES_PATH = ROOT / "backend" / "reference_brand_fit.json"
CACHE_PATH = ROOT / "scripts" / ".risk_thresholds_cache.json"
OUT_PATH = ROOT / "backend" / "reference_risk_thresholds.json"

# Isti parametri kao u produkcijskom pozivu, radi uporedivosti.
MAX_VIDEOS = 20
MAX_COMMENTS_PER_VIDEO = 10


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def fetch_channel_profile(handle: str, cache: dict) -> dict:
    """Vraca (kesirane) kvantitativne metrike, sentiment i broj pretplatnika kanala."""
    if handle in cache:
        return cache[handle]

    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=MAX_VIDEOS)
    videos = get_videos_stats(video_ids)
    comments = get_comments_for_videos(video_ids, max_per_video=MAX_COMMENTS_PER_VIDEO)

    profile = {
        "subscriber_count": stats["subscriber_count"],
        "quant_metrics": calculate_quantitative_metrics(stats, videos),
        "sentiment_result": analyze_comments_batch(comments),
    }

    cache[handle] = profile
    _save_cache(cache)
    return profile


def rescale_similarity(similarity: float) -> float:
    """Preracunava sirovu kosinusnu slicnost u brand-fit skor (0-100) trenutnim granicama iz brand_fit.py."""
    rescaled = (similarity - MIN_EXPECTED_SIMILARITY) / (
        MAX_EXPECTED_SIMILARITY - MIN_EXPECTED_SIMILARITY
    )
    return round(max(0.0, min(1.0, rescaled)) * 100, 2)


def percentile(values: list, p: float) -> float:
    """Percentil po istom obrascu kao scripts/calibrate_brand_fit.py."""
    ordered = sorted(values)
    if p <= 0.5:
        idx = max(0, int(p * len(ordered)) - 1)
    else:
        idx = min(len(ordered) - 1, int(p * len(ordered)))
    return round(ordered[idx], 2)


def summarize(values: list) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "std": round(statistics.stdev(values), 2) if len(values) > 1 else None,
        "median": round(statistics.median(values), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
    }


def cliffs_delta(a: list, b: list) -> float:
    """Cliffova delta - velicina efekta za razdvajanje dvije grupe."""
    veci = sum(1 for x in a for y in b if x > y)
    manji = sum(1 for x in a for y in b if x < y)
    return round((veci - manji) / (len(a) * len(b)), 4)


def youden_cut(stvarni: list, ukrsteni: list) -> dict:
    """Granica koja maksimizuje Youdenov indeks J = osjetljivost + specificnost - 1."""
    kandidati = sorted(set(stvarni + ukrsteni))
    najbolji = {"cut": None, "J": -1}
    for c in kandidati:
        osjetljivost = sum(1 for x in stvarni if x >= c) / len(stvarni)
        specificnost = sum(1 for x in ukrsteni if x < c) / len(ukrsteni)
        J = osjetljivost + specificnost - 1
        if J > najbolji["J"]:
            najbolji = {
                "cut": round(c, 2),
                "J": round(J, 4),
                "osjetljivost": round(osjetljivost, 4),
                "specificnost": round(specificnost, 4),
            }
    return najbolji


def main() -> None:
    pairs_data = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    handles_data = json.loads(HANDLES_PATH.read_text(encoding="utf-8"))

    kombinacije = pairs_data["sve_kombinacije"]
    handle_po_kanalu = {
        x["kanal"]: x["handle"] for x in handles_data["sve_kombinacije"]
    }

    kanali = sorted({k["kanal"] for k in kombinacije})
    print(f"Parova: {len(kombinacije)}  |  jedinstvenih kanala: {len(kanali)}\n")

    cache = _load_cache()
    profili = {}
    neuspjeli = []

    for i, kanal in enumerate(kanali, 1):
        handle = handle_po_kanalu.get(kanal, "").strip()
        if not handle:
            print(f"[{i}/{len(kanali)}] {kanal} - PRESKACEM (nema handle)")
            neuspjeli.append(kanal)
            continue
        print(f"[{i}/{len(kanali)}] {kanal} ({handle})")
        try:
            profili[kanal] = fetch_channel_profile(handle, cache)
        except Exception as exc:
            print(f"    GRESKA: {exc}")
            neuspjeli.append(kanal)

    print(f"\nUspjesno: {len(profili)} kanala, neuspjelo: {len(neuspjeli)}\n")

    stvarni, ukrsteni, zapisi = [], [], []

    for par in kombinacije:
        kanal = par["kanal"]
        if kanal not in profili:
            continue
        p = profili[kanal]

        rezultat = calculate_final_risk_score(
            quant_metrics=p["quant_metrics"],
            sentiment_result=p["sentiment_result"],
            brand_fit_result={"brand_fit_score": rescale_similarity(par["sim_sadrzaj"])},
            subscriber_count=p["subscriber_count"],
            primary_niche=None,  # nedeterministican u produkciji, izostavljeno radi reproducibilnosti
        )
        skor = rezultat["final_score"]

        (stvarni if par["stvarni_par"] else ukrsteni).append(skor)
        zapisi.append({
            "kanal": kanal,
            "brend": par["brend"],
            "stvarni_par": par["stvarni_par"],
            "sim_sadrzaj": par["sim_sadrzaj"],
            "brand_fit_score": rezultat["module_scores"]["brand_fit"],
            "final_score": skor,
            "triggered_caps": rezultat["triggered_caps"],
        })

    if len(stvarni) < 10 or len(ukrsteni) < 10:
        raise SystemExit("Premalo uspjesnih parova za kalibraciju.")

    # Gornja granica (nizak rizik): Youdenov indeks
    y = youden_cut(stvarni, ukrsteni)
    low_threshold = y["cut"]

    # Donja granica (visok rizik): 33. percentil, poravnat sa cap sistemom
    # da svaki ozbiljan aktiviran cap povlaci kategoriju "Visok rizik"
    capovi = sorted(r["cap"] for r in RISK_CAP_RULES)
    strogi_cap = capovi[-2]      # najvisi od "strogih" capova
    najblazi_cap = capovi[-1]    # blagi brand-fit cap

    high_threshold = percentile(stvarni + ukrsteni, 0.33)
    if not (strogi_cap < high_threshold < najblazi_cap):
        print(f"UPOZORENJE: granica {high_threshold} nije izmedju "
              f"{strogi_cap} i {najblazi_cap} - provjeriti cap vrijednosti.")

    izlaz = {
        "_opis": (
            "Granice kategorija rizika izvedene na skupu sa kontrolnom "
            "grupom. Generisano skriptom scripts/calibrate_risk_thresholds.py "
            "- ne uredjivati rucno."
        ),
        "izvor_parova": pairs_data.get("_izvor_parova") or handles_data.get("izvor_parova"),
        "postupak": (
            "low_threshold = 95. percentil ukrstenih parova; "
            "high_threshold = 5. percentil stvarnih parova. Srednji pojas "
            "je zona preklapanja dviju distribucija."
        ),
        "ogranicenja": (
            "Zanrovska korekcija engagement benchmarka nije primijenjena "
            "(primary_niche=None) jer se u produkciji dobija nedeterministickim "
            "pozivom jezickog modela. 'Stvaran par' oznacava iskazanu "
            "preferenciju brenda, ne ishod saradnje."
        ),
        "brand_fit_opseg_u_trenutku_kalibracije": [
            MIN_EXPECTED_SIMILARITY,
            MAX_EXPECTED_SIMILARITY,
        ],
        "neuspjeli_kanali": neuspjeli,
        "stvarni_parovi": summarize(stvarni),
        "ukrsteni_parovi": summarize(ukrsteni),
        "granice": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "postupak_gornje": "Youdenov indeks nad stvarnim vs. ukrstenim parovima",
            "postupak_donje": (
                f"33. percentil svih parova, poravnat sa cap sistemom "
                f"(iznad najstrozeg capa {strogi_cap}, ispod blagog {najblazi_cap})"
            ),
        },
        "validacija": {
            "cliffs_delta": cliffs_delta(stvarni, ukrsteni),
            "youden": y,
            "cliffs_delta_po_brendu": {
                b: cliffs_delta(
                    [x["final_score"] for x in zapisi
                     if x["brend"] == b and x["stvarni_par"]],
                    [x["final_score"] for x in zapisi
                     if x["brend"] == b and not x["stvarni_par"]],
                )
                for b in sorted({x["brend"] for x in zapisi})
            },
        },
        "poredjenje_sa_starim_pristupom": {
            "tercili_svih_parova": {
                "p33": percentile(stvarni + ukrsteni, 0.33),
                "p67": percentile(stvarni + ukrsteni, 0.67),
            },
            "stare_granice_n5": {"low_threshold": 83.48, "high_threshold": 58.97},
        },
        "svi_parovi": zapisi,
    }

    OUT_PATH.write_text(
        json.dumps(izlaz, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 60)
    print(f"Stvarni parovi   n={len(stvarni):3d}  "
          f"sredina={izlaz['stvarni_parovi']['mean']:6.2f}  "
          f"p05={izlaz['stvarni_parovi']['p05']:6.2f}")
    print(f"Ukrsteni parovi  n={len(ukrsteni):3d}  "
          f"sredina={izlaz['ukrsteni_parovi']['mean']:6.2f}  "
          f"p95={izlaz['ukrsteni_parovi']['p95']:6.2f}")
    print("-" * 60)
    print(f"low_threshold  (Nizak rizik  >=) : {low_threshold}")
    print(f"high_threshold (Visok rizik   <) : {high_threshold}")
    print(f"Cliffova delta                   : {izlaz['validacija']['cliffs_delta']}")
    print(f"Youdenova granica                : {izlaz['validacija']['youden']['cut']}")
    
    print("=" * 60)
    print(f"\nZapisano u {OUT_PATH}")


if __name__ == "__main__":
    main()