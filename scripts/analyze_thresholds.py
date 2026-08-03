"""
Dijagnostika granica kategorija rizika.

Ne poziva nijedan API - cita samo backend/reference_risk_thresholds.json
koji je vec generisan. Sluzi da se, prije nego sto se granice fiksiraju,
vidi oblik obje distribucije, popunjenost kategorija pod razlicitim
kandidatima za granice, i da li se razdvajanje razlikuje po brendu.

POKRETANJE (iz korijena repoa)
------------------------------
    python scripts/analyze_thresholds.py
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "backend" / "reference_risk_thresholds.json"


def cliffs_delta(a, b):
    if not a or not b:
        return None
    veci = sum(1 for x in a for y in b if x > y)
    manji = sum(1 for x in a for y in b if x < y)
    return round((veci - manji) / (len(a) * len(b)), 4)


def tumaci_deltu(d):
    if d is None:
        return "-"
    ad = abs(d)
    if ad < 0.147:
        return "zanemarljiv"
    if ad < 0.33:
        return "mali"
    if ad < 0.474:
        return "srednji"
    return "veliki"


def opis(v):
    o = sorted(v)
    n = len(o)

    def p(q):
        idx = max(0, int(q * n) - 1) if q <= 0.5 else min(n - 1, int(q * n))
        return round(o[idx], 2)

    return {
        "n": n, "min": round(o[0], 2), "p05": p(0.05), "p10": p(0.10),
        "p25": p(0.25), "med": round(statistics.median(o), 2),
        "p75": p(0.75), "p90": p(0.90), "p95": p(0.95),
        "max": round(o[-1], 2),
        "sredina": round(statistics.mean(o), 2),
        "std": round(statistics.stdev(o), 2) if n > 1 else 0.0,
    }


def raspodjela(stvarni, ukrsteni, low, high):
    """Koliko procenata svake grupe pada u koju kategoriju."""
    def udio(v):
        n = len(v)
        nizak = sum(1 for x in v if x >= low)
        visok = sum(1 for x in v if x < high)
        return (round(nizak / n * 100, 1),
                round((n - nizak - visok) / n * 100, 1),
                round(visok / n * 100, 1))
    return udio(stvarni), udio(ukrsteni)


def main():
    d = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    parovi = d["svi_parovi"]

    stvarni = [p["final_score"] for p in parovi if p["stvarni_par"]]
    ukrsteni = [p["final_score"] for p in parovi if not p["stvarni_par"]]

    print("=" * 74)
    print("DISTRIBUCIJE FINALNIH SKOROVA")
    print("=" * 74)
    for ime, v in [("stvarni", stvarni), ("ukrsteni", ukrsteni)]:
        o = opis(v)
        print(f"\n{ime.upper()}  n={o['n']}  sredina={o['sredina']}  std={o['std']}")
        print(f"  min={o['min']}  p05={o['p05']}  p10={o['p10']}  p25={o['p25']}  "
              f"med={o['med']}")
        print(f"  p75={o['p75']}  p90={o['p90']}  p95={o['p95']}  max={o['max']}")

    print("\n" + "=" * 74)
    print("KANDIDATI ZA GRANICE - popunjenost kategorija")
    print("(nizak% / srednji% / visok%)")
    print("=" * 74)

    o_s, o_u = opis(stvarni), opis(ukrsteni)
    sve = sorted(stvarni + ukrsteni)
    t33 = round(sve[max(0, int(0.33 * len(sve)) - 1)], 2)
    t67 = round(sve[min(len(sve) - 1, int(0.67 * len(sve)))], 2)
    youden = d.get("validacija", {}).get("youden", {}).get("cut")

    kandidati = [
        ("p95 ukrstenih / p05 stvarnih", o_u["p95"], o_s["p05"]),
        ("p95 ukrstenih / p25 stvarnih", o_u["p95"], o_s["p25"]),
        ("p90 ukrstenih / p10 stvarnih", o_u["p90"], o_s["p10"]),
        ("p75 ukrstenih / p25 stvarnih", o_u["p75"], o_s["p25"]),
        ("tercili svih parova", t67, t33),
    ]
    if youden:
        kandidati.append(("Youden / p25 ukrstenih", youden, o_u["p25"]))

    print(f"\n{'varijanta':32s} {'low':>7s} {'high':>7s}   "
          f"{'STVARNI':>20s}   {'UKRSTENI':>20s}")
    for ime, low, high in kandidati:
        if low <= high:
            print(f"{ime:32s} {low:7.2f} {high:7.2f}   NEVALIDNO (low <= high)")
            continue
        s, u = raspodjela(stvarni, ukrsteni, low, high)
        print(f"{ime:32s} {low:7.2f} {high:7.2f}   "
              f"{s[0]:5.1f}/{s[1]:5.1f}/{s[2]:5.1f}   "
              f"{u[0]:5.1f}/{u[1]:5.1f}/{u[2]:5.1f}")

    y = d.get("validacija", {}).get("youden", {})
    if y:
        print(f"\nYoudenova granica: {y.get('cut')}  J={y.get('J')}  "
              f"osjetljivost={y.get('osjetljivost')}  "
              f"specificnost={y.get('specificnost')}")

    print("\n" + "=" * 74)
    print("RAZDVAJANJE PO BRENDU")
    print("(Semeradova & Weinlich navode cetiri brenda vrlo razlicitog tipa -")
    print(" nise brendovi vs. masovni sponzori. Ako se delta bitno razlikuje,")
    print(" to je nalaz, ne greska.)")
    print("=" * 74)
    print(f"\n{'brend':24s} {'n_st':>5s} {'n_uk':>5s} {'sred_st':>8s} "
          f"{'sred_uk':>8s} {'delta':>7s}  tumacenje")
    for brend in sorted({p["brend"] for p in parovi}):
        s = [p["final_score"] for p in parovi
             if p["brend"] == brend and p["stvarni_par"]]
        u = [p["final_score"] for p in parovi
             if p["brend"] == brend and not p["stvarni_par"]]
        dd = cliffs_delta(s, u)
        print(f"{brend:24s} {len(s):5d} {len(u):5d} "
              f"{statistics.mean(s):8.2f} {statistics.mean(u):8.2f} "
              f"{dd if dd is not None else 0:7.4f}  {tumaci_deltu(dd)}")

    print("\n" + "=" * 74)
    print("PET NAJNIZIH I PET NAJVISIH STVARNIH PAROVA")
    print("(provjera da li p05 vuce jedan atipican par)")
    print("=" * 74)
    st = sorted((p for p in parovi if p["stvarni_par"]),
                key=lambda p: p["final_score"])
    for p in st[:5] + st[-5:]:
        print(f"  {p['final_score']:6.2f}  {p['kanal'][:28]:28s} x {p['brend'][:18]:18s} "
              f"bf={p['brand_fit_score']:6.2f}  caps={p['triggered_caps']}")


if __name__ == "__main__":
    main()