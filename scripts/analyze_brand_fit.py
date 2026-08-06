"""Statisticka validacija brand-fit modula: Mann-Whitney U test stvarnih vs. ukrstenih parova."""

import json
import statistics
from pathlib import Path

from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "backend" / "reference_brand_fit.json"


def cliffs_delta(a: list, b: list) -> tuple:
    """Cliffov delta: (P(a>b) - P(a<b))."""
    veci = sum(1 for x in a for y in b if x > y)
    manji = sum(1 for x in a for y in b if x < y)
    ukupno = len(a) * len(b)
    d = (veci - manji) / ukupno

    ad = abs(d)
    if ad < 0.147:
        opis = "zanemarljiv"
    elif ad < 0.33:
        opis = "mali"
    elif ad < 0.474:
        opis = "srednji"
    else:
        opis = "veliki"
    return round(d, 4), opis


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"Nema {DATA_PATH}. Prvo pokrenuti scripts/calibrate_brand_fit.py"
        )

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    rows = data["sve_kombinacije"]

    stvarni = [r["cosine_similarity"] for r in rows if r["stvarni_par"]]
    ukrsteni = [r["cosine_similarity"] for r in rows if not r["stvarni_par"]]

    if len(stvarni) < 3 or len(ukrsteni) < 3:
        raise SystemExit("Premalo podataka za test.")

    u_stat, p = mannwhitneyu(stvarni, ukrsteni, alternative="greater")
    rb = 1 - (2 * u_stat) / (len(stvarni) * len(ukrsteni))  # rank-biserial
    d, d_opis = cliffs_delta(stvarni, ukrsteni)

    print("=" * 62)
    print("VALIDACIJA BRAND-FIT MODULA")
    print("=" * 62)
    print(f"Stvarni parovi   n={len(stvarni):<4} medijana={statistics.median(stvarni):.4f}")
    print(f"Ukrsteni parovi  n={len(ukrsteni):<4} medijana={statistics.median(ukrsteni):.4f}")
    print()
    print("Mann-Whitney U (jednosmjeran, H1: stvarni > ukrsteni)")
    print(f"  U = {u_stat:.1f}")
    print(f"  p = {p:.6f}")
    print(f"  rank-biserial r = {abs(rb):.4f}")
    print(f"  Cliffov delta = {d} ({d_opis} efekat)")
    print()

    if p < 0.05:
        print("ZAKLJUCAK: H0 se odbacuje (p < 0.05).")
        print("Stvarni parovi imaju statisticki znacajno visu kosinusnu")
        print("slicnost od ukrstenih. Mjera razlikuje poklapanje od")
        print("nepoklapanja - diskriminantna validnost je potvrdjena.")
    else:
        print("ZAKLJUCAK: H0 se NE odbacuje (p >= 0.05).")
        print("Mjera ne razlikuje stvarne od ukrstenih parova na ovom")
        print("uzorku. Ovo je NEGATIVAN NALAZ i mora se navesti u radu -")
        print("vidi nize moguce uzroke.")
        print()
        print("Moguci uzroci koje treba ispitati prije zakljucka:")
        print("  1. Sadrzajni profil kanala je previse opsti (naslovi i")
        print("     tagovi ne razlikuju dovoljno tematiku)")
        print("  2. Opisi brendova su previse slicni jedan drugom")
        print("  3. Pretpostavka 'saradnja = poklapanje' ne vazi za")
        print("     Audible i Skillshare (masovno oglasavanje)")
        print("  4. Kanali su se sadrzajno promijenili od 2022.")

    print()
    print("-" * 62)
    print("PO BRENDU")
    print("-" * 62)
    print("Ocekivanje: niche brendovi (ThredUp, Function of Beauty) treba")
    print("da pokazu vecu razliku od masovnih oglasivaca (Audible,")
    print("Skillshare). Razlika medju njima je nalaz, ne greska.")
    print()
    for ime, d_brend in data["po_brendu"].items():
        s = [r["cosine_similarity"] for r in rows if r["stvarni_par"] and r["brend"] == ime]
        u = [r["cosine_similarity"] for r in rows if not r["stvarni_par"] and r["brend"] == ime]
        if len(s) < 3 or len(u) < 3:
            continue
        _, p_b = mannwhitneyu(s, u, alternative="greater")
        d_b, opis_b = cliffs_delta(s, u)
        razlika = statistics.mean(s) - statistics.mean(u)
        print(f"{ime:<20} razlika={razlika:+.4f}  p={p_b:.4f}  delta={d_b} ({opis_b})")

    print()
    print("-" * 62)
    print("NAJVISE I NAJNIZE OCIJENJENE KOMBINACIJE")
    print("-" * 62)
    for r in rows[:5]:
        oznaka = "STVARNI " if r["stvarni_par"] else "ukrsteni"
        print(f"  {r['cosine_similarity']:.4f}  {oznaka}  {r['brend']:<20} x {r['kanal']}")
    print("  ...")
    for r in rows[-5:]:
        oznaka = "STVARNI " if r["stvarni_par"] else "ukrsteni"
        print(f"  {r['cosine_similarity']:.4f}  {oznaka}  {r['brend']:<20} x {r['kanal']}")

    print()
    print("Provjeriti rucno: da li se medju najvisim kombinacijama")
    print("nalaze ukrsteni parovi koji intuitivno IMAJU smisla")
    print("(npr. Skillshare x kreativni kanal)? Ako da, to nije greska")
    print("modela nego ogranicenje pretpostavke da je ukrsteni par nuzno")
    print("los par - i tako treba biti navedeno u radu.")


if __name__ == "__main__":
    main()