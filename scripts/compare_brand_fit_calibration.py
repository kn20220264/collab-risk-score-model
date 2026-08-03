"""
Poredjenje brand-fit skora pod starom i novom kalibracijom.

Racuna sirovu kosinusnu slicnost za zadati par brend-kanal, pa prikazuje
kako se ista vrijednost preslikava u skor pod:
  - STAROM kalibracijom  [0.15, 0.45]  - procjena iz n=4 para
  - NOVOM kalibracijom   iz reference_brand_fit.json - 144 mjerenja
    na 36 dokumentovanih parova (Semeradova & Weinlich, 2023)

POKRETANJE (iz korijena repoa)
------------------------------
    python scripts/compare_brand_fit_calibration.py @mkbhd "Apple"

Drugi argument je naziv brenda; opis se uzima iz recnika OPISI_BRENDOVA
nize, ili se moze proslijediti kao treci argument:

    python scripts/compare_brand_fit_calibration.py @mkbhd "Apple" "Tech brand selling..."

NAPOMENA O REPRODUCIBILNOSTI: opisi brendova su ovdje FIKSIRANI, a ne
generisani pozivom jezickog modela, da bi ponovljeno pokretanje davalo
istu vrijednost.
"""

import json
import sys
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
REF_PATH = ROOT / "backend" / "reference_brand_fit.json"

# Stare granice - trenutno hardkodovane u backend/brand_fit.py
STARA_MIN = 0.15
STARA_MAX = 0.45

# Fiksirani opisi brendova koriscenih u testiranju, radi reproducibilnosti.
OPISI_BRENDOVA = {
    "Apple": (
        "Tech brand selling premium smartphones, laptops, tablets, "
        "smartwatches and consumer electronics accessories, focused on "
        "innovation, industrial design, ecosystem integration and build quality."
    ),
    "NYX Professional Makeup": (
        "Affordable professional makeup brand selling lipsticks, foundations, "
        "eyeshadow palettes, primers and makeup tools. Positions itself around "
        "bold colour, artistry, cruelty-free formulation and accessibility."
    ),
    "Skyscanner": (
        "Travel search engine for comparing flights, hotels and car rental. "
        "Positions itself around finding cheap fares, flexible travel planning "
        "and discovering destinations."
    ),
    "Booking.com": (
        "Online travel agency for booking accommodation, flights, car rental "
        "and attractions worldwide. Positions itself around choice, verified "
        "reviews and free cancellation."
    ),
    "Prada": (
        "Italian luxury fashion house selling ready-to-wear clothing, leather "
        "handbags, footwear, eyewear and fragrance. Positions itself around "
        "craftsmanship, minimalist design, heritage and high fashion."
    ),
    "SurfShark": (
        "VPN and online privacy service offering encrypted connections, ad "
        "blocking, antivirus and identity protection. Positions itself around "
        "digital security, privacy from tracking and access to geo-restricted content."
    ),
    "Red Bull": (
        "Energy drink brand associated with extreme sports, motorsport, "
        "esports, music events and adventure content. Positions itself around "
        "energy, performance, adrenaline and youth culture."
    ),
}


def skala(sim: float, lo: float, hi: float) -> float:
    """Min-max preslikavanje kosinusne slicnosti u skor 0-100."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(100.0, (sim - lo) / (hi - lo) * 100.0))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            'Upotreba: python scripts/compare_brand_fit_calibration.py <@handle> "<Brend>" ["opis"]'
        )

    handle = sys.argv[1]
    brend = sys.argv[2]
    opis = sys.argv[3] if len(sys.argv) > 3 else OPISI_BRENDOVA.get(brend)

    if not opis:
        raise SystemExit(
            f'Nema fiksiranog opisa za "{brend}". Dodati ga u OPISI_BRENDOVA '
            f"ili proslijediti kao treci argument."
        )

    if not REF_PATH.exists():
        raise SystemExit(
            f"Nema {REF_PATH}. Prvo pokrenuti scripts/calibrate_brand_fit.py"
        )

    ref = json.loads(REF_PATH.read_text(encoding="utf-8"))
    nova_min = ref["kalibracioni_opseg"]["min_expected_similarity"]
    nova_max = ref["kalibracioni_opseg"]["max_expected_similarity"]

    print(f"Kanal: {handle}")
    print(f"Brend: {brend}")
    print("Povlacim podatke...")

    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)

    try:
        transcripts = get_transcripts_for_videos(
            video_ids, max_videos=6, max_words_per_video=50
        )
        nivo = "Nivo 1 + Nivo 2 (sa transkriptima)"
    except Exception as exc:
        print(f"  upozorenje: transkripti nedostupni ({exc})")
        transcripts = None
        nivo = "samo Nivo 1 (bez transkripata)"

    profil = build_channel_content_summary(stats, videos, transcripts=transcripts)
    sim = cosine_similarity(get_embedding(opis), get_embedding(profil))

    stari = skala(sim, STARA_MIN, STARA_MAX)
    novi = skala(sim, nova_min, nova_max)

    print()
    print("=" * 60)
    print(f"{stats['title']}  x  {brend}")
    print("=" * 60)
    print(f"Izvor sadrzaja: {nivo}")
    print(f"Sirova kosinusna slicnost: {sim:.4f}")
    print()
    print(f"{'Kalibracija':<14}{'Opseg':<20}{'Skor':>8}")
    print("-" * 44)
    print(f"{'stara':<14}[{STARA_MIN}, {STARA_MAX}]{'':<8}{stari:>8.2f}")
    print(f"{'nova':<14}[{nova_min}, {nova_max}]{'':<4}{novi:>8.2f}")
    print("-" * 44)
    print(f"{'razlika':<14}{'':<20}{novi - stari:>+8.2f}")
    print()

    # Kontekst: gdje ova vrijednost stoji u kalibracionom uzorku
    stvarni = [r["cosine_similarity"] for r in ref["sve_kombinacije"] if r["stvarni_par"]]
    ukrsteni = [r["cosine_similarity"] for r in ref["sve_kombinacije"] if not r["stvarni_par"]]
    pct_st = sum(1 for x in stvarni if x < sim) / len(stvarni) * 100
    pct_uk = sum(1 for x in ukrsteni if x < sim) / len(ukrsteni) * 100

    print("Polozaj u kalibracionom uzorku:")
    print(f"  visa od {pct_st:.0f}% stvarnih parova (n={len(stvarni)})")
    print(f"  visa od {pct_uk:.0f}% ukrstenih parova (n={len(ukrsteni)})")
    print()
    print("Napomena: brand-fit nosi 52% tezine, pa se razlika u modulu")
    print("mnozi sa 0.52 pri prenosu na finalni skor. Granice kategorija")
    print("rizika izvedene su iz finalnih skorova pod STAROM kalibracijom")
    print("i moraju se ponovo izracunati.")


if __name__ == "__main__":
    main()