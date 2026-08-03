"""
Test uticaja jezika i formatiranja opisa brenda na kosinusnu slicnost.

PITANJE
-------
Opisi brendova u kesu generisani su na srpskom, dok se sadrzajni profil
kanala gradi iz naslova videa i opisa kanala, koji su kod vecine
testiranih kanala na engleskom. Embedding model jeste visejezican, ali
se postavlja pitanje da li poredjenje preko jezicke granice sistematski
snizava izmjerenu slicnost.

Dodatno, generisani opisi sadrze markdown formatiranje (zvjezdice) i
kod nekih brendova pocinju nazivom brenda, a kod drugih ne - sto ih
cini medjusobno neuporedivim.

POSTUPAK
--------
Isti kanal se poredi sa vise varijanti opisa ISTOG brenda:
  1. srpski, sa markdownom i nazivom brenda (kako je sada u kesu)
  2. srpski, ocisceno
  3. engleski, ocisceno

Buduci da je sadrzajni profil kanala u sve tri varijante identican,
svaka razlika u rezultatu pripisuje se iskljucivo opisu.

POKRETANJE (iz korijena repoa)
------------------------------
    python scripts/test_description_language.py "@NikkieTutorials"
"""

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

# Varijante opisa istog brenda. Sadrzajno ekvivalentne - razlikuju se
# samo po jeziku i formatiranju.
VARIJANTE = {
    "Prada": [
        (
            "srpski + markdown + naziv",
            "**Prada** — Luksuzna moda.  Prodaje: torbice, kožna galanterija, "
            "obuća, odjeća (konfekcija), nakit i modni dodaci. Teme: visoka "
            "moda, dizajn, luksuz, modne revije.",
        ),
        (
            "srpski, ocisceno",
            "Luksuzna moda. Prodaje: torbice, kožna galanterija, obuća, odjeća, "
            "nakit, modni dodaci. Teme: visoka moda, dizajn, luksuz, modne revije.",
        ),
        (
            "engleski, ocisceno",
            "Luxury fashion. Sells: handbags, leather goods, footwear, "
            "ready-to-wear clothing, jewellery, fashion accessories. Themes: "
            "high fashion, design, luxury, runway shows.",
        ),
    ],
    "Booking.com": [
        (
            "srpski + markdown + naziv",
            "**Booking.com**\n Online turistička agencija, smještaj i putovanja. "
            "Prodaje: rezervacije hotela, apartmana, letova, najam automobila. "
            "Teme: putovanja, odmor, destinacije, smještaj.",
        ),
        (
            "srpski, ocisceno",
            "Online turistička agencija, smještaj i putovanja. Prodaje: "
            "rezervacije hotela, apartmana, letova, najam automobila. Teme: "
            "putovanja, odmor, destinacije, smještaj.",
        ),
        (
            "engleski, ocisceno",
            "Online travel agency, accommodation and travel. Sells: hotel "
            "bookings, apartments, flights, car rental. Themes: travel, holiday, "
            "destinations, accommodation.",
        ),
    ],
}


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            'Upotreba: python scripts/test_description_language.py "@handle" [Brend]'
        )

    handle = sys.argv[1]
    brend = sys.argv[2] if len(sys.argv) > 2 else "Prada"

    if brend not in VARIJANTE:
        raise SystemExit(f"Nema varijanti za '{brend}'. Dostupno: {list(VARIJANTE)}")

    print(f"Kanal: {handle}   Brend: {brend}")
    print("Povlacim sadrzaj kanala (racuna se JEDNOM za sve varijante)...")

    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)

    try:
        transcripts = get_transcripts_for_videos(
            video_ids, max_videos=6, max_words_per_video=50
        )
    except Exception as exc:
        print(f"  upozorenje: transkripti nedostupni ({exc})")
        transcripts = None

    profil = build_channel_content_summary(stats, videos, transcripts=transcripts)
    profil_emb = get_embedding(profil)

    print()
    print("=" * 66)
    print(f"{stats['title']}  x  {brend}")
    print("=" * 66)
    print(f"{'Varijanta opisa':<32}{'slicnost':>10}{'razlika':>12}")
    print("-" * 66)

    osnovna = None
    for naziv, opis in VARIJANTE[brend]:
        sim = cosine_similarity(get_embedding(opis), profil_emb)
        if osnovna is None:
            osnovna = sim
            print(f"{naziv:<32}{sim:>10.4f}{'-':>12}")
        else:
            print(f"{naziv:<32}{sim:>10.4f}{sim - osnovna:>+12.4f}")

    print("-" * 66)
    print()
    print("TUMACENJE")
    print("  Razlika izmedju prve i druge varijante = uticaj markdowna i")
    print("  naziva brenda na pocetku.")
    print("  Razlika izmedju druge i trece = uticaj jezika opisa.")
    print()
    print("  Ako je jezicka razlika izrazena, to je ozbiljno ogranicenje:")
    print("  kalibracioni opseg izveden je iskljucivo na engleskim")
    print("  parovima, pa srpski opisi nisu uporedivi sa tom skalom.")
    print()
    print("  Napomena: sadrzajni profil kanala je u sve tri varijante")
    print("  identican, pa je svaka razlika iskljucivo posljedica opisa.")


if __name__ == "__main__":
    main()