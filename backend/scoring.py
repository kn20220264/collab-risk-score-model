"""
Modul kvantitativnih metrika kanala.

Formule i izvori (za poglavlje 4.1 rada):
- Engagement rate: (lajkovi + komentari) / pregledi * 100
  Izvor: Lopez-Navarrete, A.J. et al. (2021). "Formula para medir el
  engagement del espectador en YouTube". Revista Mediterranea de
  Comunicacion, 12(2), 143-156. (peer-reviewed, koristi identican oblik
  formule, bez dislajkova jer YouTube API vise ne vraca javni broj
  dislajkova).
- Konzistentnost pregleda: koeficijent varijacije CV = sigma/mu.
  Standardna deskriptivna statisticka mjera varijabilnosti.
- Konzistentnost objavljivanja: prosjek i standardna devijacija razmaka
  (u danima) izmedju uzastopnih objava. Operacionalizacija autora, jer
  ne postoji jedinstvena formalna formula u pregledanoj literaturi.
- Subscriber-to-view ratio: pretplatnici / prosjecni pregledi.
  Heuristika inspirisana praksom komercijalnih alata (HypeAuditor,
  CreatorScore) za detekciju "naduvane" baze pratilaca; tacne formule
  tih alata nisu javno objavljene (proprietary).
"""

from datetime import datetime
from statistics import mean, pstdev


def _parse_date(video: dict) -> datetime:
    return datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ")


def calculate_engagement_rate(videos: list) -> float:
    """
    ER = (ukupno lajkova + ukupno komentara) / ukupno pregleda * 100,
    usrednjeno preko poslednjih N videa (Lopez-Navarrete et al., 2021).
    """
    total_likes = sum(v["like_count"] for v in videos)
    total_comments = sum(v["comment_count"] for v in videos)
    total_views = sum(v["view_count"] for v in videos)

    if total_views == 0:
        return 0.0

    return round((total_likes + total_comments) / total_views * 100, 4)


def calculate_view_consistency_cv(videos: list) -> float:
    """
    Koeficijent varijacije (CV = sigma/mu * 100) broja pregleda
    poslednjih N videa. Veci CV = manje predvidiv doseg.
    """
    views = [v["view_count"] for v in videos]
    if len(views) < 2:
        return 0.0

    mu = mean(views)
    if mu == 0:
        return 0.0

    sigma = pstdev(views)
    return round((sigma / mu) * 100, 2)


def calculate_posting_consistency(videos: list) -> dict:
    """
    Koeficijent varijacije (CV = sigma/mu) razmaka (u danima) izmedju
    uzastopnih objava. Isti statisticki pristup kao
    calculate_view_consistency_cv (koeficijent varijacije), primijenjen
    ovdje na razmake objavljivanja umjesto na broj pregleda.

    METODOLOSKO OBRAZLOZENJE (izmjena u odnosu na raniju verziju):
    Ranija verzija je vracala prosjek i std. devijaciju razmaka kao dva
    odvojena, nepovezana broja, bez formalnog nacina da se oni objedine
    u jedinstven, uporediv skor. Umjesto toga, konzistentnost
    objavljivanja se sada racuna kao CV (isto kao konzistentnost
    pregleda), sto joj daje:
    (1) Formalno, vec potkrijepljeno statisticko uporiste - CV je
        standardna deskriptivna statisticka mjera relativne
        varijabilnosti, koja se ovdje koristi na identican nacin kao
        za konzistentnost pregleda (vidi calculate_view_consistency_cv).
    (2) Jedinstvenu skalu uporedivu izmedju modula - manji CV = veca
        predvidivost/konzistentnost, bilo da se odnosi na razmake
        objavljivanja ili na broj pregleda, sto omogucava dosljedniju
        interpretaciju kroz cijeli kvantitativni modul.
    Ovim se izbjegava potreba za posebnim, nepotkrijepljenim izvorom
    za "konzistentnost objavljivanja" kao zaseban koncept - umjesto
    toga, koristi se ista, vec dokumentovana tehnika primijenjena na
    drugi tip podataka.

    Vraca i dalje avg_gap_days/std_gap_days (opisni podaci, korisni za
    izvjestavanje i AI obrazlozenje), ali dodaje posting_consistency_cv
    kao glavnu, uporedivu mjeru za dalju upotrebu u agregaciji.
    """
    if len(videos) < 2:
        return {"avg_gap_days": 0.0, "std_gap_days": 0.0, "posting_consistency_cv": 0.0}

    dates = sorted((_parse_date(v) for v in videos), reverse=True)
    gaps = [
        (dates[i] - dates[i + 1]).total_seconds() / 86400
        for i in range(len(dates) - 1)
    ]

    avg_gap = mean(gaps)
    std_gap = pstdev(gaps) if len(gaps) > 1 else 0.0

    # CV = sigma/mu * 100, isti obrazac kao calculate_view_consistency_cv
    cv = round((std_gap / avg_gap) * 100, 2) if avg_gap > 0 else 0.0

    return {
        "avg_gap_days": round(avg_gap, 2),
        "std_gap_days": round(std_gap, 2),
        "posting_consistency_cv": cv,
    }

def calculate_subscriber_to_view_ratio(channel_stats: dict, videos: list) -> float:
    """
    Odnos pretplatnici : prosjecni pregledi po videu.
    Visok odnos moze ukazivati na neaktivnu ili "kupljenu" bazu pratilaca
    u odnosu na stvarni doseg sadrzaja.
    """
    if not videos:
        return 0.0

    avg_views = mean(v["view_count"] for v in videos)
    if avg_views == 0:
        return float("inf")

    return round(channel_stats["subscriber_count"] / avg_views, 2)


def calculate_quantitative_metrics(channel_stats: dict, videos: list) -> dict:
    posting = calculate_posting_consistency(videos)

    return {
        "engagement_rate": calculate_engagement_rate(videos),
        "view_consistency_cv": calculate_view_consistency_cv(videos),
        "avg_posting_gap_days": posting["avg_gap_days"],
        "std_posting_gap_days": posting["std_gap_days"],
        "posting_consistency_cv": posting["posting_consistency_cv"],
        "subscriber_to_view_ratio": calculate_subscriber_to_view_ratio(channel_stats, videos),
        "sample_size": len(videos),
    }

# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)

    result = calculate_quantitative_metrics(stats, videos)
    print("KVANTITATIVNE METRIKE:")
    for key, value in result.items():
        print(f"  {key}: {value}")
