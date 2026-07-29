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
- Subscriber-to-view ratio: pretplatnici / CJELOZIVOTNI prosjecni
  pregledi po videu (vidi VAZNA ISPRAVKA ispod).
  Heuristika inspirisana praksom komercijalnih alata (HypeAuditor,
  CreatorScore) za detekciju "naduvane" baze pratilaca; tacne formule
  tih alata nisu javno objavljene (proprietary).

VAZNA ISPRAVKA (otkrivena testiranjem MKBHD + Samsung i MKBHD + Apple,
poglavlje 4.3 - "recency bias" problem):

Prethodna verzija racunala je subscriber_to_view_ratio kao odnos
pretplatnika i PROSJECNIH PREGLEDA POSLEDNJIH N (recentnih) VIDEA:

    avg_views = mean(v["view_count"] for v in videos)  # videos=recentnih 20

Ovo je stvaralo sistematsku, VELICINSKI NEZAVISNU pristrasnost pri
poredjenju sa referentnom distribucijom (n=625 kanala, Youtube_
Influencer_Analysis_-_Updated.csv, poglavlje 4.1.3): referentni
dataset ima MEDIJANU STAROSTI VIDEA OD ~1600 DANA (4.4 godine) - dakle
videa koji su imali godine da akumuliraju preglede. Nasuprot tome,
"poslednjih 20 videa" analiziranog kanala su TEK OBJAVLJENI - jos
uvijek akumuliraju preglede, koji rastu godinama nakon objave. Ovo je
sistematski naduvavalo (pogorsavalo) ratio SVAKOG kanala analiziranog
uzivo, nezavisno od velicine ili autenticnosti - "mlad" video se
neopravdano poredio sa "sazrelim, starim" videima iz referentnog
skupa.

RJESENJE: subscriber_to_view_ratio se sada racuna kao odnos
pretplatnika i CJELOZIVOTNOG prosjeka pregleda po videu na nivou
CIJELOG KANALA (channel_stats["view_count"] / channel_stats
["video_count"], oba direktno dostupna iz YouTube Data API-ja bez
dodatnih poziva), a NE prosjeka uzorka od N recentnih videa. Ovo je
direktno uporedivo sa referentnim CSV-om, koji takodje sadrzi
cjelozivotne kanal-nivo kolone ("Total Chanel Views", "No of Videos
the Channel"), na osnovu kojih je i referentna distribucija
rekonstruisana (vidi risk_aggregation.py, _load_reference_data).

Preostalo ogranicenje (transparentno navedeno): cjelozivotni prosjek
i dalje ne razlikuje kanal koji je oduvijek imao stabilan doseg od
onog cija je popularnost naglo opala (ili porasla) - za takvu analizu
bio bi potreban vremenski niz podataka kroz vise mjeseci/godina, sto
trenutni jednokratni (snapshot) pipeline ne prikuplja. Navedeno kao
pravac buduceg istrazivanja (poglavlje 5).
"""

from datetime import datetime
from statistics import mean, pstdev


def _parse_date(video: dict) -> datetime:
    return datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ")


def calculate_engagement_rate(videos: list) -> float:
    """
    ER = (ukupno lajkova + ukupno komentara) / ukupno pregleda * 100,
    usrednjeno preko poslednjih N videa (Lopez-Navarrete et al., 2021).

    NAPOMENA: ovdje NAMJERNO ostaje "poslednjih N videa" (ne
    cjelozivotni prosjek) - engagement rate mjeri TRENUTNU angazovanost
    publike, za koju je najnoviji sadrzaj relevantniji signal od
    istorijskog prosjeka (vidi troslojni benchmark u
    risk_aggregation.py, koji je kalibrisan upravo na ovakvom,
    recentnom shvatanju ER-a). Ispravka opisana na vrhu fajla odnosi se
    ISKLJUCIVO na subscriber_to_view_ratio, ne na engagement_rate.
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
    Odnos pretplatnici : CJELOZIVOTNI prosjecni pregledi po videu.
    Visok odnos moze ukazivati na neaktivnu ili "kupljenu" bazu pratilaca
    u odnosu na stvarni doseg sadrzaja.

    ISPRAVKA (vidi napomenu na vrhu fajla - "recency bias"): koristi
    channel_stats["view_count"] (ukupan broj pregleda CIJELOG kanala,
    kroz cijelu istoriju) i channel_stats["video_count"] (ukupan broj
    objavljenih videa), oba direktno iz YouTube Data API-ja
    (youtube_service.get_channel_stats), umjesto prosjeka pregleda
    SAMO poslednjih N (recentnih) videa. Ovo uklanja sistematsku
    pristrasnost prema "mladim" videima koji jos nisu akumulirali
    preglede, i cini ratio direktno uporedivim sa referentnom
    distribucijom (risk_aggregation.py), koja je takodje izracunata na
    cjelozivotnom, kanal-nivo prosjeku (Total Chanel Views / No of
    Videos the Channel iz istog CSV-a).

    `videos` parametar se vise ne koristi u samom racunu (zadrzan u
    potpisu funkcije radi kompatibilnosti poziva u
    calculate_quantitative_metrics), ali napomena ostaje: broj
    objavljenih videa iz channel_stats moze biti (blago) veci od
    broja u `videos` listi, jer `videos` obicno sadrzi samo uzorak
    (npr. poslednjih 20) - to je ocekivano i namjerno, jer je cilj
    CIJELA istorija kanala, ne uzorak.
    """
    video_count = channel_stats.get("video_count", 0)
    total_views = channel_stats.get("view_count", 0)

    if video_count == 0 or total_views == 0:
        return 0.0

    lifetime_avg_views = total_views / video_count
    if lifetime_avg_views == 0:
        return float("inf")

    return round(channel_stats["subscriber_count"] / lifetime_avg_views, 2)


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
    print(f"\n(Za provjeru: channel_stats video_count={stats['video_count']}, "
          f"view_count={stats['view_count']}, lifetime_avg_views="
          f"{stats['view_count']/stats['video_count']:.0f})")