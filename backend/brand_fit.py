import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"

_EMBEDDING_MAX_RETRIES = 3
_EMBEDDING_RETRY_BACKOFF_SECONDS = 2


def get_embedding(text: str) -> list:
    """
    Vraca embedding vektor za dati tekst.

    Retry sa eksponencijalnim backoff-om za tranzijentne greske
    OpenAI servera (5xx, rate limit, konekcija) - jedan povremeni
    500 ne bi trebalo da obori citavu analizu.
    """
    text = text.replace("\n", " ").strip()

    for attempt in range(_EMBEDDING_MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input=text,
                model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            is_last_attempt = attempt == _EMBEDDING_MAX_RETRIES - 1
            is_retriable = isinstance(e, (APIConnectionError, RateLimitError)) or e.status_code >= 500
            if is_last_attempt or not is_retriable:
                raise
            time.sleep(_EMBEDDING_RETRY_BACKOFF_SECONDS * (2 ** attempt))


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Racuna kosinusnu slicnost izmedju dva vektora (opseg -1 do 1).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_channel_content_summary(channel_stats: dict, videos: list, transcripts: dict = None) -> str:
    """
    Spaja opis kanala, naslove videa, tagove, teme (topic categories) i
    (opciono) transkript-isjecke u jedan tekst koji predstavlja
    'sadrzajni profil' kanala, spreman za embedding.

    NIVO 1 OBOGACIVANJE (dodano nakon testiranja - poglavlje 4.3):
    Uz naslove videa, ukljucuju se i:
    - tags: kljucne rijeci koje kreator sam dodaje svakom videu (SEO
      kategorizacija) - cesto precizniji, gusci semanticki signal nego
      sami naslovi, jer su birani specificno da opisu temu videa.
    - topic_categories: Wikipedia-bazirane teme koje YouTube AUTOMATSKI
      dodjeljuje svakom videu preko topicDetails polja - eksterna,
      standardizovana kategorizacija (npr. "Technology", "Mobile phone"),
      nezavisna od kreatorovog subjektivnog izbora rijeci.

    NIVO 2 OBOGACIVANJE (opciono, preko `transcripts` parametra):
    - transcripts: dict {video_id: transcript_snippet}, iz
      transcript_service.get_transcripts_for_videos(). Sadrzi kratke
      isjecke stvarnog govorenog sadrzaja videa (ne samo naslov/tagove),
      za do N najnovijih videa. Parametar je opcion - ako se ne
      proslijedi, sadrzajni profil koristi samo Nivo 1 (kompatibilnost
      unazad).

    Teorijsko uporiste: princip kombinovanja metapodataka i transkripta
    u jedinstven sadrzajni profil je operacionalizacija pristupa iz
    "Developing a Multimodal Approach to Channel Characterization on
    YouTube", koji eksplicitno koristi metapodatke I transkripte kao
    komplementarne izvore za semanticku analizu kanala. Ocekivani
    efekat bogaceg konteksta je takodje u skladu sa Bleckmann &
    Tschisgale (2025) hipotezom da similarity analiza bez okolnog
    konteksta predstavlja donju granicu (lower-bound) sposobnosti
    embedding modela.
    """
    parts = [channel_stats.get("description", "")]

    for v in videos:
        parts.append(v["title"])
        if v.get("tags"):
            parts.append(", ".join(v["tags"]))
        if v.get("topic_categories"):
            parts.append(", ".join(v["topic_categories"]))
        if transcripts and v.get("video_id") in transcripts:
            parts.append(transcripts[v["video_id"]])

    return " | ".join(p for p in parts if p)


def calculate_brand_fit_score(brand_description: str, channel_stats: dict, videos: list, transcripts: dict = None) -> dict:
    """
    Racuna semanticko poklapanje izmedju opisa brenda i sadrzaja kanala.
    Vraca skor od 0 do 100 (0 = nema poklapanja, 100 = savrseno poklapanje).

    transcripts: opciono, vidi build_channel_content_summary - ako se
    proslijedi, sadrzajni profil kanala ukljucuje i transkript-isjecke
    (Nivo 2 obogacivanje), inace se koristi samo Nivo 1 (naslovi +
    tagovi + teme).

    VAZNA ISPRAVKA KALIBRACIJE: sirova kosinusna slicnost izmedju dva
    RAZLICITA teksta (opis brenda vs. sadrzaj kanala), kod modela
    text-embedding-3-small, empirijski skoro nikad ne prelazi ~0.5-0.6
    cak ni za genuinski dobro povezane parove - nepovezani tekstovi
    obicno daju ~0.0-0.15, dobro povezani ~0.25-0.45, veoma bliski
    ~0.45-0.60. Direktno skaliranje "similarity * 100" (kao u prethodnoj
    verziji) implicitno pretpostavlja da je 1.0 realno dostizno, cime
    se SVI brand-fit skorovi sistematski potcjenjuju.

    Zato se ovdje primjenjuje min-max preskaliranje na realisticniji
    opseg [MIN_EXPECTED_SIMILARITY, MAX_EXPECTED_SIMILARITY] prije
    konacnog skaliranja na 0-100. Ove granice su procjena autora
    zasnovana na empirijskom ponasanju modela, ne iz objavljenog rada
    - treba ih dodatno kalibrisati testiranjem na vise poznatih parova
    brend-kanal (planirano poglavlje 4.3).
    
    
    """
    channel_content = build_channel_content_summary(channel_stats, videos, transcripts=transcripts)

    brand_embedding = get_embedding(brand_description)
    channel_embedding = get_embedding(channel_content)

    similarity = cosine_similarity(brand_embedding, channel_embedding)

    MIN_EXPECTED_SIMILARITY = 0.15   # empirijski izvedeno iz sopstvenog kalibracionog testa (n=4 para), poglavlje 4.3
    MAX_EXPECTED_SIMILARITY = 0.45   # empirijski izvedeno iz sopstvenog kalibracionog testa (n=4 para), poglavlje 4.3

    rescaled = (similarity - MIN_EXPECTED_SIMILARITY) / (
        MAX_EXPECTED_SIMILARITY - MIN_EXPECTED_SIMILARITY
    )
    score = round(max(0, min(1, rescaled)) * 100, 2)

    warning = None
    if len(brand_description.strip().split()) < 5:
        warning = (
            "Opis brenda je vrlo kratak (manje od 5 rijeci). Za pouzdaniji "
            "brand-fit skor, opisite brend detaljnije (kategorija proizvoda, "
            "ciljna publika, vrijednosti/ton) umjesto samo naziva brenda."
        )

    return {
        "brand_fit_score": score,
        "raw_cosine_similarity": round(similarity, 4),
        "calibration_range": [MIN_EXPECTED_SIMILARITY, MAX_EXPECTED_SIMILARITY],
        "content_profile_level": "level_2_with_transcripts" if transcripts else "level_1_metadata_only",
        "warning": warning,
    }


# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats
    from transcript_service import get_transcripts_for_videos

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)
    transcripts = get_transcripts_for_videos(video_ids, max_videos=6, max_words_per_video=50)

    brand_description = "Tech brand selling premium smartphones, laptops and consumer electronics accessories, focused on innovation and design quality."

    result = calculate_brand_fit_score(brand_description, stats, videos, transcripts=transcripts)
    print("BRAND-FIT REZULTAT (Nivo 1 + Nivo 2):")
    for key, value in result.items():
        print(f"  {key}: {value}")