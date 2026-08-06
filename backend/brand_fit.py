"""Brand-fit modul: semanticko poklapanje opisa brenda i sadrzaja kanala preko embeddinga."""

import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"

# Kalibracioni opseg (5. percentil ukrstenih / 95. percentil stvarnih parova)
# - vidi scripts/calibrate_brand_fit.py i backend/reference_brand_fit.json.
MIN_EXPECTED_SIMILARITY = 0.0893
MAX_EXPECTED_SIMILARITY = 0.3767

_EMBEDDING_MAX_RETRIES = 3
_EMBEDDING_RETRY_BACKOFF_SECONDS = 2


def get_embedding(text: str) -> list:
    """Vraca embedding vektor za dati tekst, sa retry na tranzijentne API greske."""
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
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_channel_content_summary(channel_stats: dict, videos: list, transcripts: dict = None) -> str:
    """Spaja opis kanala, naslove/opise/tagove/teme videa i (opciono) transkript-isjecke u tekst za embedding."""
    parts = [channel_stats.get("description", "")]

    for v in videos:
        parts.append(v["title"])
        if v.get("description"):
            desc_words = v["description"].split()[:100]
            parts.append(" ".join(desc_words))
        if v.get("tags"):
            parts.append(", ".join(v["tags"]))
        if v.get("topic_categories"):
            parts.append(", ".join(v["topic_categories"]))
        if transcripts and v.get("video_id") in transcripts:
            parts.append(transcripts[v["video_id"]])

    return " | ".join(p for p in parts if p)


def calculate_brand_fit_score(brand_description: str, channel_stats: dict, videos: list, transcripts: dict = None) -> dict:
    """Racuna semanticko poklapanje brenda i sadrzaja kanala (0-100) preko min-max preskalirane kosinusne slicnosti."""
    channel_content = build_channel_content_summary(channel_stats, videos, transcripts=transcripts)

    brand_embedding = get_embedding(brand_description)
    channel_embedding = get_embedding(channel_content)

    similarity = cosine_similarity(brand_embedding, channel_embedding)

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
