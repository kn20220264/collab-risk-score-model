import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> list:
    """
    Vraca embedding vektor za dati tekst.
    """
    text = text.replace("\n", " ").strip()
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Racuna kosinusnu slicnost izmedju dva vektora (opseg -1 do 1).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_channel_content_summary(channel_stats: dict, videos: list) -> str:
    """
    Spaja opis kanala i naslove videa u jedan tekst koji predstavlja
    'sadrzajni profil' kanala, spreman za embedding.
    """
    parts = [channel_stats.get("description", "")]
    parts += [v["title"] for v in videos]
    return " | ".join(p for p in parts if p)


def calculate_brand_fit_score(brand_description: str, channel_stats: dict, videos: list) -> dict:
    """
    Racuna semanticko poklapanje izmedju opisa brenda i sadrzaja kanala.
    Vraca skor od 0 do 100 (0 = nema poklapanja, 100 = savrseno poklapanje).
    """
    channel_content = build_channel_content_summary(channel_stats, videos)

    brand_embedding = get_embedding(brand_description)
    channel_embedding = get_embedding(channel_content)

    similarity = cosine_similarity(brand_embedding, channel_embedding)

    # Cosine similarity je obicno u opsegu ~0.0-1.0 za srodne tekstove kod ovog modela,
    # pa skaliramo na 0-100 radi lakseg tumacenja u finalnom skoru.
    score = round(max(0, min(1, similarity)) * 100, 2)

    return {
        "brand_fit_score": score,
        "raw_cosine_similarity": round(similarity, 4),
    }


# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)

    # Primjer opisa brenda - tehnoloski brend, trebalo bi da ima visok fit sa MKBHD
    brand_description = "Tech brand selling premium smartphones, laptops and consumer electronics accessories, focused on innovation and design quality."

    result = calculate_brand_fit_score(brand_description, stats, videos)
    print("BRAND-FIT REZULTAT:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n--- Poredjenje sa nepovezanim brendom ---")
    unrelated_brand = "Organic baby food and toddler nutrition products, focused on natural ingredients and family wellness."
    result_unrelated = calculate_brand_fit_score(unrelated_brand, stats, videos)
    print("BRAND-FIT REZULTAT (nepovezan brend):")
    for key, value in result_unrelated.items():
        print(f"  {key}: {value}")
        

        
        