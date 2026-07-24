"""
Modul za dodatnu kvalitativnu analizu sadrzaja kanala (Creator Persona /
Profanity / Brand Partners), po uzoru na dodatne "agente" komercijalnih
alata poput CreatorScore.

Za razliku od kvantitativnih modula (scoring.py, brand_fit.py), ove tri
kategorije se ne mogu pouzdano izracunati deterministicki iz metapodataka
- zahtijevaju razumijevanje PRIRODNOG JEZIKA (o cemu kreator prica, da li
koristi psovke, koje brendove pominje i u kom kontekstu). Zato se, kao i
kod explanation_service.py i brand_research_service.py, koristi Claude API
- jedan strukturiran poziv koji vraca JSON, umjesto tri odvojena poziva
(brzina/cijena).

Ulaz je isti sadrzajni profil kanala koji koristi i brand_fit.py (naslovi
videa, tagovi, transkript-isjecci) - vidi build_channel_content_summary
u brand_fit.py za teorijsko obrazlozenje izbora ovih izvora.

VAZNO OGRANICENJE (transparentno navedeno, ne skriveno): profanity
analiza je ogranicena na video-e za koje je transkript dostupan (vidi
transcript_service.py - podrazumijevano do 6 najnovijih), jer se psovke
mogu pouzdano detektovati samo iz stvarnog izgovorenog teksta, ne iz
naslova/tagova. "posts_analyzed" u odgovoru eksplicitno odrazava taj
manji uzorak, a ne ukupan broj videa kanala.
"""

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

MODEL = "claude-haiku-4-5-20251001"

_EMPTY_RESULT = {
    "creator_persona": {
        "primary_niche": "Nepoznato",
        "secondary_niches": [],
        "content_style": [],
        "confidence": 0.0,
        "summary": "Nedovoljno sadrzaja (naslova/transkripata) za procjenu persone kreatora.",
        "context_for_risk": "",
    },
    "profanity_analysis": {
        "level": "Nepoznato",
        "posts_analyzed": 0,
        "posts_with_profanity": 0,
        "total_instances": 0,
        "profanity_rate_pct": 0.0,
        "summary": "Nema dostupnih transkripata za analizu jezika.",
    },
    "brand_partners": {
        "total_brands": 0,
        "organic_count": 0,
        "sponsored_count": 0,
        "top_brands": [],
        "categories": [],
    },
}


def _build_content_block(channel_stats: dict, videos: list, transcripts: dict) -> str:
    lines = [f"Opis kanala: {channel_stats.get('description', '')[:400]}"]

    for v in videos:
        line = f"- Video: \"{v['title']}\""
        if v.get("tags"):
            line += f" | Tagovi: {', '.join(v['tags'][:8])}"
        transcript = transcripts.get(v["video_id"])
        if transcript:
            line += f" | Transkript (isjecak): {transcript}"
        lines.append(line)

    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """
    Claude ponekad omota JSON u markdown code fence ili doda kratak uvod
    uprkos instrukciji da odgovori samo sa JSON-om - ovo robustno izvlaci
    prvi validan JSON objekat iz odgovora.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Odgovor ne sadrzi JSON objekat.")
    return json.loads(text[start : end + 1])


def analyze_creator_content(channel_stats: dict, videos: list, transcripts: dict) -> dict:
    """
    Vraca dict sa tri kljuca: creator_persona, profanity_analysis,
    brand_partners - vidi main.py za nacin koriscenja u API odgovoru.
    """
    if not videos:
        return _EMPTY_RESULT

    posts_with_transcript = sum(1 for v in videos if v["video_id"] in transcripts)
    content_block = _build_content_block(channel_stats, videos, transcripts)

    prompt = f"""Analiziras sadrzaj YouTube kanala na osnovu opisa kanala, naslova videa,
tagova i kratkih transkript-isjecaka. Odgovori ISKLJUCIVO validnim JSON objektom,
bez ikakvog dodatnog teksta prije ili poslije, tacno u ovom formatu:

{{
  "creator_persona": {{
    "primary_niche": "jedna rijec/kratka fraza, npr. Lifestyle, Gaming, Tech, Fitness, Beauty, Finance, Comedy, Education",
    "secondary_niches": ["do 2 dodatne teme ako postoje, inace prazna lista"],
    "content_style": ["do 2 stila, biraj izmedju: Personal, Promotional, Educational, Entertainment, Reviews"],
    "confidence": 0.0 do 1.0,
    "summary": "1-2 recenice na srpskom/crnogorskom jeziku koje opisuju o cemu kreator pravi sadrzaj",
    "context_for_risk": "1-2 recenice na srpskom/crnogorskom - da li nesto u sadrzaju/tonu zasluzuje paznju brend menadzera prije saradnje (ili navedi da nema posebnih napomena)"
  }},
  "profanity": {{
    "level": "None, Low, Moderate ili High - procjena na osnovu DOSTUPNIH transkript-isjecaka",
    "posts_with_profanity": broj videa (od onih sa transkriptom) gdje se pojavljuje eksplicitan/vulgaran jezik,
    "total_instances": procijenjen ukupan broj pojava psovki/vulgarizama u dostupnim transkriptima,
    "summary": "1 recenica na srpskom/crnogorskom jeziku"
  }},
  "brand_partners": {{
    "brands": [
      {{"name": "naziv brenda", "mentions": broj pominjanja, "type": "organic ili sponsored - sponsored SAMO ako tekst eksplicitno pominje sponzorstvo/reklamu/paid partnership/ad, inace organic", "category": "kratka kategorija brenda, npr. Fashion, Tech, Food, Beauty"}}
    ],
    "categories": [{{"name": "kategorija", "count": broj razlicitih brendova u toj kategoriji}}]
  }}
}}

Ako u sadrzaju nema pominjanja brendova, "brands" neka bude prazna lista.
Ne izmisljaj brendove koji nisu eksplicitno pomenuti u tekstu ispod.

Sadrzaj kanala (broj videa sa transkriptom: {posts_with_transcript}/{len(videos)}):
{content_block}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text

    try:
        parsed = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError):
        return _EMPTY_RESULT

    persona = parsed.get("creator_persona", {})
    profanity = parsed.get("profanity", {})
    brands_data = parsed.get("brand_partners", {})

    posts_with_profanity = profanity.get("posts_with_profanity", 0)
    profanity_rate_pct = (
        round(posts_with_profanity / posts_with_transcript * 100, 1)
        if posts_with_transcript > 0
        else 0.0
    )

    brands = brands_data.get("brands", [])
    organic_count = sum(1 for b in brands if b.get("type") != "sponsored")
    sponsored_count = sum(1 for b in brands if b.get("type") == "sponsored")
    top_brands = sorted(brands, key=lambda b: b.get("mentions", 0), reverse=True)[:10]

    return {
        "creator_persona": {
            "primary_niche": persona.get("primary_niche", "Nepoznato"),
            "secondary_niches": persona.get("secondary_niches", []),
            "content_style": persona.get("content_style", []),
            "confidence": persona.get("confidence", 0.0),
            "summary": persona.get("summary", ""),
            "context_for_risk": persona.get("context_for_risk", ""),
        },
        "profanity_analysis": {
            "level": profanity.get("level", "Nepoznato"),
            "posts_analyzed": posts_with_transcript,
            "posts_with_profanity": posts_with_profanity,
            "total_instances": profanity.get("total_instances", 0),
            "profanity_rate_pct": profanity_rate_pct,
            "summary": profanity.get("summary", ""),
        },
        "brand_partners": {
            "total_brands": len(brands),
            "organic_count": organic_count,
            "sponsored_count": sponsored_count,
            "top_brands": top_brands,
            "categories": brands_data.get("categories", []),
        },
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

    result = analyze_creator_content(stats, videos, transcripts)
    print(json.dumps(result, indent=2, ensure_ascii=False))
