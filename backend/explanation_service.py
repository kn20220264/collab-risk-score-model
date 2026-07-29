import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Haiku je brz i jeftin model, sasvim dovoljan za generisanje kratkog,
# strukturiranog izvjestaja na osnovu vec izracunatih metrika.
MODEL = "claude-haiku-4-5-20251001"


def generate_risk_explanation(channel_name: str, brand_description: str, risk_result: dict) -> str:
    """
    Generise citljivo tekstualno obrazlozenje risk skora na osnovu
    vec izracunatih metrika iz svih modula.
    """
    prompt = f"""Ti si analiticar za procjenu rizika poslovne saradnje sa YouTube kreatorima.

Kanal: {channel_name}
Opis brenda koji razmatra saradnju: {brand_description}

Rezultati analize:
- Skor kvantitativnih metrika (engagement, konzistentnost): {risk_result['module_scores']['quantitative']}/100
- Skor autenticnosti (odnos pretplatnika/pregleda): {risk_result['module_scores']['authenticity']}/100
- Skor sentimenta publike: {risk_result['module_scores']['sentiment']}/100
- Skor poklapanja sa brendom (brand-fit): {risk_result['module_scores']['brand_fit']}/100
- Finalni skor: {risk_result['final_score']}/100
- Kategorija rizika: {risk_result['risk_category']}
- Aktivirani kriticni nalazi: {risk_result['triggered_caps'] if risk_result['triggered_caps'] else "nema"}

Napisi kratko, jasno obrazlozenje (3-5 recenica) za brend menadzera koji odlucuje
da li da sarađuje sa ovim kreatorom. Fokusiraj se na najvaznije nalaze i konkretnu preporuku.
Piši na srpskom/crnogorskom jeziku, poslovnim ali razumljivim tonom.
Koristi iskljucivo srpska/crnogorska slova i rijeci - bez mijesanja sa drugim jezicima ili pismima.
NE koristi Markdown formatiranje - bez #, ##, **, -, ni ikakvih drugih specijalnih znakova za naslove/bold/liste. Piši iskljucivo cist tekst, organizovan u pasuse."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats, get_comments_for_videos
    from scoring import calculate_quantitative_metrics
    from ai_service import analyze_comments_batch
    from brand_fit import calculate_brand_fit_score
    from risk_aggregation import calculate_final_risk_score

    print("Prikupljam podatke i racunam skorove...")

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)
    comments = get_comments_for_videos(video_ids, max_per_video=10)

    quant_metrics = calculate_quantitative_metrics(stats, videos)
    sentiment_result = analyze_comments_batch(comments)

    brand_description = "Tech brand selling premium smartphones, laptops and consumer electronics accessories, focused on innovation and design quality."
    brand_fit_result = calculate_brand_fit_score(brand_description, stats, videos)

    final_result = calculate_final_risk_score(quant_metrics, sentiment_result, brand_fit_result)

    print("Generisem AI obrazlozenje...")
    explanation = generate_risk_explanation(stats["title"], brand_description, final_result)

    print("\n=== AI OBRAZLOZENJE ===")
    print(explanation)