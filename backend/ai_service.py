from transformers import pipeline

# Model podrzava vise jezika, ukljucujuci engleski - dobar izbor za YouTube komentare
# koji su cesto mesavina jezika. Model se ucitava jednom pri pokretanju.
_sentiment_pipeline = None


def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    return _sentiment_pipeline


def analyze_comment_sentiment(text: str) -> dict:
    """
    Model vraca ocjenu od 1 do 5 zvjezdica (1 = vrlo negativno, 5 = vrlo pozitivno).
    Mapiramo to na pozitivno/neutralno/negativno.
    """
    pipeline_model = get_sentiment_pipeline()

    # Model ima limit od 512 karaktera po ulazu
    result = pipeline_model(text[:512])[0]
    stars = int(result["label"][0])

    if stars <= 2:
        category = "negative"
    elif stars == 3:
        category = "neutral"
    else:
        category = "positive"

    return {"stars": stars, "category": category, "confidence": round(result["score"], 3)}


def analyze_comments_batch(comments: list) -> dict:
    """
    Analizira listu komentara i vraca agregirane rezultate.
    comments: lista dict-ova sa kljucem 'text'
    """
    results = []
    for comment in comments:
        sentiment = analyze_comment_sentiment(comment["text"])
        results.append(sentiment)

    total = len(results)
    if total == 0:
        return {"positive_pct": 0, "neutral_pct": 0, "negative_pct": 0, "sentiment_score": 0}

    positive = sum(1 for r in results if r["category"] == "positive")
    neutral = sum(1 for r in results if r["category"] == "neutral")
    negative = sum(1 for r in results if r["category"] == "negative")

    positive_pct = round(positive / total * 100, 2)
    neutral_pct = round(neutral / total * 100, 2)
    negative_pct = round(negative / total * 100, 2)

    # Sentiment skor: pozitivno% - negativno%, opseg -100 do +100
    sentiment_score = round(positive_pct - negative_pct, 2)

    return {
        "positive_pct": positive_pct,
        "neutral_pct": neutral_pct,
        "negative_pct": negative_pct,
        "sentiment_score": sentiment_score,
        "sample_size": total,
    }


# Brzi test
if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_comments_for_videos

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=5)
    comments = get_comments_for_videos(video_ids, max_per_video=10)

    print(f"Analiziram {len(comments)} komentara...")
    result = analyze_comments_batch(comments)
    print("REZULTAT SENTIMENT ANALIZE:")
    for key, value in result.items():
        print(f"  {key}: {value}")