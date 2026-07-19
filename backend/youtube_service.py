import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)


def get_channel_id_from_handle(handle: str) -> str:
    """
    Pronalazi channel ID na osnovu @handle-a kanala (npr. '@mkbhd').
    """
    request = youtube.channels().list(
        part="id",
        forHandle=handle
    )
    response = request.execute()

    if not response.get("items"):
        raise ValueError(f"Kanal sa handle-om '{handle}' nije pronađen.")

    return response["items"][0]["id"]


def get_channel_stats(handle: str) -> dict:
    """
    Vraća osnovne statistike kanala na osnovu @handle-a.
    """
    channel_id = get_channel_id_from_handle(handle)

    request = youtube.channels().list(
        part="snippet,statistics",
        id=channel_id
    )
    response = request.execute()
    item = response["items"][0]

    return {
        "channel_id": channel_id,
        "title": item["snippet"]["title"],
        "description": item["snippet"]["description"],
        "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
        "video_count": int(item["statistics"].get("videoCount", 0)),
        "view_count": int(item["statistics"].get("viewCount", 0)),
    }



def get_channel_uploads_playlist_id(channel_id: str) -> str:
    """
    Vraća ID plejliste koja sadrži sve uploade kanala.
    """
    request = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    )
    response = request.execute()
    return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_recent_video_ids(channel_id: str, max_results: int = 20) -> list:
    """
    Vraća listu ID-jeva posljednjih N videa sa kanala.
    """
    uploads_playlist_id = get_channel_uploads_playlist_id(channel_id)

    request = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=uploads_playlist_id,
        maxResults=max_results
    )
    response = request.execute()

    return [item["contentDetails"]["videoId"] for item in response["items"]]


def get_videos_stats(video_ids: list) -> list:
    """
    Vraća detaljne statistike (pregledi, lajkovi, komentari, datum) za listu video ID-jeva.
    YouTube API dozvoljava do 50 ID-jeva po pozivu.
    """
    request = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        stats = item["statistics"]
        videos.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
        })
    return videos


def get_video_comments(video_id: str, max_results: int = 50) -> list:
    """
    Vraca listu komentara (tekst) za dati video.
    Ako su komentari onemoguceni na videu, vraca praznu listu.
    """
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance",
            textFormat="plainText"
        )
        response = request.execute()

        for item in response["items"]:
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            like_count = item["snippet"]["topLevelComment"]["snippet"]["likeCount"]
            comments.append({
                "text": comment_text,
                "like_count": like_count
            })

    except Exception as e:
        # Komentari mogu biti onemoguceni na videu, ili druga greska - ne prekidamo ceo proces
        print(f"  (Napomena: komentari nisu dostupni za video {video_id}: {e})")

    return comments


def get_comments_for_videos(video_ids: list, max_per_video: int = 30) -> list:
    """
    Prikuplja komentare za listu videa i vraca ih kao jednu spojenu listu.
    """
    all_comments = []
    for video_id in video_ids:
        comments = get_video_comments(video_id, max_results=max_per_video)
        all_comments.extend(comments)
    return all_comments




# Brzi test — pokreni ovaj fajl direktno da provjeriš da radi
if __name__ == "__main__":
    test_handle = "@mkbhd"
    stats = get_channel_stats(test_handle)
    print("KANAL:", stats)

    video_ids = get_recent_video_ids(stats["channel_id"], max_results=5)
    print("VIDEO ID-JEVI:", video_ids)

    videos = get_videos_stats(video_ids)
    print("STATISTIKE VIDEA:")
    for v in videos:
        print(v)

    comments = get_comments_for_videos(video_ids, max_per_video=10)
    print(f"\nPRIKUPLJENO KOMENTARA: {len(comments)}")
    for c in comments[:5]:
        print(f"  - ({c['like_count']} lajkova) {c['text'][:80]}")