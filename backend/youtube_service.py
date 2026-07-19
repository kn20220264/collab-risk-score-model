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


# Brzi test — pokreni ovaj fajl direktno da provjeriš da radi
if __name__ == "__main__":
    test_handle = "@mkbhd"  # možeš zamijeniti bilo kojim YouTube handle-om
    stats = get_channel_stats(test_handle)
    print(stats)