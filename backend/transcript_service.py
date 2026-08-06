"""Povlacenje kratkih transkript-isjecaka za najnovije videe kanala (youtube-transcript-api)."""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_ytt_api = YouTubeTranscriptApi()


def get_video_transcript_snippet(video_id: str, max_words: int = 50) -> str:
    """Vraca prvih `max_words` rijeci transkripta (en/sr/hr/bs fallback); prazan string ako nije dostupan."""
    language_attempts = [
        ["en"], ["sr"], ["hr"], ["bs"], ["en-US"], ["en-GB"],
    ]

    for languages in language_attempts:
        try:
            fetched = _ytt_api.fetch(video_id, languages=languages)
            full_text = " ".join(snippet.text for snippet in fetched)
            words = full_text.split()
            return " ".join(words[:max_words])
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
            continue
        except Exception:
            continue

    return ""


def get_transcripts_for_videos(video_ids: list, max_videos: int = 6, max_words_per_video: int = 50) -> dict:
    """Vraca {video_id: transcript_snippet} za prvih `max_videos` iz liste; videi bez transkripta se izostavljaju."""
    transcripts = {}
    for video_id in video_ids[:max_videos]:
        snippet = get_video_transcript_snippet(video_id, max_words=max_words_per_video)
        if snippet:
            transcripts[video_id] = snippet

    return transcripts


if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)

    transcripts = get_transcripts_for_videos(video_ids, max_videos=6, max_words_per_video=50)

    print(f"=== TRANSKRIPTI ({len(transcripts)}/{min(6, len(video_ids))} videa uspjesno) ===")
    for video_id, snippet in transcripts.items():
        print(f"\n[{video_id}]")
        print(f"  {snippet}")