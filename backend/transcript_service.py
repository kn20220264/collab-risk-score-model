"""
Modul za povlacenje transkripata videa (Nivo 2 obogacivanja sadrzajnog
profila kanala).

Motivacija i teorijsko uporiste: "Developing a Multimodal Approach to
Channel Characterization on YouTube" eksplicitno koristi transkripte
(uz metapodatke) kao dio "multimodal" sadrzajnog profila kanala za
semanticku analizu - princip koji ovaj modul operacionalizuje u manjem
obimu. Takodje se nadovezuje na Bleckmann & Tschisgale (2025), koji
navode da je similarity analiza BEZ okolnog konteksta donja granica
(lower-bound) sposobnosti embedding modela, i da bogatiji kontekst
treba da poboljsa performanse - transkript je upravo taj "bogatiji
kontekst" za sadrzaj videa, u odnosu na sam naslov.

Obim (namjerna dizajn odluka, ne tehnicko ogranicenje): transkripti se
povlace samo za N NAJNOVIJIH videa (podrazumijevano 6), i samo prvih
~50 rijeci po videu (otprilike prvih 15-20 sekundi govora, gdje
kreatori obicno najave temu videa). Ovo je namjeran balans izmedju
bogatijeg konteksta i vremena izvrsavanja.

Biblioteka: youtube-transcript-api (neslužbena, ali sirok koriscena
Python biblioteka koja povlaci javno dostupne YouTube titlove -
automatski generisane ili rucno dodane).
"""

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_ytt_api = YouTubeTranscriptApi()


def get_video_transcript_snippet(video_id: str, max_words: int = 50) -> str:
    """
    Vraca prvih `max_words` rijeci transkripta datog videa.
    Pokusava prvo engleski, pa srpski/hrvatski/bosanski kao fallback
    (jezici mogu varirati zavisno od kanala).

    Ako transkript nije dostupan (iskljucen od strane kreatora, video
    nedostupan, ili ne postoji ni na jednom pokusanom jeziku), vraca
    prazan string - ne prekida ostatak procesa (isti obrazac kao
    get_video_comments u youtube_service.py).
    """
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
    """
    Povlaci kratke transkript-isjecke za do `max_videos` NAJNOVIJIH
    videa iz liste (pretpostavlja se da je video_ids vec sortirana
    od najnovijeg ka najstarijem, sto je slucaj sa
    get_recent_video_ids u youtube_service.py).

    Vraca dict {video_id: transcript_snippet}. Video-i bez dostupnog
    transkripta se izostavljaju iz rezultata.
    """
    transcripts = {}
    for video_id in video_ids[:max_videos]:
        snippet = get_video_transcript_snippet(video_id, max_words=max_words_per_video)
        if snippet:
            transcripts[video_id] = snippet

    return transcripts


# Brzi test
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