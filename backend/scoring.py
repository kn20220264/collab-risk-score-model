"""Modul kvantitativnih metrika kanala (engagement, konzistentnost, subscriber/view ratio)."""

from datetime import datetime
from statistics import mean, pstdev


def _parse_date(video: dict) -> datetime:
    return datetime.strptime(video["published_at"], "%Y-%m-%dT%H:%M:%SZ")


def calculate_engagement_rate(videos: list) -> float:
    """ER = (lajkovi + komentari) / pregledi * 100, usrednjeno preko poslednjih N videa (Lopez-Navarrete et al., 2021)."""
    total_likes = sum(v["like_count"] for v in videos)
    total_comments = sum(v["comment_count"] for v in videos)
    total_views = sum(v["view_count"] for v in videos)

    if total_views == 0:
        return 0.0

    return round((total_likes + total_comments) / total_views * 100, 4)


def calculate_view_consistency_cv(videos: list) -> float:
    """Koeficijent varijacije (CV = sigma/mu * 100) broja pregleda poslednjih N videa."""
    views = [v["view_count"] for v in videos]
    if len(views) < 2:
        return 0.0

    mu = mean(views)
    if mu == 0:
        return 0.0

    sigma = pstdev(views)
    return round((sigma / mu) * 100, 2)


def calculate_posting_consistency(videos: list) -> dict:
    """Koeficijent varijacije (CV = sigma/mu) razmaka (u danima) izmedju uzastopnih objava."""
    if len(videos) < 2:
        return {"avg_gap_days": 0.0, "std_gap_days": 0.0, "posting_consistency_cv": 0.0}

    dates = sorted((_parse_date(v) for v in videos), reverse=True)
    gaps = [
        (dates[i] - dates[i + 1]).total_seconds() / 86400
        for i in range(len(dates) - 1)
    ]

    avg_gap = mean(gaps)
    std_gap = pstdev(gaps) if len(gaps) > 1 else 0.0

    cv = round((std_gap / avg_gap) * 100, 2) if avg_gap > 0 else 0.0

    return {
        "avg_gap_days": round(avg_gap, 2),
        "std_gap_days": round(std_gap, 2),
        "posting_consistency_cv": cv,
    }


def calculate_subscriber_to_view_ratio(channel_stats: dict, videos: list) -> float:
    """Odnos pretplatnici : cjelozivotni prosjecni pregledi po videu (kanal-nivo, ne uzorak od N videa)."""
    video_count = channel_stats.get("video_count", 0)
    total_views = channel_stats.get("view_count", 0)

    if video_count == 0 or total_views == 0:
        return 0.0

    lifetime_avg_views = total_views / video_count
    if lifetime_avg_views == 0:
        return float("inf")

    return round(channel_stats["subscriber_count"] / lifetime_avg_views, 2)


def calculate_quantitative_metrics(channel_stats: dict, videos: list) -> dict:
    posting = calculate_posting_consistency(videos)

    return {
        "engagement_rate": calculate_engagement_rate(videos),
        "view_consistency_cv": calculate_view_consistency_cv(videos),
        "avg_posting_gap_days": posting["avg_gap_days"],
        "std_posting_gap_days": posting["std_gap_days"],
        "posting_consistency_cv": posting["posting_consistency_cv"],
        "subscriber_to_view_ratio": calculate_subscriber_to_view_ratio(channel_stats, videos),
        "sample_size": len(videos),
    }


if __name__ == "__main__":
    from youtube_service import get_channel_stats, get_recent_video_ids, get_videos_stats

    handle = "@mkbhd"
    stats = get_channel_stats(handle)
    video_ids = get_recent_video_ids(stats["channel_id"], max_results=20)
    videos = get_videos_stats(video_ids)

    result = calculate_quantitative_metrics(stats, videos)
    print("KVANTITATIVNE METRIKE:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print(f"\n(Za provjeru: channel_stats video_count={stats['video_count']}, "
          f"view_count={stats['view_count']}, lifetime_avg_views="
          f"{stats['view_count']/stats['video_count']:.0f})")
