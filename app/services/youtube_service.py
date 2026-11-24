import os
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

from app.utils.youtube_utils import extract_video_id, is_valid_youtube_url


load_dotenv()


@dataclass
class Comment:
    comment_id: str
    video_id: str
    text: str
    likes: int
    published_at: datetime
    author: str


class YouTubeServiceError(Exception):
    """Custom exception for YouTube service errors."""
    pass


class YouTubeService:
    """Service for fetching YouTube comments using YouTube Data API v3."""

    DEFAULT_MAX_COMMENTS = 100
    DEFAULT_MAX_RESULTS_PER_PAGE = 100

    def __init__(self, api_key: Optional[str] = None, max_comments: int = DEFAULT_MAX_COMMENTS):
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self.max_comments = max_comments
        self._youtube = None

    def _get_client(self):
        """Lazy initialization of YouTube API client."""
        if self._youtube is None:
            if not self.api_key or self.api_key == "YOUR_YOUTUBE_API_KEY_HERE":
                raise YouTubeServiceError(
                    "YouTube API key not configured. Please set YOUTUBE_API_KEY in .env file."
                )
            self._youtube = build("youtube", "v3", developerKey=self.api_key)
        return self._youtube

    def fetch_comments(self, video_url: str) -> pd.DataFrame:
        """
        Fetch top-level comments from a YouTube video.

        Args:
            video_url: YouTube video URL

        Returns:
            DataFrame with columns: comment_id, video_id, text, likes, published_at, author

        Raises:
            YouTubeServiceError: For invalid URLs, API errors, or unavailable videos
        """
        if not is_valid_youtube_url(video_url):
            raise YouTubeServiceError(f"Invalid YouTube URL: {video_url}")

        video_id = extract_video_id(video_url)
        if not video_id:
            raise YouTubeServiceError(f"Could not extract video ID from URL: {video_url}")

        return self._fetch_comments_by_video_id(video_id)

    def _fetch_comments_by_video_id(self, video_id: str) -> pd.DataFrame:
        """Fetch comments using video ID with pagination."""
        youtube = self._get_client()
        comments = []
        next_page_token = None
        total_fetched = 0

        try:
            while total_fetched < self.max_comments:
                max_results = min(
                    self.DEFAULT_MAX_RESULTS_PER_PAGE,
                    self.max_comments - total_fetched
                )

                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=max_results,
                    pageToken=next_page_token,
                    order="relevance",
                    textFormat="plainText"
                )

                response = request.execute()

                if not response.get("items"):
                    break

                for item in response["items"]:
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment = Comment(
                        comment_id=item["snippet"]["topLevelComment"]["id"],
                        video_id=video_id,
                        text=snippet["textDisplay"],
                        likes=snippet.get("likeCount", 0),
                        published_at=datetime.fromisoformat(
                            snippet["publishedAt"].replace("Z", "+00:00")
                        ),
                        author=snippet["authorDisplayName"]
                    )
                    comments.append(comment)

                total_fetched += len(response["items"])
                next_page_token = response.get("nextPageToken")

                if not next_page_token:
                    break

        except HttpError as e:
            error_details = self._parse_http_error(e)
            raise YouTubeServiceError(error_details) from e

        if not comments:
            return pd.DataFrame(columns=[
                "comment_id", "video_id", "text", "likes", "published_at", "author"
            ])

        df = pd.DataFrame([vars(c) for c in comments])
        return df

    def _parse_http_error(self, error: HttpError) -> str:
        """Parse HTTP error and return user-friendly message."""
        status = error.resp.status

        if status == 400:
            return "Invalid request. Please check the video URL."
        elif status == 401:
            return "Invalid API key. Please check your YouTube API key in .env file."
        elif status == 403:
            error_reason = ""
            try:
                error_content = error.content.decode("utf-8")
                if "quotaExceeded" in error_content:
                    return "API quota exceeded. Please try again later or request a quota increase."
                if "commentsDisabled" in error_content:
                    return "Comments are disabled for this video."
                if "videoNotFound" in error_content or "videoUnavailable" in error_content:
                    return "Video not found or unavailable."
            except Exception:
                pass
            return "Access forbidden. Check API key permissions or video availability."
        elif status == 404:
            return "Video not found."
        elif status >= 500:
            return "YouTube API server error. Please try again later."
        else:
            return f"API error (HTTP {status}): {error}"


def fetch_video_comments(video_url: str, max_comments: int = YouTubeService.DEFAULT_MAX_COMMENTS) -> pd.DataFrame:
    """
    Convenience function to fetch comments from a video URL.

    Args:
        video_url: YouTube video URL
        max_comments: Maximum number of comments to fetch

    Returns:
        DataFrame with comment data
    """
    service = YouTubeService(max_comments=max_comments)
    return service.fetch_comments(video_url)