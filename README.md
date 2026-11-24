# YouTube Comment Sentiment Analyzer

A Streamlit web application that analyzes sentiment of YouTube video comments and provides analytics.

## Features

- Fetch comments from YouTube videos using YouTube Data API v3
- Sentiment analysis using NLP models
- 1-5 star rating derived from sentiment
- Visual analytics and statistics

## Setup

### Prerequisites

- Python 3.8+
- YouTube Data API v3 key

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env` file from `.env.example`:
   ```bash
   copy .env.example .env  # Windows
   # cp .env.example .env  # Linux/Mac
   ```
5. Add your YouTube API key to `.env`:
   ```
   YOUTUBE_API_KEY=your_actual_api_key_here
   ```

### Running the Application

```bash
streamlit run app/main.py
```

## Project Structure

```
youtube-sentiment-analyzer/
├── app/
│   ├── services/
│   │   ├── youtube_service.py
│   │   ├── sentiment_service.py
│   │   └── similarity_service.py
│   ├── utils/
│   │   ├── youtube_utils.py
│   │   └── text_preprocessing.py
│   └── main.py
├── tests/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Getting a YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable "YouTube Data API v3"
4. Create credentials (API Key)
5. Add the API key to your `.env` file

## Current Phase

Phase 1: YouTube comment collection module implemented.
- Video ID extraction from URLs
- Comment fetching with pagination
- Error handling for API issues