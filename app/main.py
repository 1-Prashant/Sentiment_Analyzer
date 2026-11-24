import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from app.services.youtube_service import YouTubeService, fetch_video_comments, YouTubeServiceError
from app.services.sentiment_service import SentimentService, create_sentiment_service, SentimentResult
from app.services.similarity_service import SimilarityService, create_similarity_service, SimilarityResult
from app.utils.text_preprocessing import preprocess_comment, is_meaningful_comment
from app.utils.youtube_utils import extract_video_id, is_valid_youtube_url


st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_sentiment_service():
    return create_sentiment_service()


@st.cache_resource
def get_similarity_service(_sentiment_service=None):
    if _sentiment_service is None:
        _sentiment_service = get_sentiment_service()
    return create_similarity_service(embedding_model=_sentiment_service.embedding_model)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_comments_cached(video_url: str, max_comments: int) -> pd.DataFrame:
    """Cached comment fetching."""
    return fetch_video_comments(video_url, max_comments=max_comments)


@st.cache_data(show_spinner=False)
def analyze_sentiments_cached(texts: tuple) -> list:
    """Cached sentiment analysis."""
    service = get_sentiment_service()
    results = service.analyze_batch(list(texts))
    return [
        {
            "label": r.label,
            "score": r.score,
            "stars": r.stars,
            "all_scores": r.all_scores
        }
        for r in results
    ]


@st.cache_data(show_spinner=False)
def find_similar_opinion_cached(texts: tuple, min_similarity: float) -> dict:
    """Cached similarity analysis."""
    sentiment_service = get_sentiment_service()
    service = get_similarity_service(_sentiment_service=sentiment_service)
    service.min_similarity = min_similarity
    result = service.find_most_repeated_opinion(list(texts))
    return {
        "representative_index": result.representative_index,
        "similar_indices": result.similar_indices,
        "representative_text": result.representative_text,
        "similar_texts": result.similar_texts,
        "average_similarity": result.average_similarity,
        "cluster_size": result.cluster_size
    }


def display_star_rating(stars: int, label: str = "") -> str:
    """Generate star rating display."""
    filled = "★" * stars
    empty = "☆" * (5 - stars)
    return f"{filled}{empty} {label}"


def main():
    st.title("📊 YouTube Comment Sentiment Analyzer")
    st.caption("Analyze sentiment of YouTube comments and visualize audience reactions")

    with st.sidebar:
        st.header("⚙️ Settings")
        max_comments = st.slider(
            "Max comments to fetch",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="More comments = better analysis but slower"
        )

        remove_spam = st.checkbox(
            "Remove spam patterns",
            value=True,
            help="Filter common spam phrases (subscribe, check my channel, etc.)"
        )

        min_similarity = st.slider(
            "Similarity threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Threshold for grouping similar opinions"
        )

        st.divider()
        st.caption("💡 Add your YouTube API key to `.env` file")

    col1, col2 = st.columns([3, 1])

    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL (watch, youtu.be, shorts, embed)"
        )

    with col2:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if not analyze_btn:
        st.info("👆 Enter a YouTube video URL and click **Analyze** to begin")
        return

    if not video_url or not is_valid_youtube_url(video_url):
        st.error("❌ Please enter a valid YouTube video URL")
        return

    video_id = extract_video_id(video_url)
    st.caption(f"Video ID: `{video_id}`")

    with st.spinner("Fetching comments from YouTube..."):
        try:
            comments_df = fetch_comments_cached(video_url, max_comments)
        except YouTubeServiceError as e:
            st.error(f"❌ {e}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return

    if comments_df.empty:
        st.warning("⚠️ No comments found for this video")
        return

    st.success(f"✅ Fetched **{len(comments_df)}** comments")

    with st.spinner("Preprocessing comments..."):
        comments_df["clean_text"] = comments_df["text"].apply(
            lambda x: preprocess_comment(x, remove_spam=remove_spam)
        )
        comments_df["meaningful"] = comments_df["clean_text"].apply(is_meaningful_comment)

    meaningful_df = comments_df[comments_df["meaningful"]].copy()

    if meaningful_df.empty:
        st.warning("⚠️ No meaningful comments found after preprocessing")
        return

    st.info(f"📝 Analyzing **{len(meaningful_df)}** meaningful comments...")

    with st.spinner("Running sentiment analysis..."):
        sentiment_results = analyze_sentiments_cached(tuple(meaningful_df["clean_text"].tolist()))

    sentiment_df = pd.DataFrame(sentiment_results)
    meaningful_df = pd.concat([meaningful_df.reset_index(drop=True), sentiment_df], axis=1)

    st.header("📈 Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⭐ Star Distribution", "💬 Comments", "🔍 Similar Opinions"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        pos_count = (meaningful_df["label"] == "Positive").sum()
        neu_count = (meaningful_df["label"] == "Neutral").sum()
        neg_count = (meaningful_df["label"] == "Negative").sum()
        total = len(meaningful_df)

        with col1:
            st.metric("😊 Positive", f"{pos_count}", f"{pos_count/total*100:.1f}%")
        with col2:
            st.metric("😐 Neutral", f"{neu_count}", f"{neu_count/total*100:.1f}%")
        with col3:
            st.metric("☹️ Negative", f"{neg_count}", f"{neg_count/total*100:.1f}%")
        with col4:
            avg_stars = meaningful_df["stars"].mean()
            st.metric("⭐ Avg Rating", f"{avg_stars:.1f}/5.0")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            fig_pie = px.pie(
                values=[pos_count, neu_count, neg_count],
                names=["Positive", "Neutral", "Negative"],
                title="Sentiment Distribution",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Neutral": "#f39c12",
                    "Negative": "#e74c3c"
                },
                hole=0.4
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            fig_conf = px.histogram(
                meaningful_df,
                x="score",
                color="label",
                nbins=20,
                title="Confidence Score Distribution",
                color_discrete_map={
                    "Positive": "#2ecc71",
                    "Neutral": "#f39c12",
                    "Negative": "#e74c3c"
                }
            )
            st.plotly_chart(fig_conf, use_container_width=True)

    with tab2:
        star_counts = meaningful_df["stars"].value_counts().sort_index()

        fig_stars = px.bar(
            x=star_counts.index.astype(str),
            y=star_counts.values,
            title="1–5 Star Rating Distribution (Derived from Sentiment)",
            labels={"x": "Star Rating", "y": "Number of Comments"},
            color=star_counts.index,
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"]
        )
        fig_stars.update_layout(showlegend=False)
        st.plotly_chart(fig_stars, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⭐ Rating Breakdown")
            for stars in range(5, 0, -1):
                count = star_counts.get(stars, 0)
                pct = count / total * 100 if total > 0 else 0
                stars_display = display_star_rating(stars)
                st.write(f"{stars_display} — **{count}** comments ({pct:.1f}%)")

        with col2:
            st.subheader("📊 Statistics")
            st.write(f"**Mean:** {meaningful_df['stars'].mean():.2f}")
            st.write(f"**Median:** {meaningful_df['stars'].median():.1f}")
            st.write(f"**Std Dev:** {meaningful_df['stars'].std():.2f}")
            st.write(f"**Total Comments:** {total}")

    with tab3:
        st.subheader("Most Liked Comment")

        if not meaningful_df.empty:
            top_comment = meaningful_df.loc[meaningful_df["likes"].idxmax()]
            with st.container(border=True):
                st.markdown(f"**{top_comment['author']}** · ❤️ {top_comment['likes']} likes · "
                           f"{top_comment['published_at'].strftime('%Y-%m-%d')}")
                st.write(top_comment["text"])
                st.caption(f"Sentiment: {top_comment['label']} ({top_comment['score']:.2f}) — "
                          f"{display_star_rating(top_comment['stars'])}")

        st.divider()
        st.subheader("All Analyzed Comments")

        display_cols = ["author", "likes", "published_at", "label", "score", "stars", "text"]
        display_df = meaningful_df[display_cols].copy()
        display_df["published_at"] = display_df["published_at"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["stars_display"] = display_df["stars"].apply(lambda s: "★" * s + "☆" * (5 - s))
        display_df = display_df.rename(columns={
            "author": "Author",
            "likes": "Likes",
            "published_at": "Date",
            "label": "Sentiment",
            "score": "Confidence",
            "stars": "Stars",
            "stars_display": "Rating",
            "text": "Comment"
        })

        st.dataframe(
            display_df[["Author", "Likes", "Date", "Sentiment", "Confidence", "Rating", "Comment"]],
            use_container_width=True,
            height=400,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1,
                    format="%.2f"
                ),
                "Comment": st.column_config.TextColumn("Comment", width="large"),
            }
        )

        csv = meaningful_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download as CSV",
            csv,
            f"youtube_sentiment_{video_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

    with tab4:
        st.subheader("Most Repeated Opinion (Semantic Similarity)")

        with st.spinner("Finding similar opinions..."):
            sim_result = find_similar_opinion_cached(tuple(meaningful_df["clean_text"].tolist()), min_similarity)

        if sim_result["cluster_size"] >= 2:
            st.success(f"Found a cluster of **{sim_result['cluster_size']}** similar comments "
                      f"(avg similarity: {sim_result['average_similarity']:.2f})")

            with st.container(border=True):
                st.markdown("**Representative Comment:**")
                st.write(sim_result["representative_text"])
                st.caption(f"Comment index: {sim_result['representative_index']}")

            with st.expander(f"View all {sim_result['cluster_size']} similar comments"):
                for idx in sim_result["similar_indices"]:
                    comment = meaningful_df.iloc[idx]
                    st.write(f"**{comment['author']}** (★{comment['stars']}) — {comment['clean_text'][:200]}...")
        else:
            st.info("No significant opinion clusters found (comments are too diverse)")

        st.divider()
        st.subheader("All Opinion Clusters")

        sentiment_service = get_sentiment_service()
        similarity_service = get_similarity_service(_sentiment_service=sentiment_service)
        similarity_service.min_similarity = min_similarity
        clusters = similarity_service.find_opinion_clusters(
            meaningful_df["clean_text"].tolist(),
            min_cluster_size=2
        )

        if clusters:
            for i, cluster in enumerate(clusters[:5]):
                with st.expander(f"Cluster {i+1}: {cluster.cluster_size} comments "
                               f"(avg sim: {cluster.average_similarity:.2f})"):
                    st.write(f"**Representative:** {cluster.representative_text}")
                    for idx in cluster.similar_indices[:10]:
                        comment = meaningful_df.iloc[idx]
                        st.write(f"• {comment['author']} (★{comment['stars']}) — "
                               f"{comment['clean_text'][:150]}...")
                    if len(cluster.similar_indices) > 10:
                        st.caption(f"... and {len(cluster.similar_indices) - 10} more")
        else:
            st.info("No clusters found with current similarity threshold")


if __name__ == "__main__":
    main()