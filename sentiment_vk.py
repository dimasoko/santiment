import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from wordcloud import WordCloud

CSV_PATH = "vk_filtered_posts_metadata.csv"

RUSSIAN_STOPWORDS = set([
    'в', 'на', 'и', 'с', 'по', 'к', 'из', 'у', 'о', 'об', 'от', 'до', 'для',
    'за', 'при', 'через', 'над', 'под', 'а', 'но', 'или', 'что', 'как', 'то',
    'так', 'это', 'все', 'еще', 'уже', 'только', 'же', 'бы', 'ли', 'не', 'ни',
    'я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они', 'мой', 'твой', 'его',
    'её', 'наш', 'ваш', 'их', 'себя', 'этот', 'тот', 'такой', 'весь', 'который',
    'чем', 'где', 'куда', 'когда', 'почему', 'зачем', 'если', 'чтобы', 'хотя',
    'ведь', 'вот', 'вон', 'даже', 'лишь', 'нет', 'да', 'ну', 'ой', 'ах', 'эх'
])


def load_data(path=CSV_PATH, text_col="Text"):
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    return df


def basic_clean(text: str, remove_stopwords=True) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[#@]\S+", " ", text)
    text = re.sub(r"[^а-яa-zё\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    if remove_stopwords:
        words = text.split()
        words = [w for w in words if w not in RUSSIAN_STOPWORDS and len(w) > 2]
        text = " ".join(words)
    
    return text


def build_classifier():
    clf = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment"
    )
    return clf


def add_sentiment(df, text_col="clean_text", batch_size=32):
    clf = build_classifier()
    texts = df[text_col].tolist()
    sentiments = []

    print("Analyzing sentiment...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        preds = clf(batch)
        sentiments.extend(preds)
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + batch_size}/{len(texts)} posts")

    df["sent_label"] = [p["label"] for p in sentiments]
    df["sent_score"] = [p["score"] for p in sentiments]
    return df


def sentiment_statistics(df):
    """Базовая статистика по сантиментам"""
    print("\n=== SENTIMENT STATISTICS ===")
    counts = df["sent_label"].value_counts()
    percentages = df["sent_label"].value_counts(normalize=True) * 100
    
    stats_df = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })
    print(stats_df)
    
    print(f"\nAverage confidence score: {df['sent_score'].mean():.3f}")
    print(f"Total posts analyzed: {len(df)}")
    
    return stats_df


def plot_sentiment_distribution(df, out_file="sentiment_distribution.png"):
    """Диаграмма распределения сантиментов"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    counts = df["sent_label"].value_counts()
    colors = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"}
    bar_colors = [colors.get(label, "#3498db") for label in counts.index]
    
    axes[0].bar(counts.index, counts.values, color=bar_colors, alpha=0.8)
    axes[0].set_title("Sentiment Distribution (Count)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Number of Posts")
    axes[0].grid(axis="y", alpha=0.3)
    
    # Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[colors.get(label, "#3498db") for label in counts.index],
        startangle=90
    )
    axes[1].set_title("Sentiment Distribution (%)", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")


def plot_text_length_by_sentiment(df, out_file="text_length_by_sentiment.png"):
    """Длина текстов по сантиментам"""
    df["text_length"] = df["clean_text"].str.split().str.len()
    
    plt.figure(figsize=(10, 6))
    df.boxplot(column="text_length", by="sent_label", grid=False, patch_artist=True)
    plt.suptitle("")
    plt.title("Text Length Distribution by Sentiment", fontsize=14, fontweight="bold")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Words")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")


def top_words_by_sentiment(df, top_n=15):
    """Топ слов по каждому сантименту"""
    print("\n=== TOP WORDS BY SENTIMENT ===")
    for sentiment in df["sent_label"].unique():
        subset = df[df["sent_label"] == sentiment]
        all_text = " ".join(subset["clean_text"].tolist())
        words = all_text.split()
        word_freq = pd.Series(words).value_counts().head(top_n)
        
        print(f"\n{sentiment}:")
        print(word_freq.to_string())


def make_wordcloud(
    df,
    text_col="clean_text",
    label_col=None,
    label_value=None,
    out_file="wordcloud.png",
):
    if label_col is not None and label_value is not None:
        df = df[df[label_col] == label_value]
    
    if len(df) == 0:
        print(f"Warning: No data for {label_value}")
        return

    text = " ".join(df[text_col].tolist())
    wc = WordCloud(
        width=1600,
        height=800,
        background_color="white",
        collocations=False,
        min_word_length=3,
        max_words=200
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud - {label_value if label_value else 'All Posts'}", 
              fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    print("Loading data...")
    df = load_data()
    
    print("Cleaning text...")
    df["clean_text"] = df["Text"].apply(basic_clean)
    df = df[df["clean_text"].str.len() > 2]
    
    df = add_sentiment(df)
    
    # Сохраняем результаты
    df.to_csv("vk_with_sentiment.csv", index=False)
    print("\nSaved: vk_with_sentiment.csv")
    
    # Статистика
    sentiment_statistics(df)
    top_words_by_sentiment(df, top_n=15)
    
    # Визуализации
    print("\nGenerating visualizations...")
    plot_sentiment_distribution(df)
    plot_text_length_by_sentiment(df)
    
    # Облака слов
    make_wordcloud(df, out_file="wc_all.png")
    make_wordcloud(df, label_col="sent_label", 
                   label_value="POSITIVE", out_file="wc_positive.png")
    make_wordcloud(df, label_col="sent_label",
                   label_value="NEGATIVE", out_file="wc_negative.png")
    make_wordcloud(df, label_col="sent_label",
                   label_value="NEUTRAL", out_file="wc_neutral.png")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(df[["Text", "clean_text", "sent_label", "sent_score"]].head(10))


if __name__ == "__main__":
    main()
