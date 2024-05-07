import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# Initialize Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(reviews):
    """
    Analyze sentiment of a review using Sentiment Intensity Analyzer.
    Returns a sentiment score.
    """
    sentiments = []
    
    for review in reviews:
        sentiment_score = sia.polarity_scores(review)
        sentiments.append(sentiment_score['compound'])
    
    return sentiments  # Compound score ranges from -1 (most negative) to 1 (most positive)

def extract_topics(reviews, num_topics=5):
    """
    Extract key topics/themes from reviews using Latent Dirichlet Allocation (LDA).
    Returns the top keywords for each topic.
    """
    # Preprocessing
    reviews = [review.lower() for review in reviews]  # Convert to lowercase
    reviews = [''.join([c for c in review if c not in string.punctuation]) for review in reviews]  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    reviews = [' '.join([word for word in word_tokenize(review) if word not in stop_words]) for review in reviews]  # Remove stopwords

    # Need to figure out how to get the write topics
    # Vectorize the reviews
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(reviews)
    
    # Apply LDA
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    
    # Get top keywords for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords_idx = topic.argsort()[:-10:-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
         
        # Filter out certain types of words (e.g., nouns, adjectives)
        filtered_keywords = []
        for keyword in top_keywords:
            if keyword.lower() in ['movie', 'movies', 'watch', 'film', 'films', 'scene', 'scenes' , 'review', 'reviews']:
                continue
            pos_tags = pos_tag(word_tokenize(keyword))
            # Check if the word is a noun or an adjective
            if pos_tags[0][1] in ['NN', 'NNS','JJ', 'JJR', 'JJS']:
                filtered_keywords.append(keyword)
        
        topics.append(filtered_keywords)
    
    return topics
    