from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from imdb import Cinemagoer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import sentiment as sta
import json


load_dotenv()
ia = Cinemagoer('http')
input_file = './top250_movies.json'

# Global variable to store cached data (initially empty)
cached_data = {}

def load_top_250_movies():
    global cached_data
    if not cached_data:
        with open(input_file, 'r') as json_file:
            cached_data = json.load(json_file)
    return cached_data

# Function to search for movies based on genre and premise
def search_similar_movies(user_genre, user_plot):
    
    top250_movies = load_top_250_movies()
  
    similar_movies_filtered = []

    filtered_movie_ids = [movie_id for movie_id, info in top250_movies.items() if user_genre in info['genres']]

    # Get all the same genre movies from the filtered movie ids
    for m_id in filtered_movie_ids:

        # Get plot of the movie
        plot = top250_movies[m_id]['plot_outline']

        # Compare plot outlines using language model
        similarity_score = compare_plots(user_plot, plot)
        
        # If similarity score is above a certain threshold, consider it a similar movie
        threshold = 0.5 
        if similarity_score >= threshold:
            similar_movies_filtered.append((m_id, similarity_score))
    
    if not similar_movies_filtered:
        return 0
    
    # Get the most alike movie
    similar_movies_filtered.sort(key=lambda x: x[1], reverse=True)
    return similar_movies_filtered[0][0]

# Function to compare two plot outlines using a language model
def compare_plots(user_plot, plot):
    # Preprocess the text
    def preprocess_text(text):
            
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        text = text.lower()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text
    
    user_plot = preprocess_text(user_plot)
    plot = preprocess_text(plot)
    
    # Initialize the language model chain
    core_embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
    
    # Generate embeddings for user plot and movie plot
    user_embedding = core_embeddings.embed_query(user_plot)
    movie_embedding = core_embeddings.embed_query(plot)

    # Calculate cosine similarity between embeddings
    similarity_score = np.dot(user_embedding, movie_embedding) / (np.linalg.norm(user_embedding) * np.linalg.norm(movie_embedding))

    return similarity_score

# Function to retrieve sentiment score for a movie premise
def anaylze_movie(sentiments, topics):
    movie_llm = OpenAI(temperature=0.7)

    prompt_template_title = PromptTemplate(
        input_variables = ['sentiments', 'topics'],
        template = """
        You are a robust and creative Hollywood production algorithm specialized in analyzing movie plots to make accurate audience perception estimations.
        
        You achieve this by analyzing sentiment scores and keywords from movie reviews.

        Analyze the following data:

        Sentiments: {sentiments} 
        Keywords: {topics}

        **Write a compelling and insightful movie analysis for a brand new movie idea based on the summary above, without directly mentioning the specific sentiments or topics.**
        Use only first conditional mood tenses in your answer to describe.
        Your review should be engaging, technical and informative, providing an unbiased overall opinion of the movie and its potential reception by the audience and should not exceed a paragraph. 

        Focus on the movie itself and avoid mentioning that you were given any specific details about sentiments or keywords.
        """
    )

    movie_name_chain = LLMChain(llm = movie_llm, prompt = prompt_template_title, output_key = "movie_sentiment")
    response = movie_name_chain.invoke({'sentiments' : sentiments, 'topics' : topics})

    return response

# The function that improvises if there are no similar movies in top 250
def no_similar_movie(user_genre, user_plot):
    movie_llm = OpenAI(temperature=0.7)

    prompt_template_title = PromptTemplate(
        input_variables = ['user_genre', 'user_plot'],
        template = """
        You are a robust and creative Hollywood production algorithm specialized in analyzing movie plots to make accurate audience perception estimations.
        
        You achieve this by analyzing genre and plot that's given to you.

        Analyze the following data:

        Genre: {user_genre} 
        Plot: {user_plot}

        **Write a compelling and insightful movie analysis for a brand new movie idea based on the summary above, without directly mentioning the specific sentiments or topics.**
        Use only first conditional mood tenses in your answer to describe.
        Your review should be engaging, technical and informative, providing an unbiased overall opinion of the movie and its potential reception by the audience and should not exceed a paragraph.
        """
    )

    movie_name_chain = LLMChain(llm = movie_llm, prompt = prompt_template_title, output_key = "movie_sentiment")
    response = movie_name_chain.invoke({'user_genre' : user_genre, 'user_plot' : user_plot})

    return response


# the main function that will be called in main
def check_user_movie(user_genre, user_plot):
    the_movie_id = search_similar_movies(user_genre, user_plot)
    if not the_movie_id:
        response = no_similar_movie(user_genre, user_plot)
        return(response)
    the_movie = ia.get_movie(the_movie_id, ['reviews'])
    allreviews = the_movie['reviews'] 
  
    json_data = json.dumps(allreviews)
    parsed_data = json.loads(json_data) 
  
    reviews = [ sub['content'] for sub in parsed_data ]
    
    # Function to predict a score based on IMDb data and sentiment scores
    sentiments = sta.analyze_sentiment(reviews)
    topics = sta.extract_topics(reviews)

    response = anaylze_movie(sentiments, topics)
    return(response)
