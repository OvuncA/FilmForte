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
def anaylze_movie(plot, sentiments, topics):
    movie_llm = OpenAI(temperature=0.6)

    prompt_template_title = PromptTemplate(
        input_variables = ['plot', 'sentiments', 'topics'],
        template = """
        As an advanced movie critic algorithm, your task is to analyze new movie plots along with sentiment scores and keywords extracted from reviews of existing movies with similar plots. 
        By doing this, you can estimate the overall audience perception accurately.
        Start by examining the {plot} (new movie plot) with keen attention to detail. Break down the storyline, character arcs, conflicts, and resolutions. 
        Then, review sentiment scores: {sentiments} from reviews of existing movies with similar plots and identify the keywords that commonly appear in positive and negative reviews to understand audience reactions better from the keywords: {topics}.
        
        Next, correlate the sentiment scores and keywords from past movies with the new movie plot. Look for patterns that indicate audience preferences and dislikes. 
        This analysis will help you gauge how the audience might perceive the new movie.
        Keep in mind that the accuracy of your estimation relies on the thoroughness of your analysis. 
        Pay close attention to nuances in the plot, sentiments expressed in reviews, and recurring keywords. By applying this methodology diligently, you can provide valuable insights into the potential reception of the new movie.
        
        For example, when analyzing a new sci-fi movie plot, consider sentiments and keywords from past successful sci-fi films. 
        Look for keywords like "engaging," "futuristic," or "mind-bending" in positive reviews, and "predictable," "disjointed," or "underdeveloped" in negative reviews to draw informed conclusions about audience perception.
        
        Remember to focus on how the movie can be improved to appeal to a wider audience and receive positive feedback.
        """
    )

    movie_name_chain = LLMChain(llm = movie_llm, prompt = prompt_template_title, output_key = "movie_sentiment")
    response = movie_name_chain.invoke({'plot' : plot, 'sentiments' : sentiments, 'topics' : topics})

    return response

# The function that improvises if there are no similar movies in top 250
def no_similar_movie(user_genre, user_plot):
    movie_llm = OpenAI(temperature=0.6)

    prompt_template_title = PromptTemplate(
        input_variables = ['user_genre', 'user_plot'],
        template = """
        As an advanced movie critic algorithm, your task is to analyze new movie plots with genres.
        By doing this, you can estimate the overall audience perception accurately.
        Start by examining the {user_plot} (new movie plot) that's in {user_genre} (genre) the with keen attention to detail. Break down the storyline, character arcs, conflicts, and resolutions. 

        Imagine you are analyzing a new movie plot and genre details. Considering the audience's preferences, give recommendations on how the film can be enhanced to cater better to their tastes. 
        You should focus on aspects such as character development, plot twists, pacing, emotions, and overall engagement to provide constructive feedback for the movie creators to implement for a more successful outcome.
        This analysis will help you gauge how the audience might perceive the new movie.
        Keep in mind that the accuracy of your estimation relies on the thoroughness of your analysis. 

        For instance, after evaluating a science fiction movie with a complex plot and lacking character depth, suggest enriching the main characters' backstories to create a stronger emotional connection with the viewers.
        Remember to focus on how the movie can be improved to appeal to a wider audience and receive positive feedback.
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

    response = anaylze_movie(user_plot, sentiments, topics)
    return(response)
