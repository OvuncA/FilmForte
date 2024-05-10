import main as ms
import streamlit as st

col1, col2, col3 = st.columns((1, 3, 1))
with col2:
    st.image("./image/logo.png")
st.markdown("**FilmForte** is your movie-loving friend with AI smarts who knows just how audiences will feel about your film ideas!")

st.sidebar.markdown('''
**Share the Story:** Tell us your film's genre and plot.

**Insightful Analysis:** See how audiences are feeling with accurate data from IMDb's top 250 movies.

**Make Smart Moves:** Armed with insights, make confident decisions about your film's journey. ''')
movie_form = st.sidebar.form('my_form')
app_movie_genre = movie_form.selectbox("Select your Genre:", 
                                       ("Action", "Adventure", "Animation",
                                        "Biography", "Crime", "Comedy", "Drama",
                                        "Family", "Fantasy", "Film-Noir",
                                        "History", "Horror", "Musical", "Mystery",
                                        "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"))

if app_movie_genre:
    app_movie_plot = movie_form.text_area("What is your plot?", max_chars = 200, height=180)
    submit_button = movie_form.form_submit_button(label='Submit')

if submit_button:
    response = ms.check_user_movie(app_movie_genre, app_movie_plot)
    st.markdown(response['movie_sentiment'])