import streamlit as st
import joblib
import re
import requests
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.joblib")

model = load_model()

# Load the TfidfVectorizer
@st.cache_resource
def load_vectorizer():
    return joblib.load("tfidf_vectorizer.joblib")

vectorizer = load_vectorizer()

# Function to preprocess the tweet
def preprocess_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\S+", "", tweet)
    tweet = re.sub(r"#\S+", "", tweet)
    tweet = re.sub(r"[^A-Za-z0-9]", " ", tweet)
    tweet = tweet.lower().strip()
    return tweet

# URL of the background image
image_url = "https://media.istockphoto.com/id/1333043586/photo/tornado-in-stormy-landscape-climate-change-and-natural-disaster-concept.jpg?s=612x612&w=0&k=20&c=uo4HoloU79NEle1-rgVoLhKBE-RrfPSeinKAdczCo2I="

# Add CSS for background image, text color, and blurring
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: left;
    }}
    .blur-background {{
        backdrop-filter: blur(55px);
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(0, 0, 0, 0.8);
        color: white;
    }}
    .title {{
        color: white;
    }}
    .disaster {{
        color: red;
        background-color: #ffd1d1;
        padding: 10px;
        border-radius: 5px;
        display: inline-block;
    }}
    .not-disaster {{
        color: green;
        background-color: #d1ffd1;
        padding: 10px;
        border-radius: 5px;
        display: inline-block;
    }}
    .stTextInput > label {{
        color: white !important;
    }}
    .stTextInput > div > div > textarea {{
        background-color: white !important;
        color: black !important;
    }}
    .tweet-container {{
        color: white;
    }}
    .stButton > button {{
        background-color: white !important;
        color: black !important;
    }}
    .stMarkdown {{
        color: white;
    }}
    .line-separator {{
        border-top: 2px solid white;
        margin-top: 10px;
        margin-bottom: 10px;
    }}
    .stSelectbox > label {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Set up your bearer token
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAO9puQEAAAAAibBDDQEa%2BOYP%2BI%2BkQNdGzbfgOQQ%3DSryMUwRRPymuUcj0pA44K7TkHdRKXPE8LQrBeXkrQsIKgUHWDd'

def create_headers(bearer_token):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    return headers

def fetch_tweet_text(tweet_id):
    tweet_url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=geo"
    headers = create_headers(bearer_token)
    response = requests.get(tweet_url, headers=headers)
    if response.status_code != 200:
        st.error(f"Request returned an error: {response.status_code} {response.text}")
        return None, None
    tweet_data = response.json().get('data', {})
    tweet_text = tweet_data.get('text', '')
    tweet_location = tweet_data.get('geo', {}).get('place_id', 'Location not available')
    if tweet_location == 'Location not available':
        # Attempt to extract location from tweet text
        location_match = re.search(r"\b(in|at|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", tweet_text)
        if location_match:
            tweet_location = location_match.group(2)
        else:
            tweet_location = 'Unknown'
    return tweet_text, tweet_location

def extract_tweet_id(tweet_url):
    tweet_id = tweet_url.split('/')[-1]
    return tweet_id

# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Function to fetch recent disaster-related tweets
def fetch_recent_disaster_tweets(bearer_token, hours=5):
    end_time = datetime.utcnow() - timedelta(seconds=20)  # 20-second buffer to avoid invalid request time
    start_time = end_time - timedelta(hours=hours)
    query = '(fire OR earthquake OR flood OR hurricane OR tornado OR disaster) lang:en -is:retweet'
    headers = create_headers(bearer_token)
    params = {
        'query': query,
        'start_time': start_time.isoformat() + 'Z',
        'end_time': end_time.isoformat() + 'Z',
        'max_results': 10,
        'tweet.fields': 'created_at,id,text,geo'
    }
    response = requests.get("https://api.twitter.com/2/tweets/search/recent", headers=headers, params=params)
    if response.status_code != 200:
        st.error(f"Request returned an error: {response.status_code} {response.text}")
        return None
    return response.json()

# Streamlit app
st.markdown('<h1 class="title">Disaster Tweet Classifier</h1>', unsafe_allow_html=True)

# Input text box
st.markdown('<label style="color:white;">Enter the Tweet URL or Text</label>', unsafe_allow_html=True)
tweet_input = st.text_area("", key="tweet_input")

# Prediction button
if st.button("Fetch and Predict"):
    if tweet_input:
        try:
            if tweet_input.startswith("http"):
                tweet_id = extract_tweet_id(tweet_input)
                tweet_text, tweet_location = fetch_tweet_text(tweet_id)
                if tweet_text is None:
                    raise Exception("Failed to fetch tweet text")
                st.markdown(f'<div class="tweet-container blur-background">Tweet: {tweet_text}<br>Location: {tweet_location}</div>', unsafe_allow_html=True)
            else:
                tweet_text = tweet_input
            
            cleaned_text = preprocess_tweet(tweet_text)
            
            tfidf_features = vectorizer.transform([cleaned_text])
            
            prediction = model.predict(tfidf_features)
            
            if prediction[0] == 1:
                st.markdown('<div class="disaster">This tweet is likely about a real disaster</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="not-disaster">This tweet is not likely about a real disaster</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Fetch recent disaster tweets button with selection
st.markdown('<label style="color:white;">Select Time Period:</label>', unsafe_allow_html=True)
option = st.selectbox(
    'Select Time Period:',
    ('Today', 'Last 5 Hours')
)

if st.button("Fetch Recent Disaster Tweets"):
    try:
        if option == 'Today':
            tweets = fetch_recent_disaster_tweets(bearer_token, hours=24)
        elif option == 'Last 5 Hours':
            tweets = fetch_recent_disaster_tweets(bearer_token, hours=5)
        
        if tweets and 'data' in tweets:
            for tweet in tweets['data']:
                tweet_text = tweet['text']
                tweet_date = tweet['created_at']
                tweet_location = tweet.get('geo', {}).get('place_id', 'Unknown')
                if tweet_location == 'Unknown':
                    # Attempt to extract location from tweet text
                    location_match = re.search(r"\b(in|at|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", tweet_text)
                    if location_match:
                        tweet_location = location_match.group(2)
                st.markdown(f'<div class="tweet-container blur-background">Tweet: {tweet_text}<br>Time: {tweet_date}<br>Location: {tweet_location}</div>', unsafe_allow_html=True)
                st.markdown('<div class="line-separator"></div>', unsafe_allow_html=True)
        else:
            st.write("No disaster tweets found in the selected time period.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
