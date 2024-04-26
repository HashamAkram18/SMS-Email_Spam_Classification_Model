import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize NLTK's Porter Stemmer
ps = PorterStemmer()

# Load the TF-IDF vectorizer and the trained model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Function to preprocess and transform text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove non-alphanumeric tokens and punctuation
    tokens = [word for word in tokens if word.isalnum() and word not in string.punctuation]
    # Remove stopwords and apply stemming
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into text
    return " ".join(tokens)

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for user to enter the message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    # Preprocess and transform input text
    transformed_sms = transform_text(input_sms)
    # Vectorize transformed text
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the trained model
    result = model.predict(vector_input)[0]
    # Clear input text area
    input_sms = ""
    # Display prediction
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")