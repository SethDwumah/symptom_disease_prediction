import streamlit as st
import pickle
from streamlit_chat import message
from helper_prabowo_ml import clean_html, remove_, remove_digits, remove_special_characters, removeStopWords, remove_links, punct, email_address, lower, non_ascii
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# Load the machine learning model components
vect = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('Disease_model.pkl', 'rb'))
lab_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

def clean_text_for_prediction(text):
    # Apply the cleaning functions step by step on a single text instance
    text = clean_html(text)                # Remove HTML tags
    text = remove_links(text)              # Remove URLs
    text = email_address(text)             # Remove email addresses
    text = remove_digits(text)             # Remove digits
    text = remove_special_characters(text) # Remove special characters
    text = removeStopWords(text)           # Remove stopwords
    text = punct(text)                     # Remove punctuation
    text = non_ascii(text)                 # Remove non-ASCII characters
    text = lower(text)                     # Convert text to lowercase
    
    return text

def init():
    st.set_page_config(page_title="Disease Prediction Assistant", page_icon=":pill:")
    st.header("Disease Prediction Based on Symptoms")
    

def main():
    init()

    # Initialize conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Get user input
    user_input = st.chat_input("Describe your symptoms...", key='user_input')

    if user_input:
        st.session_state.messages.append({"content": user_input, "is_user": True})

        # Clean the user input and make a prediction using the ML model
        cleaned_text = clean_text_for_prediction(user_input)
        emb_text = vect.transform([cleaned_text])  # Transform the cleaned text
        prediction = model.predict(emb_text)  # Predict the disease
        label = lab_encoder.inverse_transform(prediction)  # Get the label in text form

        # Generate response with predicted disease
        response = f"The predicted disease based on your symptoms is: {label[0]}"

        # Append response to conversation
        st.session_state.messages.append({"content": response, "is_user": False})

    # Display the conversation
    placeholder = st.empty()
    with placeholder.container():
        messages = st.session_state.get('messages', [])
        for i, msg in enumerate(messages):
            message(msg['content'], is_user=msg['is_user'], key=str(i))

if __name__ == '__main__':
    main()
