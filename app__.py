# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences
# import streamlit as st

# word_index = imdb.get_word_index()
# reverse_word_index = {value: key for key, value in word_index.items()}

# model = load_model('simple_rnn_model.h5')

# # Function to decode the review
# def decode_review(encoded_review):
#     return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# # Function to preprocess user input
# def preprocess_text(text):
#     words = text.lower().split()
#     encoded_review = [word_index.get(word, 2) + 3 for word in words]
#     padded_review = pad_sequences([encoded_review], maxlen=500)
#     return padded_review

# def predict_sentiment(review):
#     preprocessed_input = preprocess_text(review)
#     prediction = model.predict(preprocessed_input)
#     sentiment = "positive" if prediction >= 0.5 else "negative"
#     return sentiment, prediction[0][0]

# # Streamlit app layout
# st.title('IMDB Movie Review Sentiment Analysis')

# st.write('Enter a movie review to classify if it is positive or negative')

# # Get user input
# user_input = st.text_area('Movie Review')

# if st.button('Classify'):
#     if user_input.strip():  # Check if the input is not empty
#         sentiment, score = predict_sentiment(user_input)
#         st.write(f'Sentiment: {sentiment}')
#         st.write(f'Prediction Score: {score}')
#     else:
#         st.warning("Please enter a review!")



import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_model.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=500, padding='post', truncating='post')
    print(f"Padded Input Shape: {padded_review.shape}")
    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    print(f"Preprocessed Input: {preprocessed_input}")
    prediction = model.predict(preprocessed_input)
    print(f"Prediction Output: {prediction}")
    sentiment = "positive" if prediction[0][0] >= 0.5 else "negative"
    return sentiment, prediction[0][0]

st.title('IMDB Movie Review Sentiment Analysis')

st.write('Enter a movie review to classify it as positive or negative.')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    if user_input.strip():
        try:
            sentiment, score = predict_sentiment(user_input)
            st.write(f'**Sentiment:** {sentiment}')
            st.write(f'**Prediction Score:** {score:.2f}')
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.warning("Please enter a review!")

