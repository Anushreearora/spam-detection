import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the trained logistic regression model
with open("logistic_regression_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# Load the BERT vectorizer
bert_model = SentenceTransformer("bert_vectorizer")

st.title("BERT-based Spam Detection Web App")
st.write("Enter a message below to check if it's Spam or Not Spam.")

# User input
user_input = st.text_area("Enter your message:")

if st.button("Check Message"):
    if user_input:
        # Convert user input to BERT embedding
        input_embedding = bert_model.encode([user_input])
        
        # Predict spam or not
        prediction = classifier.predict(input_embedding)[0]
        
        # Display result
        if prediction == "spam":
            st.error("🚨 This message is classified as SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("⚠️ Please enter a message to classify.")