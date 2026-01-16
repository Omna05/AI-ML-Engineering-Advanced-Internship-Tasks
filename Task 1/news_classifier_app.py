import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and fine-tuned model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./saved_model')  # replace with your path
model.eval()

st.title("News Topic Classifier")

user_input = st.text_area("Enter a news headline:")

if st.button("Predict"):
    inputs = tokenizer(user_input, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    st.write(f"Predicted Category: {prediction}")
