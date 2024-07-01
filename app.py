import streamlit as st
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer, BertForSequenceClassification
import numpy as np
import re
import html
import torch

LABELS = ['NEGATIVE', 'POSITIVE']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


load_directory = './bert_tiny_finetuned-1'
tokenizer = BertTokenizer.from_pretrained(load_directory)
model = BertForSequenceClassification.from_pretrained(load_directory)
model.to(device)

# Define Streamlit app
def main():
    st.title('Sentiment Analysis App')

    # User input
    text_input = st.text_input('Enter a sentence:', placeholder='Movies was fantastic!')

 # Analyze sentiment
    if st.button('Analyze'):
        if text_input:
            inputs = tokenizer(text_input, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            pred_prob = torch.sigmoid(logits)
            pred = torch.argmax(pred_prob, dim=1)
            sentiment = LABELS[pred.item()]
            st.write(f'The sentiment of the sentence is: {sentiment} with a probability of {pred_prob.max().item() * 100:.2f}% confidence.')

if __name__ == '__main__':
    main()