import streamlit as st
from transformers import AutoTokenizer, DataCollatorWithPadding, BertTokenizer, BertForSequenceClassification
import torch

LABELS = ['NEGATIVE', 'POSITIVE']
device = 'cuda' if torch.cuda.is_available() else 'cpu'


load_directory = './bert_tiny_finetuned-1'
tokenizer = BertTokenizer.from_pretrained(load_directory)
model = BertForSequenceClassification.from_pretrained(load_directory)
model.to(device)

# Define Streamlit app


def main():
    """
    Runs the Sentiment Analysis App.

    This function displays a Streamlit app that allows users to input a sentence and analyze its sentiment. The app prompts the user to enter a sentence using a text input field. When the user clicks the "Analyze" button, the function tokenizes the input text, passes it through a pre-trained BERT model, and predicts the sentiment of the sentence. The predicted sentiment and the confidence score are then displayed on the app.

    Parameters:
    None

    Returns:
    None
    """
    st.title('Sentiment Analysis App')

    # User input
    text_input = st.text_input(
        'Enter a sentence:', placeholder='Movies was fantastic!')

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
            st.write(
                f'The review is: {sentiment} with a probability of {pred_prob.max().item() * 100:.2f}% confidence.')


if __name__ == '__main__':
    main()
