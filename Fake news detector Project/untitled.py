import numpy as np
import pandas as pd
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC


def predict_fake_news(text):
    data = pd.read_csv("fake_or_real_news.csv")
    data['fake'] = data['label'].apply(lambda x: 0 if x == 'REAL' else 1)
    data = data.drop('label', axis=1)
    x, y = data['text'], data['fake']
    x_train ,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
    vectorizer = TfidfVectorizer(stop_words = 'english' ,max_df = 0.7)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    clf = LinearSVC()
    clf.fit(x_train_vectorized, y_train)
    with open('mytext.txt', 'w', encoding = 'utf-8') as f1:
        f1.write(text)
    with open('mytext.txt', 'r' , encoding='utf-8') as f:
        text = f.read()
    vectorized_text = vectorizer.transform([text])
    prediciton =clf.predict(vectorized_text)
    if prediciton[0] ==1:
        predictionResult = 'fake'
    elif prediciton[0] ==0:
        predictionResult ='true'
    else :
        predictionResult ==None
    os.remove('mytext.txt')
    return predictionResult
# Streamlit app
def main():
    # Set the title and description
    st.title('Fake News Detection App')
    st.markdown('Enter a news article text and click the "Check" button to detect if it is fake or real.')

    # Get user input
    text = st.text_area('Enter the news article text:', height=300)


    # Check if the user submitted a text
    if st.button('Check'):
        if text:
            # Make the prediction
            prediction = predict_fake_news(text)

            # Display the prediction
            if prediction == 'fake':
                st.error('This news article is **fake**.')
            elif prediction =='true':
                st.success('This news article is **real**.')
            else:
            	st.warning("Error")
        else:
            st.warning('Please enter a news article text.')

# Run the app
if __name__ == '__main__':
    main()

