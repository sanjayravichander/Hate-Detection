import streamlit as st
import pickle
import eli5
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk import trigrams
import pandas as pd
import base64
import time

# Function to set the background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
add_bg_from_local('C:/Users/DELL/Downloads/hate_classifier_img.jpeg')


# Function to load a pickle file
def load_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Load your trained logistic regression model and vectorizer
vectorizer = load_pickle("C:\\Users\\DELL\\Downloads\\Hate_or_NoHate_Classifier\\vec.pkl")
model = load_pickle("C:\\Users\\DELL\\Downloads\\Hate_or_NoHate_Classifier\\Log_Model.pkl")

# Assuming df_1 is loaded here from your dataset
df_1 = pd.read_excel("C:\\Users\\DELL\\Downloads\\Hate_or_NoHate_Classifier\\Hate_or_noHate.xlsx")
df_1.drop(['Unnamed: 0'],axis=1,inplace=True)
df_1['Preprocessed_Text'].fillna('NIL', inplace=True)
df_1 = df_1[~df_1['label'].isin(['relation', 'idk/skip'])]

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, y=None, **fit_params):
        return X.toarray()


# Initialize your Streamlit app
st.title('Hate Speech Detection')

# Text input for classification
user_input = st.text_area("Enter text to classify", "")

# Predict and Explain button
# Predict and Explain button
if st.button('Submit'):
    with st.spinner("Wait for it..."):
        time.sleep(2)
    # Transform user input using the loaded vectorizer and predict using the loaded model
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)
    prediction_proba = model.predict_proba(transformed_input)
    
    # Displaying prediction
    prediction_text = 'Hate Speech' if prediction[0] == 1 else 'No Hate Speech'
    st.write(f"Prediction: {prediction_text}")
    
    # Displaying prediction probabilities
    st.write(f"Probability of No Hate Speech: {prediction_proba[0][0]:.4f}")
    st.write(f"Probability of Hate Speech: {prediction_proba[0][1]:.4f}")
    
    # Creating a pipeline with your vectorizer and model for ELI5
    pipe = Pipeline([
        ('vectorizer', vectorizer),
        ('to_dense', DenseTransformer()),
        ('classifier', model)
    ])
    
    # Assuming 'explain_prediction' is a function that returns the explanation
    # Make sure to implement this or adjust according to your needs
    # explanation = explain_prediction(pipe, user_input)
    # st.write(explanation)

# Plot selection
plot_options = [
    "Pie Chart - Distribution of Hate and notHate",
    "Histogram - Distribution of Hate by user_id",
    "Histogram - Distribution of num_contexts for Hate",
    "Word Cloud - Hate",
    "Word Cloud - No Hate",
    "Bar Chart - Distribution of Contexts among Subforums"
]
selected_plot = st.selectbox("Choose a plot to display", plot_options)

# Function to display the selected plot
def display_selected_plot(option):
    # Placeholder for the DataFrame loading or processing
    # You need to load your DataFrame here
    # df_1 = pd.read_csv('path_to_your_dataframe.csv') or any appropriate loading method

    if option == "Pie Chart - Distribution of Hate and notHate":
        fig = px.pie(df_1, names='label', title='Distribution of Hate and notHate')
        st.plotly_chart(fig)
    
    elif option == "Histogram - Distribution of Hate by user_id":
        df_filtered = df_1[df_1['label'] == 'hate']
        fig = px.histogram(df_filtered, x='user_id', color='label', title='Distribution of Hate by user_id')
        st.plotly_chart(fig)
    
    elif option == "Histogram - Distribution of num_contexts for Hate":
        df_hate = df_1[df_1['label'] == 'hate']
        fig = px.histogram(df_hate, x='num_contexts', title='Distribution of num_contexts for Hate')
        st.plotly_chart(fig)
    
    elif option == "Bar Chart - Distribution of Contexts among Subforums":
        # Generate the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        subforum_counts = df_1['subforum_id'].value_counts()
        subforum_counts.plot(kind='bar', ax=ax, title='Distribution of Contexts among Subforums')
        ax.set_xlabel('Subforum ID')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif option == "Word Cloud - Hate":
        text = ' '.join(df_1[df_1['label'] == 'hate']['Preprocessed_Text']).lower()
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        showPyplotGlobalUse = False
        st.pyplot()
        
    
    elif option == "Word Cloud - No Hate":
        text = ' '.join(df_1[df_1['label'] == 'noHate']['Preprocessed_Text']).lower()
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        showPyplotGlobalUse = False
        st.pyplot()
       

# Display the selected plot
display_selected_plot(selected_plot)

