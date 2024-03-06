#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Contextual Analysis (Career Fair Final Project)
# For getting texts 
import os
import pandas as pd
Text_df = "C:/Users/DELL/Downloads/Career_Fair_Projects_Final/Hate_Speech/Hate_Speech_1/Hate_Speech_Dataset/Hate_Speech_Dataset_1/Text_file/" ## This is in string value
text_list = os.listdir(Text_df) ## Here we are changing the above mentioned string path to actual location path for that we are using listdir() function


# In[2]:


# Getting those text values
Columns = {"Review_Text":[]}

for text in text_list:
    text_path = os.path.join(Text_df, text)
    if os.path.isfile(text_path):
        with open(text_path, 'r', encoding='utf-8') as file:
            Columns["Review_Text"].append(file.read())
    else:
        print(f"{text_path} is not a file.")

# Converting to Dataframe from a dictionary value
Columns=pd.DataFrame(Columns)


# In[3]:


Text=Columns.copy()


# In[4]:


Text.head()


# In[5]:


# Getting another dataset
Text_info=pd.read_csv("C:\\Users\\DELL\\Downloads\\Career_Fair_Projects_Final\\Hate_Speech\\Hate_Speech_1\\Hate_Speech_Dataset\\Hate_Speech_Dataset_1\\Annotations_Metadata.csv")


# In[6]:


Text_info.head()


# In[7]:


Text_info.columns


# In[8]:


import pandas as pd

df1 = Text 
df2 = Text_info

# Creating a new column 'row_number' in both datasets
df1['row_number'] = range(1, len(df1) + 1)
df2['row_number'] = range(1, len(df2) + 1)

# Merge the datasets on 'row_number'
df = pd.merge(df1, df2, on='row_number')

# You can drop the 'row_number' column if you don't need it anymore
df = df.drop('row_number', axis=1)


# In[9]:


df.head()


# In[10]:


# checking any duplicates
df[df.duplicated]


# In[11]:


# Checking any null-values
df.isnull().sum()


# In[12]:


df['file_id'].unique()


# In[13]:


df.drop(['file_id'],axis=1,inplace=True)


# In[14]:


df


# In[15]:


import re
import nltk
import string
import spacy
import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Load English language model for spaCy
nlp = spacy.load("en_core_web_sm")

# Sample text data (replace with your actual data)
text_data = df['Review_Text']

# Function to remove special characters
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Initialization
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Sentiment analysis preprocessing
for text in text_data:
    text_no_special_chars = remove_special_characters(text.lower())
    tokens = word_tokenize(text_no_special_chars)
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation]
    sentiment_score = sia.polarity_scores(text)['compound']
    
    print("Original Text:", text)
    print("Cleaned Tokens:", cleaned_tokens)
    print("Sentiment Score:", sentiment_score)
    print()

# LDA topic modeling preprocessing
preprocessed_texts = []

for text in text_data:
    text_no_special_chars = remove_special_characters(text.lower())
    doc = nlp(text_no_special_chars)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    preprocessed_texts.append(tokens)

# Create dictionary and corpus for LDA
dictionary = Dictionary(preprocessed_texts)
corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]

# Train LDA model
lda_model = LdaModel(corpus, id2word=dictionary, num_topics=2, passes=10)

# Print topics identified by LDA
print("Topics identified by LDA:")
for topic_id, topic in lda_model.print_topics():
    print(f"Topic {topic_id + 1}: {topic}")


# In[16]:


df['label'].value_counts()


# In[17]:


# Text Preprocessing
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

text=df['Review_Text']

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Tokenization
    text = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    # Convert back to text
    text = ' '.join(text)
    return text

# Apply preprocessing function to Review_Text column
df['Preprocessed_Text'] = df['Review_Text'].apply(preprocess_text)

# Drop the non-preprocessed Review_Text column
df.drop(columns=['Review_Text'], inplace=True)


# In[18]:


df


# In[19]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import HdpModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize


# In[20]:


# NMF
def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx + 1))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))
        print()
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
data_vectorized = vectorizer.fit_transform(df['Preprocessed_Text'])
nmf_model = NMF(n_components=15, max_iter=1000, random_state=0)
nmf_model.fit(data_vectorized)
num_top_words = 20
display_topics(nmf_model, vectorizer.get_feature_names_out(), num_top_words)


# In[21]:


#Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Prepare training data
documents = [TaggedDocument(word_tokenize(doc.lower()), [i]) for i, doc in enumerate(df['Preprocessed_Text'])]

# Train a Doc2Vec model
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# Infer vector for a new document
vector = model.infer_vector(["example", "text"])

# Assuming you have access to the vocabulary
vocabulary = model.wv.index_to_key

top_words_per_topic = 15
similar_docs = model.dv.most_similar(positive=[vector], topn=top_words_per_topic)
for i, (idx, score) in enumerate(similar_docs):
    topic_words = [vocabulary[idx]]
    print(f"Topic {i + 1}: {', '.join(topic_words)}")


# In[22]:


# EDA
import pandas as pd
import plotly.express as px

# EDA
# Assuming df contains your DataFrame with 'label' column
# Create an interactive pie chart
fig = px.pie(df, names='label', title='Distribution of Hate and notHate')
fig.update_layout(width=600, height=400)  # Set width and height as desired
fig.show()


# In[23]:


df['label'].value_counts()


# In[24]:


# Neglecting the label=relation/idk/skip to focus more on hate/noHate
df_1 = df[~df['label'].isin(['relation', 'idk/skip'])]


# In[25]:


df_1['label'].value_counts()


# In[26]:


fig = px.pie(df_1, names='label', title='Distribution of Hate and notHate')
fig.update_layout(width=600, height=400)  # Set width and height as desired
fig.show()


# In[27]:


df_1.columns


# In[28]:


df_1.head()


# In[29]:


import plotly.express as px

# Filter the DataFrame for rows with hate and noHate labels
df_filtered = df_1[df_1['label'].isin(['hate'])]

# Create an interactive bar plot
fig = px.histogram(df_filtered, x='user_id', color='label', title='Distribution of Hate by user_id')

# Ensure user_id values are not abbreviated
fig.update_xaxes(categoryorder='array', categoryarray=df_filtered['user_id'].unique())

fig.show()


# In[30]:


import plotly.express as px

# Filter the DataFrame for rows with the "hate" label
df_hate = df_1[df_1['label'] == 'hate']

# Create an interactive histogram plot
fig = px.histogram(df_hate, x='num_contexts', title='Distribution of num_contexts for Hate')

fig.show()


# In[31]:


from wordcloud import WordCloud
from nltk import trigrams
from collections import Counter
import matplotlib.pyplot as plt

# Filter the dataframe for rows where label is 'hate'
hate_df = df_1[df_1['label'] == 'hate']

# Concatenate all the text into a single string
hate_text = ' '.join(hate_df['Preprocessed_Text'])

# Convert the text to lowercase (optional, for consistency)
hate_text = hate_text.lower()

# Tokenize the text into words
words = hate_text.split()

# Generate trigrams
trigram_list = list(trigrams(words))

# Convert trigrams back to strings
trigram_strings = [' '.join(trigram) for trigram in trigram_list]

# Count the frequency of each trigram
trigram_freq = Counter(trigram_strings)

# Create the word cloud object
wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(trigram_freq)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Trigram Word Cloud for Text with Label "Hate"')
plt.show()


# In[32]:


from wordcloud import WordCloud
from nltk import trigrams
from collections import Counter
import matplotlib.pyplot as plt

# Filter the dataframe for rows where label is 'hate'
no_hate_df = df_1[df_1['label'] == 'noHate']

# Concatenate all the text into a single string
no_hate_df = ' '.join(no_hate_df['Preprocessed_Text'])

# Convert the text to lowercase (optional, for consistency)
no_hate_df = no_hate_df.lower()

# Tokenize the text into words
words = no_hate_df.split()

# Generate trigrams
trigram_list = list(trigrams(words))

# Convert trigrams back to strings
trigram_strings = [' '.join(trigram) for trigram in trigram_list]

# Count the frequency of each trigram
trigram_freq = Counter(trigram_strings)

# Create the word cloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(trigram_freq)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Trigram Word Cloud for Text with Label "noHate"')
plt.show()


# In[33]:


# Convert 'Label' column to numerical format using label encoding
label_mapping = {'hate': 1, 'noHate': 0}  # Adjust based on your actual labels
df_1['Label_Num'] = df_1['label'].map(label_mapping)


# In[34]:


df_1


# In[35]:


import plotly.express as px
import pandas as pd

# Assuming df_1 is your DataFrame containing the relevant data

# Select only numerical columns excluding the target variable
numeric_features = df_1.select_dtypes(include=['number']).drop(columns=['Label_Num'])

# Compute the correlation matrix
corr_matrix = numeric_features.corrwith(df_1['Label_Num'])

# Plot an interactive bar plot
fig = px.bar(corr_matrix,
             labels=dict(x="Features", y="Correlation"),
             title='Correlation of Features with Label_Num',
             width=800,
             height=600)

# Update layout
fig.update_xaxes(title='Features')
fig.update_yaxes(title='Correlation')

# Show the plot
fig.show()


# In[50]:


import pandas as pd

# Assuming df_1 is your DataFrame containing the relevant data

# Display the unique subforum IDs and their frequencies
subforum_counts = df_1['subforum_id'].value_counts()
#print("Unique Subforum IDs and their frequencies:")
#print(subforum_counts)

# Plot a bar chart to visualize the distribution of contexts among various subforums
subforum_counts.plot(kind='bar', figsize=(10, 6), title='Distribution of Contexts among Subforums')
plt.xlabel('Subforum ID')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Filter and display data for a specific subforum (for example, subforum with ID = 1)
subforum_id_to_filter = 1
subforum_data = df_1[df_1['subforum_id'] == subforum_id_to_filter]
#print("\nData for Subforum ID", subforum_id_to_filter, ":")
#print(subforum_data)


# In[37]:


df_1['Label_Num']=df_1['label'].map({'hate':1,'noHate':0}).astype('int')


# In[38]:


df_1['Label_Num'].value_counts()


# In[39]:


from sklearn.utils import resample

# Separate majority and minority classes
majority_class = df_1[df_1['Label_Num'] == 0]
minority_class = df_1[df_1['Label_Num'] == 1]

# Upsample minority class
minority_upsampled = resample(minority_class, 
                               replace=True,     # sample with replacement
                               n_samples=len(majority_class),    # to match majority class
                               random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
balanced_df = pd.concat([majority_class, minority_upsampled])


# In[40]:


balanced_df


# In[41]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Converting the Text column to Vectors which is a king of Sparse matrix where distinct values given to each words in sentence
vec = CountVectorizer(binary=True)
train_data = vec.fit_transform(balanced_df['Preprocessed_Text'])
train_labels = balanced_df['Label_Num']

# Train the logistic regression classifier
classifier = LogisticRegression()
classifier.fit(train_data, train_labels)

# Evaluate the model on the training set
train_predictions = classifier.predict(train_data)

# Calculate confusion matrix and accuracy
c = confusion_matrix(train_labels, train_predictions)
a = accuracy_score(train_labels, train_predictions)
c


# In[42]:


a


# In[43]:


#Making predictions after training the model to check if it is working properly
prediction = classifier.predict(vec.transform(["White men and women who have many white children and teach them.to be white nationalist do more for our cause than these groups"]))
prediction_1 = classifier.predict(vec.transform(["nt know true heard pick number two french speak better english picking one english getting know pretending speak english"]))

# Print results
print("Confusion Matrix:\n", c)
print("Accuracy:", a)

# Print predictions
print("Prediction:",prediction)
print("Prediction 1:", prediction_1)


# In[44]:


from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(train_labels, classifier.predict_proba(train_data)[:,1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(train_labels, classifier.predict_proba(train_data)[:,1])

# Plot precision-recall curve
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Calculate precision, recall, and F1-score
precision, recall, f1_score, support = precision_recall_fscore_support(train_labels, train_predictions, average='binary')

# Print precision, recall, and F1-score
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)


# In[47]:


# Importing pickle to save the model
import pickle
file="C:\\Users\\DELL\\Downloads\\Hate_or_NoHate_Classifier\\Log_Model.pkl"
with open(file,'wb') as f:
    pickle.dump(classifier,f)
with open(file,'rb') as f:
    pickle.load(f)
file_vec="C:\\Users\\DELL\\Downloads\\Hate_or_NoHate_Classifier\\vec.pkl"
with open(file_vec,'wb') as g:
    pickle.dump(vec,g)
with open(file_vec,'rb') as g:
    pickle.load(g)


# In[46]:


df.to_excel('Hate_or_noHate.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




