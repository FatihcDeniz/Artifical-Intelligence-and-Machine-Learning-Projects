from sklearn.metrics import euclidean_distances
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import string

@st.cache(allow_output_mutation=True) 
def download():
    nltk.download('stopwords')
    nlp = spacy.load("en_core_web_md")
    return nlp

nlp = download()

count_vectorizer = CountVectorizer()
tfid_vectorizer = TfidfVectorizer()

st.title('Plagiarism Checker Python')


embedding_algo = st.selectbox('Embedding Type', options=['Count Vectorizer',"TF-IDF","Word2Vec","Roberta","Bert","Universal Sentence Encoder"])
similarity_metric = st.selectbox('Similarity Metric',options=['Cosine','Jaccard','Euclidean','Longest Common Substring','Hamming Distance'])

first_text = st.text_input(label='Please enter a text:',value = 'ali babanin',key=1)
second_text = st.text_input(label='Please enter a text:',value= "bir ciftligi var ",key = 2)
third_text = st.text_input(label='Please enter a text:',value= "ciftliginde inekleri var ",key = 3)
forth_text = st.text_input(label='Please enter a text:',value= "moo moo diye  ",key = 4)

text_values = [first_text,second_text,third_text,forth_text] 
def clean_text(text): 
    stop_words = stopwords.words("english")
    tokens = text.split() #Splitting text into tokens
    translator = str.maketrans("","",string.punctuation)
    tokens = [word.translate(translator) for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    text = " ".join(tokens)
    return text
text_values = [clean_text(x) for x in text_values]

@st.cache(allow_output_mutation=True)
def load_model(model_name):
    if model_name == 'Roberta':
        model = SentenceTransformer('stsb-roberta-base')
    if model_name == "Bert":
        model = SentenceTransformer('stsb-bert-base')
    if model_name == "Universal Sentence Encoder":
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
        model = hub.load(module_url)
    return model

def heatmap(similarity,annot=True):
    texts = [x[:20] for x in text_values]
    data = pd.DataFrame(similarity,columns=texts,index=texts)
    fig , ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data,cmap='YlGnBu',annot=annot,ax=ax)
    return fig

# Distance  
def sum_squared(x):
    return np.sqrt(np.sum(np.square(x)))

def cosine_similar(x,y):
    return np.dot(x,y) / (sum_squared(x) * sum_squared(y))

def euler(distance):
    return 1/np.exp(distance)

def euclidean_distance(x,y):
    return euler(np.sqrt(sum(np.square(x-y))))

def jaccard_index(x,y):
    intersections = {}
    
    x = [x.lower().split(" ")][0]
    y = [y.lower().split(" ")][0]
    
    for a in x:
        if a in set.intersection(*[set(x),set(y)]) and a != '':
            intersections[a] = 1
        if a not in set.intersection(*[set(x),set(y)]) and a != '':
            intersections[a] = 0
    for b in y:
        if b in set.intersection(*[set(x),set(y)]) and b != '':
            intersections[b] = 1
        if b not in set.intersection(*[set(x),set(y)]) and b != '':
            intersections[b] = 0
    
    intersection = len(set.intersection(*[set(x),set(y)]))
    union = len(set.union(*[set(x),set(y)]))
    intersections = pd.DataFrame(intersections,index=['Text 1', ' Text 2'])
    return intersection / union, intersections

def count_vectorize(documents):
    vectors = count_vectorizer.fit_transform(documents).toarray()
    return vectors

def tfid_vectorize(documents):
    vectors = tfid_vectorizer.fit_transform(documents).toarray()
    return vectors

def word2vec(documents):
    docs = [nlp(doc) for doc in documents]
    vector = []
    for i in range(len(documents)):
        vector.append(docs[i].vector)
    return vector

#"""
#Longes Common Substring is a character-absed similarity measure. Given two strings, it simply
#finds longest string which is substring of given two strings.
#"""

def longest_common_substring(s1,s2):
    n1, n2, length = len(s1), len(s2), 0
    index = [[0 for i in range(n2)]for _ in range(n1)]
    
    for i in range(n1):
        for j in range(n2):
            if (i == 0 or j == 0):
                index[i][j] = 0
            if s1[i] == s2[j]:
                index[i][j] = index[i-1][j-1] + 1
                length = max(length,index[i][j])
            if s1[i] != s2[j]:
                index[i][j] = 0

    return length 

#"""Hamming Distance between two equal size strings measures, minimum numer of replacements required to 
#change one string into the other."""

def Hammming_Distance(s1,s2):
    res = 0
    for i in range(min(len(s1),len(s2))):
        if s1[i] != s2[i]:
            res += 1
    return res


if first_text != " " and second_text != ' ':
    if embedding_algo == 'Count Vectorizer':
        vector = count_vectorize(text_values)
    
    if embedding_algo == "TF-IDF":
        vector = tfid_vectorize(text_values)
    
    if embedding_algo == "Word2Vec":
        vector = word2vec(text_values)
    
    if embedding_algo == "Roberta":
       model = load_model(embedding_algo) 
       vector = model.encode(text_values)
    
    if embedding_algo == "Bert":
        model = load_model(embedding_algo)
        vector = model.encode(text_values)
    
    if embedding_algo == "Universal Sentence Encoder":
        model = load_model("Universal Sentence Encoder")
        vector = model(text_values)


    if similarity_metric == 'Jaccard':
        similarity,figure = jaccard_index(first_text,second_text)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(figure,annot=True,  cmap='YlGnBu',ax=ax,linewidths=2)
        st.pyplot(fig)
    
    if similarity_metric == 'Euclidean':
        similarity_col = []
        for i in range(4):
            similarity_row = []
            for j in range(4):
                similarity_row.append(round(euclidean_distance(vector[i],vector[j]),2))
            similarity_col.append(similarity_row)
        fig = heatmap(similarity_col,annot=True)
        st.pyplot(fig)
        texts = [x[:20] for x in text_values]
        data = pd.DataFrame(similarity_col,columns=[f'Text {x+1}' for x in range(len(text_values))],index=[f'Text {x+1}' for x in range(len(text_values))])
        st.table(data)
    
    if similarity_metric == 'Cosine':
        similarity_col = []
        for i in range(4):
            similarity_row = []
            for j in range(4):
                similarity_row.append(round(cosine_similar(vector[i],vector[j]),2))
            similarity_col.append(similarity_row)
        fig = heatmap(similarity_col)
        st.pyplot(fig)
        texts = [x[:20] for x in text_values]
        data = pd.DataFrame(similarity_col,columns=[f'Text {x+1}' for x in range(len(text_values))],index=[f'Text {x+1}' for x in range(len(text_values))])
        st.table(data)
    
    if similarity_metric == 'Longest Common Substring':
        data = []
        for i in range(len(text_values)):
            data_row = []
            for j in range(len(text_values)):
                data_row.append(longest_common_substring(text_values[i],text_values[j]))
            data.append(data_row)
        data = pd.DataFrame(data,columns=[f'Text {x+1}' for x in range(len(text_values))],index=[f'Text {x+1}' for x in range(len(text_values))])
        fig, ax = plt.subplots(figsize = (10,5))
        sns.heatmap(data,annot=True,  cmap='YlGnBu',ax=ax,linewidths=2)
        st.pyplot(fig)
        st.table(data)

    if similarity_metric == 'Hamming Distance':
        data = []
        for i in range(len(text_values)):
            data_row = []
            for j in range(len(text_values)):
                data_row.append(hamming_distance(text_values[i],text_values[j]))
            data.append(data_row)
        data = pd.DataFrame(data,columns=[f'Text {x+1}' for x in range(len(text_values))],index=[f'Text {x+1}' for x in range(len(text_values))])
        fig, ax = plt.subplots(figsize = (10,5))
        sns.heatmap(data,annot=True,  cmap='YlGnBu',ax=ax,linewidths=2)
        st.pyplot(fig)
        st.table(data)
