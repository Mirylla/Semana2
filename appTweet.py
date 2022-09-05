# -*- coding: utf-8 -*-
# +
from networkx.algorithms import community
import networkx as nx
from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import hydralit as hy
from hydralit import HydraApp
import streamlit as st  
import networkx as nx
import pandas as pd
import numpy as np
import altair as alt
import math
import os
#import plotly.figure_factory as ff
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import sklearn
import markov_clustering as mc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import json
from collections import Counter

import igraph as ig
import louvain
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
except ImportError:
    try:
        import pydot
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        raise ImportError("This example needs Graphviz and either "
                          "PyGraphviz or pydot")

# +
app = hy.HydraApp(title='Science App')

@app.addapp(is_home=True)
def my_home():
    hy.info('DataSet')
        


# -

df_norte=pd.read_csv("C:\\Users\\mirya\\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\df_norteall22.csv", header=0)
df_norte.drop(df_norte.loc[df_norte['retweet_count']<10].index, inplace=True)
df_norte_reset=df_norte.sort_values("retweet_count", ascending=False).reset_index(drop=True)
df_norte_reset


@app.addapp(title='Relación de USUARIOS por NOTICIA')
def app1(): 
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([3, 2, 0.01])
 with column1:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    df_reset2=pd.read_csv("C:\\Users\\mirya\\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\elnorte_weight_names.csv")
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df_reset2,'node1','node2', edge_attr='weight', create_using=Graphtype)
    pos = nx.spring_layout(G)
    plt.figure(10,figsize=(200,80), dpi=60) 
    nx.draw_circular(G_norte_name)
    st.pyplot(plt.show())
    
   
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([3, 2, 0.01])
 with column1:
    # positions for all nodes
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color='blue')
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.9)
    plt.title('Comunidades de usuarios según retweets')
    plt.legend(loc='best')
    plt.axis('off')
    plt.figure(10,figsize=(80,40), dpi=100) 
    st.pyplot(plt.show()) 

 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([3, 2, 0.01])
 with column1:
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=550,cmap=cmap, node_color=list(partition.values()))
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.9)
    # labels
    nx.draw_networkx_labels(G, pos, font_size=4, font_family='sans-serif')
    plt.title('Comunidades de usuarios según retweets')
    plt.figure(10,figsize=(80,40), dpi=100) 
    st.pyplot(plt.show()) 


@app.addapp(title='Relación de NOTICIAS por USUARIOS')
def app2(): 
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([3, 2, 0.01])
 with column1:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    df_elnorte_weight_1_tweets=pd.read_csv("C:\\Users\\mirya\\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\elnorte_weight_tweets.csv")
    Graphtype = nx.Graph()
    G_norte_tweet = nx.from_pandas_edgelist(df_elnorte_weight_1_tweets,'node1','node2', edge_attr='weight', create_using=Graphtype)
    pos = nx.spring_layout(G_norte_tweet)
    plt.figure(10,figsize=(80,40), dpi=100) 
    nx.draw_circular(G_norte_tweet, node_size=300)
    st.pyplot(plt.show())
 
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([3, 2, 0.01])
 with column1:
    # positions for all nodes
    pos = nx.spring_layout(G_norte_tweet)
    nx.draw_networkx_nodes(G_norte_tweet, pos, node_size=250, node_color='blue')
    # edges
    nx.draw_networkx_edges(G_norte_tweet, pos, alpha=0.9)
    plt.title('Comunidades de usuarios según retweets')
    plt.legend(loc='best')
    plt.axis('off')
    plt.figure(10,figsize=(80,40), dpi=100) 
    st.pyplot(plt.show()) 

 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([3, 2, 0.01])
 with column1:
    partition_norte_tweets = community_louvain.best_partition(G_norte_tweet)
    # positions for all nodes
    pos = nx.spring_layout(G_norte_tweet)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition_norte_tweets.values()) + 1)
    # nodes
    nx.draw_networkx_nodes(G_norte_tweet, pos, partition_norte_tweets.keys(), node_size=550,cmap=cmap, node_color=list(partition_norte_tweets.values()))
    # edges
    nx.draw_networkx_edges(G_norte_tweet, pos, alpha=0.9)
    plt.title('Comunidades de usuarios según retweets')
    plt.legend(loc='best')
    plt.axis('off')
    plt.figure(10,figsize=(80,40), dpi=100) 
    st.pyplot(plt.show()) 
    partition_norte_tweet= pd.DataFrame(partition_norte_tweets.values(), index=partition_norte_tweets.keys(), columns=['Community'])
    partition_norte_tweet['Community'].nunique()
    partition_norte_tweet.to_string('partition_norte_tweet.txt',index = True)
    part_com_norte=pd.read_csv('partition_norte_tweet.txt',names=['tweet','comunity'],sep=r'\s+', header=0)
    Comunidad_norte_0=[]
    Comunidad_norte_1=[]
    Comunidad_norte_2=[]
    Comunidad_norte_3=[]


    for i in range(len(part_com_norte)):
        if part_com_norte['comunity'][i] == 0:
            Comunidad_norte_0.append(part_com_norte['tweet'][i])
        elif part_com_norte['comunity'][i] == 1:
            Comunidad_norte_1.append(part_com_norte['tweet'][i])
        elif part_com_norte['comunity'][i] == 2:
            Comunidad_norte_2.append(part_com_norte['tweet'][i])

        else:
            Comunidad_norte_3.append(part_com_norte['tweet'][i])  
            
    comunidad_norte_0_text=df_norte_reset[df_norte_reset['tweet_id'].isin(Comunidad_norte_0)].reset_index(drop=True)
    comunidad_norte_1_text=df_norte_reset[df_norte_reset['tweet_id'].isin(Comunidad_norte_1)].reset_index(drop=True)
    comunidad_norte_2_text=df_norte_reset[df_norte_reset['tweet_id'].isin(Comunidad_norte_2)].reset_index(drop=True)
    comunidad_norte_3_text=df_norte_reset[df_norte_reset['tweet_id'].isin(Comunidad_norte_3)].reset_index(drop=True)
    
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        
        return input_txt 

    comunidad_norte_0_text['text'] = np.vectorize(remove_pattern)(comunidad_norte_0_text['text'], "@[\w]*")
    comunidad_norte_1_text['text'] = np.vectorize(remove_pattern)(comunidad_norte_1_text['text'], "@[\w]*")
    comunidad_norte_2_text['text'] = np.vectorize(remove_pattern)(comunidad_norte_2_text['text'], "@[\w]*")
    comunidad_norte_3_text['text'] = np.vectorize(remove_pattern)(comunidad_norte_3_text['text'], "@[\w]*")
    
    corpus = []
    for i in range(0, len(comunidad_norte_0_text)):
        # Deleting everything which is not characters
        tweet = re.sub(r'[^a-z A-Z]', ' ', comunidad_norte_0_text['text'][i])
        tweet = tweet.lower()
        tweet = re.sub('https\S+', ' ', tweet)
        tweet = re.sub('http\S+', ' ', tweet)
        tweet = re.sub('valladolid', ' ', tweet)
        # Deleting any word which is less than 3-characters mostly those are stopwords
        tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
        tweet = tweet.split()
        ps = PorterStemmer()
        tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('spanish'))]
        tweet = ' '.join(tweet)
        corpus.append(tweet)
    
    all_words = ' '.join([text for text in corpus])
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    st.pyplot(plt.show())

app.run()
