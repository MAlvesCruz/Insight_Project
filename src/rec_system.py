
#%% Load libraries
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyemd
import random
import re
import seaborn as sns
import sklearn as skl
import spacy 
import string

from numpy import nan as NA
from numpy import *
from pylab import *
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk import download

#%% Setup options
# Natural Language Tool Kit (NLTK)
nltk.download('stopwords')      # download stopwords list from NLTK.
nltk.download('crubadan')       #
nltk.download('rslp')           #
# Download stopwords list from Spacy
nlp = spacy.load('pt_core_news_sm')
# Increase size of pandas columns display
pd.options.display.max_colwidth = 500

#%% Import pre-trained word2vec models for Portuguese-Br
# These models were published by a group in the computer science departement 
# at University of Sao Paulo (https://arxiv.org/abs/1708.06025)
# http://nilc.icmc.usp.br/embeddings
path_model = './data/'
#file_model = 'skip_s300_wo2vec.txt'
file_model = 'skip_s50_wo2vec.txt'
model = KeyedVectors.load_word2vec_format(path_model + file_model)  

#%%
path_data = './data/'
file_data = 'Concat_questions'
archive_questions = pd.read_csv(path_data + file_data)

#%% Getting stop_words from two sources
# NLTK stop words
nltk_stop_words_pt = set(stopwords.words('portuguese'))
# Spacy stop words
spacy_stop_words_pt = set(spacy.lang.pt.stop_words.STOP_WORDS)
# Adhoc stop words
adhoc_stop_words = set(['tirála','iching','terennce','workflowly','crystalreportviewer','webform','ledface',
                    'facilmentesem','cocriações','açãoaventuradramaterror','encapsulate','tibased',
                    'extrusion','tecnologiaeventos','outubroalguem','seleciondo','mantêlo','educálas',
                    'leválas','automaticos','nowadaysys','produtoforma','semisintético',
                    'urbanascomo','tijolocimentoconcreto','tonarse','discapacidade','decembrode',
                    'vittorios','carrabas','vlingo','dribbble',
                    'algum','alguns','alguém','onde','quais','melhor','gostaria','sobre','existe',
                    'preciso','pra','devo','comprar','paulo','encontrar','alguma','conhece',
                    'poderia','queria','consegui','consigo','criar','nagativas',
                    'usálos','macetepara','laciálos','fazêlas','trabalhande','usálo',
                    'estragálariscálaetc','tranco','canelars','adoreiiii','servílo','idima',
                    'ubelandia','campinassp','enxáguo','oa', 'parceiroa', 'elea',
                    'sao','distraílos','esterilizálas','erguêlo','guidão','uberlândiamg',
                    'caloi','guarujá','pitangueiras','cartasmanipulação','sentemse','seriaserá',
                    'açoinox','victorinox','aliena','présal','sobrinhoa','educáloa','preparáloa',
                    'guarujásp','muitomas','malcomo','ajudarlo','guiálo','adequála',
                    'matálos','sociiotecnicos','singia','executala','mts','ganhapão',
                    'comprálo','conteudoo','construíla','crisefinanceiroeconômica',
                    'aplicálo','disponiza','bloquealo','alguem','diferenciála','pareceme','cadei','embedar',
                    'ajudar'])
# Combine stop word sets
stop_words = list((nltk_stop_words_pt.union(spacy_stop_words_pt)).union(adhoc_stop_words))

#%%
#----------------------------------------------------------------------------------------------------------------
def pre_processing(sentence):
    '''Prepare sentences for the model.
    
    Description.
    This function performs the standard pre-processing steps before applying
    NLP:
        1) Removel of numbers and characters
        2) Removel of url type strings
        3) Removal of punctuation
        4) Lowercasing
        5) Tokenization
        6) Removal of stop words
    
    Parameters
    ----------------------------
    sentence : str
        Question from the website that will be compared to the archive questions in the database.
    
    Returns
    ----------------------------
    out : list
        Clean version of the input question.
    
    '''
    # Removes numbers
    sentence = re.sub(r'\d+', '', sentence)
    # Removes url like things
    url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'
    sentence = re.sub(url_pattern, ' ', sentence) 
    # Standardize white space
    sentence = re.sub(r'\s+', ' ', sentence)
    # Removes punctuation
    sentence=sentence.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    sentence = sentence.lower() 
    # Divide
    sentence = sentence.strip() 
    # Tokenize
    wpt = nltk.WordPunctTokenizer()
    tokens = wpt.tokenize(sentence)
    # Remove stopwords
    sentence = [w for w in tokens if w not in stop_words] 

    # Rejoin sentence
    out = ' '.join(sentence)
    
    return out
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
# This function calculates the cosine similarity between two lists of words
def calculo_similarity(sentence):
    '''Calculate cosine similarity.
    
    Description.
    
    Parameters
    ----------------------------
    sentence : str   
        Recent question from the website that will be compared to the 
        archive questions in the database.
    archive_questions : DataFrame
        DataFrame containing all the previous questions from the archive that 
        were answered by at leat one user.
    
    Returns
    ----------------------------
    sentence_similarities : list
        List containing the cosine similarity between a Question and all the 
        archive ones.
    
    '''
    # Column `words` from `archive_questions` DataFrame
    clean_words = archive_questions.words

    # Calculate cosine-similarity between sentence and each question in archive
    sentence_similarities = []
    for i in archive_questions.index:
        try:
            dist = model.n_similarity(clean_words[i].split(), sentence)
        except:
            #TODO create log of exceptions to handle errors
            dist = -9.99
        sentence_similarities.append(dist)
        
    return sentence_similarities

#----------------------------------------------------------------------------------------------------------
# This function finds the most similar questions in the data base
def finding_match(sentence, min_sim=.35, max_questions=5):
    '''Find top similar questions.
    
    Description.
    This function sort the questions in the archive by their similarity to the 
    question recently posted to the website and find the top questions.
    
    Parameters
    ----------------------------
    sentence : str
        Recent question from the website that will be compared to the 
        archive questions in the database.
    archive_questions : DataFrame
        DataFrame containing all the previous questions from the archive that 
        were answered by at leat one user.
    min_sim : float
        Threshold value for the similarity.
    max_questions : int
        Maximum number of questions to be shown.
    
    Returns
    ----------------------------
    lista_top : DataFrame
        Contains the top questions and their "properties" (id, similarity, and
        number of users).
    
    '''
    # Pre-processing of question fed to the webapp
    sentence = pre_processing(sentence)
    sentence = sentence.split()

    # Calculate the similarity between the webapp question and all the questions in the archive
    sim_list = calculo_similarity(sentence)                     # contains the similarity values
    sim_list = pd.DataFrame(sim_list, columns=['similarity'], index=archive_questions.index) # convert list to a dataframe 
    
    # Concatenate the archive questions datFrame with the similarities dataFrame
    arc_questions_sim = pd.concat([archive_questions, sim_list], axis=1)
    # Group the similarity by question_id
    top_questions = arc_questions_sim.groupby('question_id').max()['similarity']
    # Sort to find the largest values of similarities
    top_questions = top_questions.sort_values(ascending=False)
    
    # Select either the top (max_questions) or those with similarities larger 
    # than a threshold. 
    top_questions = top_questions[top_questions>=min_sim]
    if len(top_questions) > max_questions:
        top_questions = top_questions[0:max_questions]
    
    # Create list with the top questions, the number of users who answered each
    # of the top questions, the similarity values and the question_id
    top_data = []
    for idx in top_questions.index:
        top_data.append([arc_questions_sim[arc_questions_sim.question_id == idx]['body_y'].drop_duplicates().iloc[0],
                         arc_questions_sim[arc_questions_sim.question_id == idx]['user_id'].count(),
                         arc_questions_sim[arc_questions_sim.question_id == idx]['similarity'].drop_duplicates().iloc[0],
                         arc_questions_sim[arc_questions_sim.question_id == idx]['question_id'].drop_duplicates().iloc[0]])
    
    # Convert list to DataFrame
    return pd.DataFrame(top_data, columns=['body','count_user','similarity','question_id'])
