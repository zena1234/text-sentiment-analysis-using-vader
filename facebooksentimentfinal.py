#inlucde libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
# import neurolab as nl
import io
import unicodedata
import numpy as np
import re
import string
#import pybrain as pb
#SkLearn
from numpy import linalg
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
    #a=data.find(strl)
    #if(a==-1):
        #ss=sid.polarity_scores(data)
        #print(data)
        #for k in ss:
           # print(k,ss[k])
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext
with open('kindle.txt', encoding='ISO-8859-2') as f:
    text = f.read()
sent_tokenizer = PunktSentenceTokenizer(text)
sents = sent_tokenizer.tokenize(text)
print(word_tokenize(text))
print(sent_tokenize(text))
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
nltk_tokens = nltk.word_tokenize(text)
for w in nltk_tokens:
    print ("Actual: %s Stem: %s" % (w,porter_stemmer.stem(w)))
from nltk.stem.wordnet import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk_tokens = nltk.word_tokenize(text)
for w in nltk_tokens:
    print ("Actual: %s Lemma: %s" % (w,wordnet_lemmatizer.lemmatize(w)))

text=nltk.word_tokenize(text)
print(nltk.pos_tag(text))

# = tokenizer.tokenize(text)
sid=SentimentIntensityAnalyzer() 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
with open('kindle.txt', encoding='ISO-8859-2') as f:
    for text in f.read().split('\n'):
        print(text)
        scores = sid.polarity_scores(text)
        for key in sorted(scores):
            print('{0}: {1}, '.format(key, scores[key]), end='')
            
    print()
    from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
#from textblob import TextBlob
n_instances = 100
# Each document is represented by a tuple (sentence, label).
# The sentence is tokenized, so it is represented by a list of strings:
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]

# split subjective and objective instances to keep a balanced uniform class distribution
# in both train and test sets

train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs+train_obj_docs
testing_docs = test_subj_docs+test_obj_docs


sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

# simple unigram word features, handling negation
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# apply features to obtain a feature-value representation of our datasets
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

# train the Naive Bayes classifier on the training set
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

# output evaluation results
for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
    print('{0}: {1}'.format(key, value))
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import neurolab as nl
from tkinter import *
import tkinter.messagebox

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class analysis_text():
        # Main function in program
    def center(self, toplevel):
        toplevel.update_idletasks()
        w = toplevel.winfo_screenwidth()
        h = toplevel.winfo_screenheight()
        size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
        x = w/2 - size[0]/2
        y = h/2 - size[1]/2
        toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

    def callback(self):
        if tkinter.messagebox.askokcancel("Quit", "Do you really wish to quit?"):
            self.main.destroy()
        
    def setResult(self, type, res):
        A=0
        B=0
        C=0
        if (type == "neg"):
            self.negativeLabel.configure(text = "አሉታዊ አስተያየት ጽፈዋል : " + str(res) + " % \n") 
            
        elif (type == "neu"):
            self.neutralLabel.configure( text = "የተለመደ ጽሁፍ ጽፈዋል: " + str(res) + " % \n")
     
        elif (type == "pos"):
            self.positiveLabel.configure(text = "አውንታዊ አስተያየት ጽፈዋል : " + str(res) + " % \n")
           
        
    def runAnalysis(self):
        sentences = []
        sentences.append(self.line.get())
        sid = SentimentIntensityAnalyzer()
        for sentence in sentences:
            # print(sentence)
            ss = sid.polarity_scores(sentence)
            if ss['compound'] >= 0.05 : 
                self.normalLabel.configure(text = " አውንታዊ አስተያየት ጽፈዋል: ") 
  
            elif ss['compound'] <= - 0.05 : 
                self.normalLabel.configure(text = "አሉታዊ አስተያየት ጽፈዋል : ") 
  
            else : 
             self.normalLabel.configure(text = " የተለመደ ጽሁፍ ጽፈዋል: ") 
  
            for k in sorted(ss):
                self.setResult(k, ss[k])
                    # print('{0}: {1} \n'.format(k, ss[k]), end='')
        print()

    def editedText(self, event):
        self.typedText.configure(text = self.line.get() + event.char)

    def runByEnter(self, event):
        self.runAnalysis()

    def __init__(self):
        # Create main window
        self.main = Tk()
        self.main.title("facebook sentment analyser")
        self.main.geometry("700x700")
        self.main.resizable(width=FALSE, heiight=FALSE)
        self.main.protocol("WM_DELETE_WINDOW", self.callback)
        self.main.focus()
        self.center(self.main)

        # addition item on window
        self.label1 = Label(text = "type a comment here:")
        self.label1.pack()

        # Add a hidden button Enter
        self.line = Entry(self.main, width=70)
        self.line.pack()

        self.textLabel = Label(text = "\nyou are typing this:", font=("Helvetica", 15))
        self.textLabel.pack()
        self.typedText = Label(text = "", fg = "blue", font=("Helvetica", 20))
        self.typedText.pack()

        self.line.bind("<Key>",self.editedText)
        self.line.bind("<Return>",self.runByEnter)


        self.result = Label(text = "\nSentiment of this comment is", font=("Helvetica", 15))
        self.result.pack()
        self.negativeLabel = Label(text = "", fg = "red", font=("Helvetica", 20))
        self.negativeLabel.pack()
        self.neutralLabel  = Label(text = "", font=("Helvetica", 20))
        self.neutralLabel.pack()
        self.positiveLabel = Label(text = "", fg = "green", font=("Helvetica", 20))
        self.positiveLabel.pack()
        self.normalLabel =Label (text ="", fg ="red", font=("Helvetica", 20))
        self.normalLabel.pack()
        # Run program
myanalysis = analysis_text()
mainloop()
