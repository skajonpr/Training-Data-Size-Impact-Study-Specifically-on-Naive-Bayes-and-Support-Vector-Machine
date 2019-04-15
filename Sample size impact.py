import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm

# Define a function to analyze impact of sample sizes.
def impact_of_sample_size(dataset_file):
    # import training data.
    data=pd.read_csv(dataset_file,header=0)
    # create a loop to get all sample sizes 
    # Starting with 800 samples from the dataset, in each round, build a classifier with 400 more samples.
    sizes = np.zeros(19, dtype = int)
    sizes[0] = 800
    for i in range(18):
        sizes[i+1] = sizes[i]+400
    # Difine lists to store F1 score and AUC for each samples.    
    avg_f1_macro = []
    avg_auc = []
    # Create a loop to process 5 fold cross validation for each sample size.
    # and store F1 score and AUC.
    for size in sizes:
        # Define Pipelines of Naive bayes and Support vector machine classification.
        NB_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words="english")),
        ('clf', MultinomialNB())])
    
        SVM_classifier = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words="english")),
        ('clf', svm.LinearSVC())])
    
        # Finding F1 scores of 5 fold cross validation.
        f1_macro_NB_scores = cross_val_score(NB_classifier ,data["text"][0:size] , data["label"][0:size],\
                                cv=5, scoring='f1_macro')
        f1_macro_SVM_scores = cross_val_score(SVM_classifier ,data["text"][0:size] , data["label"][0:size],\
                                cv=5, scoring='f1_macro')
        
        # treating class '2' as the possitive label.
        binary_y = np.where(data["label"][0:size] ==2 ,1 ,0)
        
        # Finding AUC of 5 fold cross validation.
        roc_auc_NB_scores = cross_val_score(NB_classifier ,data["text"][0:size] , binary_y,\
                                cv=5, scoring='roc_auc')
        roc_auc_SVM_scores = cross_val_score(SVM_classifier ,data["text"][0:size] , binary_y,\
                                cv=5, scoring='roc_auc')
        
        # Storing average F1 score and AUC.
        avg_f1_macro.append( (size, f1_macro_NB_scores.mean(), f1_macro_SVM_scores.mean()) )
        avg_auc.append( (size, roc_auc_NB_scores.mean(), roc_auc_SVM_scores.mean()) )
    # Create dataframes for F1 score and AUC.    
    f1_df = pd.DataFrame(avg_f1_macro, columns = ['sizes', 'f1_nb' , 'f1_svm'])        
    auc_df = pd.DataFrame( avg_auc, columns = ['sizes', 'auc_nb' , 'auc_svm'])
    
    # Plot a line chart of each classifier showing the relationship between sample sizes and F1-score.
    f1_df.plot(kind='line',x='sizes',y=['f1_svm', 'f1_nb']);
    plt.show()
    plt.savefig("f1_score.png")
    # Plot a line chart of each classifier showing the relationship between sample sizes and AUC.
    auc_df.plot(kind='line',x='sizes',y=['auc_svm', 'auc_nb']);
    plt.show()
    plt.savefig("AUC.png")

impact_of_sample_size("train_large.csv") 