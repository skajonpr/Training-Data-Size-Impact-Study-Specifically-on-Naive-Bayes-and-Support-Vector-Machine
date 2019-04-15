# Training-Data-Size-impact-study

This study is to discover the impact of traning data sizes over classification models included Support Vector Machine and Multi Naive Bayes.

**Program step**
- Start with 800 samples from the dataset, in each round you build a classifier with 400 more samples.
- In each round:
  + Create pipelines for both SVM and NB.
  + Train a classifier using multinomial Naive Bayes model with 5-fold cross validation.
  + Train a classifier using linear support vector machine model with 5-fold cross validation.
  + For each classifier, collect average F1 macro and average AUC: treat label 2 as the positive class.
  + Plot a line chart (two lines, one for each classifier) show the relationship between sample size and F1-score.
  + Plot another line chart to show the relationship between sample size and AUC.
  
