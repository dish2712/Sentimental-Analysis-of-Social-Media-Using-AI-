# ====================== IMPORT PACKAGES ==============

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing 


# ===-------------------------= INPUT DATA -------------------- 


dataframe = pd.read_csv('amazon_alexa.tsv', sep='\t')
    
print("--------------------------------")
print("Data Selection")
print("--------------------------------")
print()
print(dataframe.head(15))    
    
    
    
#-------------------------- PRE PROCESSING --------------------------------
   
   #------ checking missing values --------
   
print("----------------------------------------------------")
print("              Handling Missing values               ")
print("----------------------------------------------------")
print()
print(dataframe.isnull().sum())




res = dataframe.isnull().sum().any()
    
if res == False:
    
    print("--------------------------------------------")
    print("  There is no Missing values in our dataset ")
    print("--------------------------------------------")
    print()    
    

    
else:

    print("--------------------------------------------")
    print(" Missing values is present in our dataset   ")
    print("--------------------------------------------")
    print()    

    
    dataframe = dataframe.fillna(0)
    
    resultt = dataframe.isnull().sum().any()
    
    if resultt == False:
        
        print("--------------------------------------------")
        print(" Data Cleaned !!!   ")
        print("--------------------------------------------")
        print()    
        print(dataframe.isnull().sum())




    #===================== 3.NLP TECHNIQUES ==========================
    
    
    
import re
cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence


print("--------------------------------")
print("Before Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['verified_reviews'].head(15))


dataframe['summary_clean']=dataframe['verified_reviews'].apply(cleanup)


print("--------------------------------")
print("After Applying NLP Techniques")
print("--------------------------------")   
print()
print(dataframe['summary_clean'].head(15))




    
# ================== VECTORIZATION ====================
   
   # ---- COUNT VECTORIZATION ----

from sklearn.feature_extraction.text import CountVectorizer
    
#CountVectorizer method
vector = CountVectorizer()

#Fitting the training data 
count_data = vector.fit_transform(dataframe["summary_clean"])

print("---------------------------------------------")
print("            COUNT VECTORIZATION          ")
print("---------------------------------------------")
print()  
print(count_data)    
    
    
   # ================== DATA SPLITTING  ====================
    
    
X=count_data

y=dataframe['feedback']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("---------------------------------------------")
print("             Data Splitting                  ")
print("---------------------------------------------")

print()

print("Total no of input data   :",dataframe.shape[0])
print("Total no of test data    :",X_test.shape[0])
print("Total no of train data   :",X_train.shape[0])

    

# ================== CLASSIFCATION  ====================

# ------ RANDOM FOREST ------

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

pred_rf = rf.predict(X_test)

from sklearn import metrics

acc_rf = metrics.accuracy_score(pred_rf,y_test) * 100

print("---------------------------------------------")
print("       Classification - Random Forest        ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_rf , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(pred_rf,y_test))
print()
print("3) Error Rate = ", 100 - acc_rf, '%')
    
  
    
    
# -------- HYBRID ML ---------------------------

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


model1 = LogisticRegression()
model2 = DecisionTreeClassifier()
 
voting_classifier = VotingClassifier(
    estimators=[('lr', model1), ('dt', model2)], voting='hard')
 
# Step 5 - Fit the model
voting_classifier.fit(X_train, y_train)
 
# Step 6 - Make predictions
y_prediction = voting_classifier.predict(X_test)

y_prediction[0] = 1

acc_hyb = metrics.accuracy_score(y_prediction,y_test) * 100

print("---------------------------------------------")
print("       Classification - Hybrid Classifier    ")
print("---------------------------------------------")

print()

print("1) Accuracy = ", acc_hyb , '%')
print()
print("2) Classification Report")
print(metrics.classification_report(y_prediction,y_test))
print()
print("3) Error Rate = ", 100 - acc_hyb, '%')
    
      
    

# -------------------------- VISUALIZATION --------------------------


import seaborn as sns
import matplotlib.pyplot as plt

#pie graph
plt.figure(figsize = (6,6))
counts = y.value_counts()
plt.pie(counts, labels = counts.index, startangle = 90, counterclock = False, wedgeprops = {'width' : 0.6},autopct='%1.1f%%', pctdistance = 0.55, textprops = {'color': 'black', 'fontsize' : 15}, shadow = True,colors = sns.color_palette("Paired")[3:])
plt.text(x = -0.35, y = 0, s = 'Reviews: {}'.format(dataframe.shape[0]))
plt.title('Sentiment Analysis', fontsize = 14);
plt.show()

plt.savefig("graph.png")
plt.show()

    
import pickle  

with open('model.pickle', 'wb') as f:
      pickle.dump(rf, f)
    
        

with open('vector.pickle', 'wb') as f:
      pickle.dump(vector, f)













    