import numpy as np
from math import sqrt

#Firstly we load the data, if you don't have pandas install you can load it as a numpy array:
signal = np.loadtxt("signal.txt")
background = np.loadtxt("background.txt")

#However, if you have pandas installed, its much nicer to use pandas DataFrames for easier manipulation
import pandas as pd

#Here we read the txt file as a pandas 'DataFrame'
signal = pd.read_csv('signal.txt', sep=" ")


#This sets the label of each feature
columns = ["PT1","PT2","P1","P2","TotalPT","VertexChisq","Isolation"]
signal.columns = columns 


background = pd.read_csv('background.txt', sep=" ")
background.columns = signal.columns

print(signal)
print(background)

'''
#Lets calculate what the accuracy is before any more selection
metric_init = len(signal)/(len(background)+len(signal))

print('Inital metric value is ',metric_init)
'''


'''
#Now we try a selection, cutting the data for values of the vertex chisq < 4.
selection_cut = 'VertexChisq < 4'

#Reduce the dataset using the query command (for arrays can use array.select)
signal_select = signal.query(selection_cut)
background_select = background.query(selection_cut)

#Calculate the signal and background efficiency using the ratio of DataFrame sizes
signal_efficiency = len(signal_select)*1.0/len(signal)
background_efficiency = len(background_select)*1.0/len(background)

print('signal efficiency is',signal_efficiency)
print('background efficiency is',background_efficiency)

#Calculate the metric 

TP = len(signal)*signal_efficiency
FP = len(signal)*(1-signal_efficiency)
TN = len(background)*(1-background_efficiency)
FN = len(background)*background_efficiency

metric = (TP+TN)/(TP+FP+TN+FN)

print('Metric after selection is',metric)
'''


'''
#The following code could be used to train a BDT

#We load the input as normal numpy arrays
signal = np.loadtxt("signal.txt")
background = np.loadtxt("background.txt")

#Combine the signal and background into one dataset
X = np.concatenate((signal, background))
#Add a label dataset y, which tells the BDT what to aim for (i.e. BDT value should be 1 for signal and 0 for background).
y = np.concatenate((np.ones(signal.shape[0]),
                    np.zeros(background.shape[0])))

#VERY IMPORTANT!!! Split the sample into a testing and training sample, you cannot test your BDT on the same sample as that will bias the result
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X, y,
                                                  test_size=0.5, random_state=492)

#Import BDT from sci-learn libaray
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

#Make a BDT
dt = DecisionTreeClassifier(max_depth=2,
                            min_samples_leaf=int(0.01*len(X)))
bdt = AdaBoostClassifier(dt,
                         algorithm='SAMME',
                         n_estimators=50,
                         learning_rate=0.1)

#Optimise the parameters of the weights using the training sample
bdt.fit(X_train, y_train)


#Test the bdt parameters on the test sample
y_predicted = bdt.predict(X_test)
#This can be used to add the BDT and label columns to your dataset.
df = pd.DataFrame(np.hstack((X_test, y_predicted.reshape(y_predicted.shape[0], -1),y_test.reshape(y_test.shape[0],-1))),
                  columns=columns+['BDT val','Label'])
print(df)
'''
