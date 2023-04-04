import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

def ex_1_a(signal, background):
    length = int(signal.shape[1])
    for i in range(length):
        plt.figure()
        plt.hist(signal[:,i], alpha=0.5, label='Signal', bins=20)
        plt.hist(background[:,i], alpha=0.5, label='Background', bins=20)
        plt.xlabel(f'Feature {i+1}')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.show()


def fisher_score(signal, background):
    merged = np.concatenate((signal, background))
    rows, cols = signal.shape
    n = rows
    mu = np.mean(merged)
    sigma = np.std(merged)
    mean_s = np.mean(signal, axis=0)
    mean_b = np.mean(background, axis=0)
    fisher = n*((mean_s-mu)**2+(mean_b-mu)**2)/sigma**2
    return fisher


def ex_1_b(signal, background):
    features = np.asarray(["PT1", "PT2", "P1", "P2", "TotalPT", "VertexChisq", "Isolation"])
    f_score = fisher_score(signal, background)
    index = np.flip(f_score.argsort(axis=0))
    ranked_score = np.take_along_axis(f_score, index, axis=0) 
    ranked_features = np.take_along_axis(features, index, axis=0)
    
    index2 = np.repeat(np.atleast_2d(index),10000, axis=0)
    ranked_signal = np.take_along_axis(signal, index2, axis=1)
    ranked_background = np.take_along_axis(background, index2, axis=1)
    print("Ranked features:")
    print(ranked_features)
    print("Ranked Scores:")
    print(ranked_score)


def ex_1_c():
    signal_df = pd.read_csv('signal.txt', sep=" ")
    columns = ["PT1", "PT2", "P1", "P2", "TotalPT", "VertexChisq", "Isolation"]
    signal_df.columns = columns
    background_df = pd.read_csv('background.txt', sep=" ")
    background_df.columns = columns
    best_three = ['TotalPT','Isolation', 'VertexChisq']
    sig = signal_df[best_three]
    back = background_df[best_three]
    back_max = [back[i].max() for i in back]
    back_min = [back[i].min() for i in back]
    sig_max = [sig[i].max() for i in sig]
    sig_min = [sig[i].min() for i in sig]
    mix_max = [max(els) for els in zip(back_max, sig_max)]
    mix_min = [min(els) for els in zip(back_min, sig_min)]
    d_a = (mix_min[0], mix_max[0])
    d_b = d_a
    d_c = (mix_min[1], mix_max[1])
    d_d = d_c
    d_e = (mix_min[2], mix_max[2])
    d_f = d_e
    bnd = (d_a, d_b, d_c, d_d, d_e, d_f)
    def minimize_metric(vals):
        a, b, c, d, e, f = vals
        selection_cut = '@a < TotalPT < @b and @c < Isolation < @d and @e < VertexChisq < @f'    
        signal_select = sig.query(selection_cut)
        background_select = back.query(selection_cut)
        signal_efficiency = len(signal_select)*1.0/len(sig)
        background_efficiency = len(background_select)*1.0/len(back)
        TP = len(sig)*signal_efficiency
        FP = len(sig)*(1-signal_efficiency)
        TN = len(back)*(1-background_efficiency)
        FN = len(back)*background_efficiency

        return 1 - (TP+TN)/(TP+FP+TN+FN)

    x0 = [2012, 35000, -1.9, 0.62, 0.031, 29]
    res = minimize(minimize_metric, x0, method='nelder-mead', bounds=bnd, tol=1e-6)
    print("Excercise 1.c)") 
    print("Minimized values for cutting the tree highest features:") 
    print(res.x)  
    print(f"Resulting in a max accuricy of {1-res.fun}") 


def ex_1_d():
    columns = ["PT1", "PT2", "P1", "P2", "TotalPT", "VertexChisq", "Isolation"]
    signal = np.loadtxt("signal.txt")
    background = np.loadtxt("background.txt")

    #Combine the signal and background into one dataset
    X = np.concatenate((signal, background))
    #Add a label dataset y, which tells the BDT what to aim for (i.e. BDT value should be 1 for signal and 0 for background).
    y = np.concatenate((np.ones(signal.shape[0]),
                        np.zeros(background.shape[0])))

    #VERY IMPORTANT!!! Split the sample into a testing and training sample, you cannot test your BDT on the same sample as that will bias the result
    from sklearn.model_selection import train_test_split
    X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.25, random_state=492)

    #Make a BDT
    dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=int(0.01*len(X)))
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
    
    
def ex_1(signal, background):
    print("Excercise 1.a)")
    #ex_1_a(signal, background)
    print("Excercise 1.b)")
    #ex_1_b(signal, background)
    #ex_1_c()
    ex_1_d()

"""
#Here we read the txt file as a pandas 'DataFrame'
signal_df = pd.read_csv('signal.txt', sep=" ")
#This sets the label of each feature
columns = ["PT1", "PT2", "P1", "P2", "TotalPT", "VertexChisq", "Isolation"]
signal_df.columns = columns

background_df = pd.read_csv('background.txt', sep=" ")
background_df.columns = columns


#Lets calculate what the accuracy is before any more selection
metric_init = len(signal)/(len(background)+len(signal))

print('Inital metric value is ',metric_init)



#Now we try a selection, cutting the data for values of the vertex chisq < 4.
selection_cut = 'VertexChisq < 4'

#Reduce the dataset using the query command (for arrays can use array.select)
signal_select = signal_df.query(selection_cut)
background_select = background_df.query(selection_cut)

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
"""
if __name__ == "__main__":
    signal = np.loadtxt("signal.txt")
    background = np.loadtxt("background.txt")
    ex_1(signal, background)
