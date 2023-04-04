import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def ex_1_a(signal, background):
    length = int(signal.shape[1])
    for i in range(length):
        plt.figure()
        plt.hist(signal[:,i], alpha=0.5, label='Signal', bins=20)
        plt.hist(background[:,i], alpha=0.5, label='Background', bins=20)
        plt.xlabel(f'Feature {i+1}')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.savefig(f"ex_1_a_{i}")


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

def metric_calc(df):
    tp = len(df[(df["BDT val"]==1.0)&(df["Label"]==1.0)])
    tn = len(df[(df["BDT val"]==0.0)&(df["Label"]==0.0)])
    fp = len(df[(df["BDT val"]==0.0)&(df["Label"]==1.0)])
    fn = len(df[(df["BDT val"]==1.0)&(df["Label"]==0.0)])
    return (tp+tn)/(tp+tn+fp+fn)
    
    
    
    
def ex_1_d():
    columns = ["TotalPT", "VertexChisq", "Isolation"]
    signal = np.loadtxt("signal.txt")
    background = np.loadtxt("background.txt")
    signal = signal[:,4:]
    background = background[:,4:]

    #Combine the signal and background into one dataset
    X = np.concatenate((signal, background))
    #Add a label dataset y, which tells the BDT what to aim for (i.e. BDT value should be 1 for signal and 0 for background).
    y = np.concatenate((np.ones(signal.shape[0]),
                        np.zeros(background.shape[0])))

    X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=492)

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=int(0.1*len(X)))
    bdt = AdaBoostClassifier(dt,
                            algorithm='SAMME',
                            n_estimators=100,
                            learning_rate=0.1)

    #Optimise the parameters of the weights using the training sample
    bdt.fit(X_train, y_train)

    #Test the bdt parameters on the test sample
    y_predicted = bdt.predict(X_test)
    y_predicted_2 = bdt.predict(X_train)
    #This can be used to add the BDT and label columns to your dataset.
    df_1 = pd.DataFrame(np.hstack((X_test, y_predicted.reshape(y_predicted.shape[0], -1),y_test.reshape(y_test.shape[0],-1))),
                    columns=columns+['BDT val','Label'])
    df_2 = pd.DataFrame(np.hstack((X_train, y_predicted_2.reshape(y_predicted_2.shape[0], -1),y_train.reshape(y_train.shape[0],-1))),
                    columns=columns+['BDT val','Label'])

    df = pd.concat([df_2,df_1],axis=0,ignore_index=True)
    print("score on test set: ",bdt.score(X_test, y_test))
    print("score on training set: ", bdt.score(X_train, y_train))
    metric = metric_calc(df)
    metric_test = metric_calc(df_1)
    metric_train = metric_calc(df_2)
    print("Metric using the bdt on the whole dataset 3 features: ",metric)
    print("Metric using the bdt on the test dataset 3 features: ",metric_test)
    print("Metric using the bdt on the train dataset 3 features: ",metric_train)
    print("This performs already better than the slicing")


def ex_1_e():
    columns = ["PT1", "PT2", "P1", "P2", "TotalPT", "VertexChisq", "Isolation"]
    signal = np.loadtxt("signal.txt")
    background = np.loadtxt("background.txt")


    #Combine the signal and background into one dataset
    X = np.concatenate((signal, background))
    #Add a label dataset y, which tells the BDT what to aim for (i.e. BDT value should be 1 for signal and 0 for background).
    y = np.concatenate((np.ones(signal.shape[0]),
                        np.zeros(background.shape[0])))

    X_train,X_test, y_train,y_test = train_test_split(X, y, test_size=0.33, random_state=492)

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=int(0.1*len(X)))
    bdt = AdaBoostClassifier(dt,
                            algorithm='SAMME',
                            n_estimators=100,
                            learning_rate=0.1)

    #Optimise the parameters of the weights using the training sample
    bdt.fit(X_train, y_train)

    #Test the bdt parameters on the test sample
    y_predicted = bdt.predict(X_test)
    y_predicted_2 = bdt.predict(X_train)
    #This can be used to add the BDT and label columns to your dataset.
    df_1 = pd.DataFrame(np.hstack((X_test, y_predicted.reshape(y_predicted.shape[0], -1),y_test.reshape(y_test.shape[0],-1))),
                    columns=columns+['BDT val','Label'])
    df_2 = pd.DataFrame(np.hstack((X_train, y_predicted_2.reshape(y_predicted_2.shape[0], -1),y_train.reshape(y_train.shape[0],-1))),
                    columns=columns+['BDT val','Label'])

    df = pd.concat([df_2,df_1],axis=0,ignore_index=True)
    print("score on test set: ",bdt.score(X_test, y_test))
    print("score on training set: ", bdt.score(X_train, y_train))
    metric = metric_calc(df)
    metric_test = metric_calc(df_1)
    metric_train = metric_calc(df_2)
    print("Metric using the bdt on the whole dataset: ",metric)
    print("Metric using the bdt on the test dataset: ",metric_test)
    print("Metric using the bdt on the train dataset: ",metric_train)
    print("This performs already better than the prefious example the accury improved")
    
    
     
def ex_1(signal, background):
    print("Excercise 1.a)")
    ex_1_a(signal, background)
    print("Excercise 1.b)")
    ex_1_b(signal, background)
    print("Excercise 1.c)")
    ex_1_c()
    print("Excercise 1.d)")
    ex_1_d()
    print("Excercise 1.e)")
    ex_1_e()

if __name__ == "__main__":
    signal = np.loadtxt("signal.txt")
    background = np.loadtxt("background.txt")
    ex_1(signal, background)