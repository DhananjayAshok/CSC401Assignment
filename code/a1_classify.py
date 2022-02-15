#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return C.diagonal().sum() / C.sum()


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    n_classes = C.shape[0]
    l = []
    for c in range(n_classes):
        row = C[c, :]
        l.append(row[c]/row.sum())
    return l


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    n_classes = C.shape[0]
    l = []
    for c in range(n_classes):
        col = C[:, c]
        l.append(col[c]/col.sum())
    return l


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    iBest = -1
    bestAcc = -1
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for index in model_dict:
            classifier_name, model = model_dict[index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, predictions)
            acc = accuracy(conf_matrix)
            rec = recall(conf_matrix)
            prec = precision(conf_matrix)
            if acc > bestAcc:
                iBest = index
                bestAcc = acc
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in rec]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in prec]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
        pass
    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    n_examples = [1_000, 5_000, 10_000, 15_000, 20_000]
    big_indices = range(len(X_train))
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for num_train in n_examples:
            name, model = model_dict[iBest]
            indices = np.random.choice(big_indices, num_train, replace=False)
            X_train_sub = X_train[indices]
            y_train_sub = y_train[indices]
            model.fit(X_train_sub, y_train_sub)
            preds = model.predict(X_test)
            conf_matrix = confusion_matrix(y_test, preds)
            acc = accuracy(conf_matrix)
            outf.write(f'{num_train}: {acc:.4f}\n')
        pass
    indices = np.random.choice(big_indices, 1000, replace=False)
    X_1k = X_train[indices]
    y_1k = y_train[indices]
    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    ks = [5, 50]
    datasets = {1: (X_1k, y_1k), 32: (X_train, y_train)}
    model = model_dict[i][1]
    top5 = {}
    p_values_dict = {}
    accuracy_1k = None
    accuracy_full = None

    for k in ks:
        for dataset in datasets:
            X, y = datasets[dataset]
            selector = SelectKBest(f_classif, k=k)
            X_new = selector.fit_transform(X, y)
            X_test_new = selector.transform(X_test)
            pp = selector.pvalues_
            p_values_dict[k] = pp[selector.get_support()]
            if k == 5:
                top5[dataset] = selector.get_support()
                model.fit(X_new, y)
                pred = model.predict(X_test_new)
                conf_matrix = confusion_matrix(y_test, pred)
                acc = accuracy(conf_matrix)
                if dataset == 1:
                    accuracy_1k = acc
                else:
                    accuracy_full = acc
    top_5 = np.where(top5[32])[0]
    print(top_5)
    feature_intersection = []
    indices = np.where(top5[1])[0]
    print(indices)
    for i in indices:
        if i in top_5:
            feature_intersection.append(i)
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        
        # for each number of features k_feat, write the p-values for
        # that number of features:
        for k_feat in ks:
            p_values = p_values_dict[k_feat]
            outf.write(f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')
    return


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    kf = KFold(n_splits=5, shuffle=True)
    acc_dict = {0: [], 1: [], 2: [], 3: [], 4: []}
    for train_index, test_index in kf.split(X_train):
        X_train_small, y_train_small = X_train[train_index], y_train[train_index]
        X_test_small, y_test_small = X_train[test_index], y_train[test_index]
        for key in model_dict:
            name, model = model_dict[key]
            model.fit(X_train_small, y_train_small)
            pred = model.predict(X_test_small)
            conf_matrix = confusion_matrix(y_test_small, pred)
            acc = accuracy(conf_matrix)
            acc_dict[key].append(acc)
    for key in acc_dict:
        acc_dict[key] = np.array(acc_dict[key])
    kfold_accuracies = []
    p_values = []
    for key in acc_dict:
        if key == i:
            kfold_accuracies.append(acc_dict[key].mean())
            continue
        kfold_accuracies.append(acc_dict[key].mean())
        pval = ttest_rel(acc_dict[i], acc_dict[key]).pvalue
        p_values.append(pval)
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        #  for each fold:
        outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')
    return


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    global model_dict
    sgd = SGDClassifier(warm_start=False)
    gaussian = GaussianNB()
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, warm_start=False)
    mlp = MLPClassifier(alpha=0.05, warm_start=False)
    ada = AdaBoostClassifier()
    model_dict = {0: ("SGDClassifier", sgd), 1: ("GaussianNB", gaussian),
                  2: ("RandomForestClassifier", rf), 3: ("MLPClassifier", mlp), 4: ("AdaBoostClassifier", ada)}

    data = np.load(args.input)["arr_0"]
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)

