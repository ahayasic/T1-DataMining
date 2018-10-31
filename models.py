from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np

np.random.seed(1)

dt = pd.read_csv('sdss-dataset.csv')

encoder = preprocessing.LabelEncoder()
Y = encoder.fit_transform(dt['class'].values)
scaler = preprocessing.StandardScaler()
X = pd.concat([dt.iloc[:,1:8], dt['redshift']], axis=1)
newX = scaler.fit_transform(X)

svm_acc = []
svm_f1s = []
svm_pre = []
svm_rec = []

x, x_test, y, y_test = train_test_split(newX, Y, test_size=0.3)
kFold = StratifiedKFold(n_splits=10)
for train, test in kFold.split(x, y):
    svm = None
    svm = SVC(C=7.0, kernel='rbf', gamma='auto')
    svm.fit(x[train], y[train])
   
    y_pred_svm = svm.predict(x[test])
    
    svm_acc.append(accuracy_score(y[test],  y_pred_svm))
    svm_f1s.append(f1_score(y[test],        y_pred_svm, average='macro'))
    svm_pre.append(precision_score(y[test], y_pred_svm, average='macro'))
    svm_rec.append(recall_score(y[test],    y_pred_svm, average='macro'))

    '''
    y_pred_nb = nb.predict(x[test])

    nb_acc.append(accuracy_score(y[test],  y_pred_nb))
    nb_f1s.append(f1_score(y[test],        y_pred_nb, average='macro'))
    nb_pre.append(precision_score(y[test], y_pred_nb, average='macro'))
    nb_rec.append(recall_score(y[test],    y_pred_nb, average='macro'))

    y_pred_svm = svm.predict(x[test])
    
    svm_acc.append(accuracy_score(y[test],  y_pred_svm))
    svm_f1s.append(f1_score(y[test],        y_pred_svm, average='macro'))
    svm_pre.append(precision_score(y[test], y_pred_svm, average='macro'))
    svm_rec.append(recall_score(y[test],    y_pred_svm, average='macro'))
    '''

    '''
    nb = None
    nb = GaussianNB()
    nb.fit(x[train], y[train])
    svm = None
    svm = SVC(C=2.5, kernel='rbf', gamma=2.0)
    svm.fit(x[train], y[train])
    '''

print('Support Vector Machine (RBF Kernel)')
print("acc: %.3f +/- %.3f\nf1s: %.3f +/- %.3f\n"
      "pre: %.3f +/- %.3f\nrec: %.3f +/- %.3f"  % 
     (np.mean(svm_acc), np.std(svm_acc), np.mean(svm_f1s), np.std(svm_f1s), 
      np.mean(svm_pre), np.std(svm_pre), np.mean(svm_rec), np.std(svm_rec)))