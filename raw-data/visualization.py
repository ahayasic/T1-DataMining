encoder = preprocessing.LabelEncoder()
Y = encoder.fit_transform(dt['class'].values)
X = dt.iloc[:,3:7].values


kNN_acc = []
kNN_f1s = []
kNN_pre = []
kNN_rec = []

nb_acc  = []
nb_f1s  = []
nb_pre  = []
nb_rec  = []

svm_acc = []
svm_f1s = []
svm_pre = []
svm_rec = []


x, x_test, y, y_test = train_test_split(X, Y, test_size=0.3)
kFold = StratifiedKFold(n_splits=10)
for train, test in kFold.split(x, y):
    '''
    kNN = None
    kNN = KNeighborsClassifier(n_neighbors=7)
    kNN.fit(x[train], y[train])
    score = kNN.score(x[test], y[test])
    '''
    '''
    nb = None
    nb = GaussianNB()
    nb.fit(x[train], y[train])
    score = nb.score(x[test], y[test])
    '''
    svm = None
    svm = SVC(C=2.5, kernel='rbf', gamma=2.0)
    svm.fit(x[train], y[train])
    
    y_pred_svm = svm.predict(x[test])
    
    svm_acc.append(accuracy_score(y[test], y_pred_svm))
    svm_f1s.append(f1_score(y[test], y_pred_svm))
    svm_pre.append(precision_score(y[test], y_pred_svm))
    svm_rec.append(recall_score(y[test], y_pred_svm))


    
print("acc: %.3f +/- %.3f\nf1s: %.3f +/- %.3f\n"
      "pre: %.3f +/- %.3f\nrec: %.3f +/- %.3f"  % 
     (np.mean(svm_acc), np.std(svm_acc), np.mean(svm_f1s), np.std(svm_f1s), 
      np.mean(svm_pre), np.std(svm_pre), np.mean(svm_rec), np.std(svm_rec)))