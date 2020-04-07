from sklearn import datasets
import time
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt

wine = datasets.load_wine()

print('Features: ', wine.feature_names)
print('X data\n', wine.data)
print('\nTarget variable:', wine.target_names)
print('\nTarget data: \n', wine.target)

general_method = 1
crossval_method = 0

if general_method == 1:

    X = wine.data
    y = wine.target

    yx = []
    for item in wine.target:
        if item == 2:
            yx.append(0)
        elif item == 1:
            yx.append(1)
        else:
            yx.append(0)

    y = yx

    ''' Splitting '''

    X_train, X_test, y_train, y_test = train_test_split (X, y , test_size=0.20, random_state=234)

    '''Model Selection'''

    svm_model = svm.SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm_model = svm_model.predict(X_test)
    print('This is my linear model:', y_pred_svm_model)

    ''' Model evaluation '''

    print('\nAccuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
    print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model, average='weighted')))
    print('Recall:{:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model, average='weighted')))

    print(' +++++++++++++++ CLASSIFICATION REPORT ++++++++++++++')
    print(classification_report(y_test, y_pred_svm_model))

    print('+++++++++++++ ROC - AUC +++++++++++++++++++++++++++++')
    y_pred_svm_model_proba = svm_model.predict_proba(X_test)[:, 1]
    print(y_pred_svm_model_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_svm_model_proba)

    plt.plot([0, 1], [0, 1],  'k--')
    plt.plot(fpr, tpr, label='SVM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Poitive Rate')
    plt.title('SVM ROC Curve')
    plt.show()

    print(roc_auc_score(y_test, y_pred_svm_model_proba))

if crossval_method == 1:

    X = wine.data
    y = wine.target

    svm = svm.SVC(kernel='linear')
    logreg = LogisticRegression()

    svm_cv_scores = cross_val_score(estimator=svm, X=X, y=y, cv=10, scoring='accuracy') # k-fold = 10
    print('\nScore: ', np.mean(svm_cv_scores))

    svm_cv_predict = cross_val_predict(estimator=svm, X=X, y=y, cv=10)
    print(svm_cv_predict)

    print('\nAccuracy: {:.2f}'.format(metrics.accuracy_score(y, svm_cv_predict)))
    print('Precision: {:.2f}'.format(metrics.precision_score(y, svm_cv_predict, average='weighted')))
    print('Recall:{:.2f}'.format(metrics.recall_score(y, svm_cv_predict, average='weighted')))


