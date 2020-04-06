from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()
print(cancer)

print(cancer['feature_names'])
print(cancer['target']) #
print(cancer['target_names'])
print(cancer['data']) #

''' Splitting between Train and test sets '''

Splitting_activation = 1
if Splitting_activation == 1:

    X = cancer['data'] # predictive variables
    y = cancer['target'] # 1 malignante and 0 benignant

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=235)

''' Model selection '''

svm_model = svm.SVC(kernel='linear') # instantiate
svm_model.fit(X_train, y_train)
y_pred_svm_model= svm_model.predict(X_test)
print('This is my linear model:\n',y_pred_svm_model)

print('Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model)))
print('Recall: {:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model)))

svm_model = svm.SVC(kernel='poly')
svm_model.fit(X_train, y_train)
y_pred_svm_model= svm_model.predict(X_test)
print('This is my Polinomial model:\n',y_pred_svm_model)

print('Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model)))
print('Recall: {:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model)))

svm_model = svm.SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm_model= svm_model.predict(X_test)
print('This is my Radial Basis model:\n',y_pred_svm_model)

print('Accuracy: {:.2f}'.format(metrics.accuracy_score(y_test, y_pred_svm_model)))
print('Precision: {:.2f}'.format(metrics.precision_score(y_test, y_pred_svm_model)))
print('Recall: {:.2f}'.format(metrics.recall_score(y_test, y_pred_svm_model)))




