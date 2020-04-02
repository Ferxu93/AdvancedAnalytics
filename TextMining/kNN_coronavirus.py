import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split # scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:.3f}'.format

covid_19_df = pd.read_csv('/Users/fer/Downloads/covid_19_clean_complete.csv')
print(covid_19_df)

covid_19_df['Date'] = pd.to_datetime(covid_19_df.Date, format='%m/%d/%Y', infer_datetime_format=True)
covid_19_df.index = covid_19_df.Date
covid_19_df = covid_19_df.drop(columns=['Date'])
print(covid_19_df)

'''
problem
============
y = Recovered (numeric)
X = [X1, X2, ... Xn] (numeric values)

'''


selected_features = ['Lat', 'Long', 'Confirmed', 'Deaths']
label = []
for recover in covid_19_df.Recovered:
    if recover == 0:
        label.append(0)
    else:
        label.append(1)

covid_19_df['Label'] = label
print(covid_19_df)
print(covid_19_df.describe())

''' Data splitting ::  Trains and Test sets '''
splitting_activation = 1
if splitting_activation == 1:
    X = covid_19_df[selected_features] # 1+3
    y = covid_19_df.Label # 2+4

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6789) # 1, 3, 2, 4

''' Model Selection and Instantiation '''

logreg_activation = 1
if logreg_activation == 1:
    logreg = LogisticRegression() # Instantiate
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)# 5
    print('\nThe predicted (box 5): ', y_pred)

''' Model evaluation '''
Cnf_Matrix = confusion_matrix(y_test, y_pred)
print('This is the raw Confusion Matrix:\n', Cnf_Matrix)