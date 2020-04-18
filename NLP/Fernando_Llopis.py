import numpy as np
import pandas as pd

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
pd.options.display.float_format = '{:.3f}'.format

train_df = pd.read_csv('/Users/fer/Downloads/train.csv')
test_df = pd.read_csv('/Users/fer/Downloads/test.csv')
sample_df = pd.read_csv('/Users/fer/Downloads/sample_submission.csv')

print(train_df.head(20))
print(test_df.head(20))
print(sample_df.head(20))

''' EDA :: Basic statistics  and type analysis '''

print('-------TRAIN--------')
print('Train types\n', train_df.dtypes)
print('Train info\n', train_df.info())
print('Train description\n', train_df.describe())
print('Train shape\n', train_df.shape)

nas_keyword = sum(pd.isnull(train_df['keyword']))
print('nas keyword', nas_keyword)
nas_location = sum((pd.isnull(train_df['location'])))
print('nas location', nas_location)

print('---------------------')

print('------TEST-----')
print('test types\n', test_df.dtypes)
print('test info\n', test_df.info())
print('test description\n', test_df.describe())
print('test shape\n', test_df.shape)

nas_keyword = sum(pd.isnull(test_df['keyword']))
print('nas keyword',nas_keyword)
nas_location = sum((pd.isnull(test_df['location'])))
print('nas location', nas_location)

print('-----------------')

''' Subset keyword and location columns :: for != to NaN'''
train_k = train_df[train_df.keyword.notna()].reset_index()
print('Train set with all observations containing any valid keyword\n', train_k)
train_l = train_df[train_df.location.notna()].reset_index()
print('Train set with all observations containing any valid location\n', train_l)
print(train_l.info())
train_kl = train_df[(train_df.keyword.notna()) & (train_df.location.notna())]
print(train_kl)

''' ReGex '''