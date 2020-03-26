'''Here are all my functions '''


def even_or_odd(num): # status :: ok
    if num % 2 == 0:
        print('This number is even ')
    else:
        print('This number is odd')

# create a function that takes a pandas dataframe and returns the statistics summary:

def summary_statistics(dataframe):
    ''' the input should be a pandas dataframe, otherwise (spark Dataframes or
    RDD's would not work '''
    return dataframe.describe()


import pandas as pd

flights_df = pd.read_csv('/Users/fer/Downloads/flights_jan08.csv')
print(flights_df)

dates_ls = []
for year, month, day in zip(flights_df.Year, flights_df.Month, flights_df.DayofMonth):
    date = str(year) + '-' + str(month) + '-' + str(day)
    dates_ls.append(date)

flights_df['Date'] = dates_ls
flights_df.Date = pd.to_datetime(flights_df.Date, format='%Y-%m-%d')
flights_df = flights_df.set_index(flights_df.Date, drop=True).drop(columns=['Date'])

print(flights_df)
print(flights_df.dtypes)
print(type(flights_df.Year))
print(list(flights_df))

def df_to_datetime(url):
    ''' the programmer should download (via pip or conda) the pandas library
    for this method to run '''
    import pandas as pd

    df = pd.read_csv(str(url)) # 1st step :: load url and transforms to pandas DataFrame
    print(df)

    dates_ls = []
    year_s = list(df)
    for year, month, day in zip(df.Year, df.Month, df.DayofMonth):
        date = str(year) + '-' + str(month) + '-' + str(day)
        dates_ls.append(date)

    flights_df['Date'] = dates_ls
    flights_df.Date = pd.to_datetime(flights_df.Date, format='%Y-%m-%d')
    flights_df = flights_df.set_index(flights_df.Date, drop=True).drop(columns=['Date'])

    print(flights_df)
    print(flights_df.dtypes)