__author__ = 'zoraida'
__version__ = "1.0"
__email__ = "zoraida@tid.es"
from datasets.consumptions import PowerConsumptions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer


def generate_aggregates(df, country, year):
    # Select the year to work with
    df = df.loc[(df['year'] == str(year - 1)) | (df['year'] == str(year -2))
                | (df['year'] == str(year - 3)) | (df['year'] == str(year))]

    # Select the countries to work with
    df = df.loc[(df['Country'] == country)]

    # Replase missing values by the default missing value representation
    df = df.replace(u'n.a.', np.nan)
    # Fill missing values with previous hour
    df = df.fillna(method='pad', axis=1)
    # Values normalization
    categorical_vars = df[['year', 'Country', 'Day', 'month', 'weekday']].astype(basestring)
    continuous_var = df[[u'01:00:00', u'02:00:00', u'03:00:00', u'04:00:00', u'05:00:00', u'06:00:00', u'07:00:00',
                       u'08:00:00', u'09:00:00', u'10:00:00', u'11:00:00', u'12:00:00', u'13:00:00', u'14:00:00',
                       u'15:00:00', u'16:00:00', u'17:00:00', u'18:00:00', u'19:00:00', u'20:00:00', u'21:00:00',
                       u'22:00:00', u'23:00:00', u'24:00:00']].astype(float)
    consumption_by_day = pd.DataFrame(continuous_var.sum(axis=1))
    consumption_by_day.columns = ['Consumption']
    # country | day | month | weekly | consumption
    return pd.concat([categorical_vars, consumption_by_day], axis=1)


def evaluate (X_train, y_train, X_test, y_test, regressor):

    regressor.fit(X_train, y_train)

    y_pred = regression.predict(X_test)

    print('Coefficients: \n', regression.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f"
          % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regression.score(X_test, y_test))

    return y_pred

def plot_curves(x, y_pred, y_test, x_label, y_label, legend):
    # Plot outputs
    plt.plot(x, y_pred)
    plt.plot(x, y_test)

    plt.legend(legend, loc='upper left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()


if __name__ == "__main__":
    dir_path = "/Users/zoraida/Desktop/TEFCON/all-country-data/hourly"
    pattern = "/Hourly_201*month*.xls"
    # year to predict
    year = 2013
    country = 'ES'
    pc = PowerConsumptions(dir_path, pattern, skiprows=9, maxcolumns=26, hourchange='3B:00:00')
    # get data transformed: country | 01-01-2011 | 01-01-2012 | 01-01-2013 | ... | 31-12-2011 | 31-12-2012 | 31-12-2013
    df = generate_aggregates(pc.df, country, year)

    df = df[df['Day'] != '2012-02-29']

    y_pred_matrix = []
    y_test_matrix = []
    regression = LinearRegression()

    #y array of 1 dimension with a day consumption for all days of the year for Spain.
    y_train = df.loc[(df['year'] == str(year - 1)) | (df['year'] == str(year - 2)) | (df['year'] == str(year - 3))]['Consumption'].values # 2010 2011 2012
    y_test = df.loc[(df['year'] == str(year))]['Consumption'].values # 2013

    # we build the model only using the previous 2 years
    X_train = df.loc[(df['year'] == str(year - 1)) | (df['year'] == str(year - 2)) | (df['year'] == str(year - 3))][['month','year','weekday']].values

    X_test = df.loc[(df['year'] == str(year))][['month', 'year', 'weekday']].values

    vec = OneHotEncoder(sparse=False, categorical_features=[0,2])
    X_train_T = vec.fit_transform(X_train).astype(int)
    X_test_T = vec.transform(X_test).astype(int)

    # Returns prediction of a day consumption for all the countries
    y_pred = evaluate(X_train_T, y_train, X_test_T, y_test, regression)

    # 2013 no es bisiesto, ojo con eso
    plot_curves(range(1, 366), y_pred, y_test, 'Days of the year', country + ' Consumption', ['Predicted', 'Truth'])


