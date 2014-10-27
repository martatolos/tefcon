__author__ = 'zoraida'
__version__ = "1.0"
__email__ = "zoraida@tid.es"
from datasets.consumptions import PowerConsumptions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_aggregates(df):
    # do not need them
    del df['month']
    del df['weekday']
    # Select the year to work with
    df = df.loc[(df['year'] == 2013) | (df['year'] == 2012)
                | (df['year'] == 2011) | (df['year'] == 2010)]

    # Select the countries to work with
    df = df.loc[(df['Country'] == 'BE') | (df['Country'] == 'ES')
                | (df['Country'] == 'IT') | (df['Country'] == 'FR')]

    # Replase missing values by the default missing value representation
    df = df.replace(u'n.a.', np.nan)
    # Fill missing values with previous hour
    df = df.fillna(method='pad', axis=1)
    # Values normalization
    categorical_vars = df[['Country', 'Day']]
    continuous_var = df[[u'01:00:00', u'02:00:00', u'03:00:00', u'04:00:00', u'05:00:00', u'06:00:00', u'07:00:00',
                       u'08:00:00', u'09:00:00', u'10:00:00', u'11:00:00', u'12:00:00', u'13:00:00', u'14:00:00',
                       u'15:00:00', u'16:00:00', u'17:00:00', u'18:00:00', u'19:00:00', u'20:00:00', u'21:00:00',
                       u'22:00:00', u'23:00:00', u'24:00:00']].astype(float)
    consumption_by_day = pd.DataFrame(continuous_var.sum(axis=1))
    consumption_by_day.columns = ['Consumption']
    df = pd.concat([categorical_vars, consumption_by_day], axis=1)

    df_dict = {}
    country_values = df['Country'].unique()
    df_dict['Country']  = country_values.tolist()

    for day in sorted(df['Day'].unique()):
        day_values = []
        for country in country_values:
            day_values.append(df.loc[(df['Country'] == country) & (df['Day'] == day)]['Consumption'].values[0])
        df_dict[day] = day_values

    return pd.DataFrame(df_dict)


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
    # year to predict when trainging
    year_train = '2012'
    # year to predict when testing
    year_test = '2013'

    pc = PowerConsumptions(dir_path, pattern, skiprows=9, maxcolumns=26, hourchange='3B:00:00')
    # get data transformed: country | 01-01-2011 | 01-01-2012 | 01-01-2013 | ... | 31-12-2011 | 31-12-2012 | 31-12-2013
    df = generate_aggregates(pc.df)

    # df_dict = {'Country': ['ES', 'FR', 'PO', 'EN'], '01-01-2010':[1, 2, 3, 4], '01-01-2011':[3, 4, 5, 6],
    #            '01-01-2012': [5, 6, 7, 8], '01-01-2013':[7, 8, 9, 10], '02-01-2010':[1, 2, 3, 4],
    #            '02-01-2011':[3, 4, 5, 6], '02-01-2012':[5, 6, 7, 8], '02-01-2013':[7, 8, 9, 10]}
    #
    # df = pd.DataFrame(df_dict)

    del df['2012-02-29']
    # targets are the 365 days of a year (ordered for later ploting)
    target_train = sorted([s for s in df.columns.values.tolist() if year_train in s])

    target_test = sorted([s for s in df.columns.values.tolist() if year_test in s])

    y_pred_matrix = []
    y_test_matrix = []
    regression = LinearRegression()
    for idx, target in enumerate(target_train):
        #y array of 1 dimension with a day consumption for all countries.
        y_train = df[target].values # 2012
        y_test = df[target_test[idx]].values # 2013
        day_month = target[4:]

        # we build the model only using the previous 2 years
        X_train_col_names = [str(int(year_train) -1) + day_month, str(int(year_train) -2) + day_month]
        X_train = df[X_train_col_names].values

        X_test_col_names = [str(int(year_test) -1) + day_month, str(int(year_test) -2) + day_month]
        X_test = df[X_test_col_names].values
        # Returns prediction of a day consumption for all the countries
        y_pred = evaluate(X_train, y_train, X_test, y_test, regression)
        y_pred_matrix.append(y_pred)
        y_test_matrix.append(y_test)

    y_pred_matrix = np.asarray(y_pred_matrix).T
    y_test_matrix = np.asarray(y_test_matrix).T

    countries = df['Country'].values
    for idx, y_test in enumerate(y_test_matrix):
        # 2013 no es bisiesto, ojo con eso
        plot_curves(range(1, 366), y_pred_matrix[idx], y_test, 'Days of the year', countries[idx] + ' Consumption',
                    ['Predicted', 'Truth'])


