__author__ = 'zoraida'
__version__ = "1.0"
__email__ = "zoraida@tid.es"
from datasets.consumptions import PowerConsumptions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


def generate_aggregates(df):
    del df['Day']
    # Select the year to work with
    df = df[df['year'] == 2013]

    # ceros = hconsum[(hconsum[u'01:00:00'] == (0.0))|(hconsum[u'02:00:00'] == (0.0))|(hconsum[u'03:00:00'] == (0.0))|
    # (hconsum[u'04:00:00'] == (0.0))|(hconsum[u'05:00:00'] == (0.0))|(hconsum[u'06:00:00'] == (0.0))|
    # (hconsum[u'07:00:00'] == (0.0))|(hconsum[u'08:00:00'] == (0.0))|(hconsum[u'09:00:00'] == (0.0))|
    # (hconsum[u'10:00:00'] == (0.0))|(hconsum[u'11:00:00'] == (0.0))|(hconsum[u'12:00:00'] == (0.0))|
    # (hconsum[u'13:00:00'] == (0.0))|(hconsum[u'14:00:00'] == (0.0))|(hconsum[u'15:00:00'] == (0.0))|
    # (hconsum[u'16:00:00'] == (0.0))|(hconsum[u'17:00:00'] == (0.0))|(hconsum[u'18:00:00'] == (0.0))|
    # (hconsum[u'19:00:00'] == (0.0))|(hconsum[u'20:00:00'] == (0.0))|(hconsum[u'21:00:00'] == (0.0))|
    # (hconsum[u'22:00:00'] == (0.0))|(hconsum[u'23:00:00'] == (0.0))|(hconsum[u'24:00:00'] == (0.0))]
    # print ceros

    # Replase missing values by the default missing value representation
    df = df.replace(u'n.a.', np.nan)

    # Fill missing values with previous hour
    df = df.fillna(method='pad', axis=1)

    # Values normalization
    categorical_vars = df[['Country', 'year', 'month', 'weekday']]
    continuous_var = df[[u'01:00:00', u'02:00:00', u'03:00:00', u'04:00:00', u'05:00:00', u'06:00:00', u'07:00:00',
                       u'08:00:00', u'09:00:00', u'10:00:00', u'11:00:00', u'12:00:00', u'13:00:00', u'14:00:00',
                       u'15:00:00', u'16:00:00', u'17:00:00', u'18:00:00', u'19:00:00', u'20:00:00', u'21:00:00',
                       u'22:00:00', u'23:00:00', u'24:00:00']].astype(float)
    continuous_var_sum = continuous_var.sum(axis=1)
    continuous_var = continuous_var.div(continuous_var_sum, axis=0)

    df = pd.concat([categorical_vars, continuous_var], axis=1)
    df = df.groupby(by=['Country', 'year', 'month', 'weekday'], sort=False).mean()
    return df

if __name__ == "__main__":
    dir_path = "/Users/zoraida/Desktop/TEFCON/all-country-data/hourly"
    pattern = "/Hourly*month*.xls"

    pc = PowerConsumptions(dir_path, pattern, skiprows=9, maxcolumns=26, hourchange='3B:00:00')
    print pc.df['Country'].unique()
    df = generate_aggregates(pc.df)


    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    labels_ = kmeans.fit_predict(df.values)
    print labels_