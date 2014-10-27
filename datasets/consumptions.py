"""
This module provides functions to parse data from https://www.entsoe.eu/ to a Pandas DataFrame
"""
__author__ = 'mtolos'
__version__ = "1.0"
__email__ = "mtolos@tid.es"

# import packages for analysis and modeling
import pandas as pd # data frame operations
import glob #pathname pattern expansion
import datetime
import os


class PowerConsumptions(object):

    DEFAULT_ENCODING = 'UTF-8'

    population = {'Country':['AT', 'BE', 'CH', 'CY', 'DE', 'DK', 'ES', 'FI', 'FR', 'GB', 'HR', 'HU', 'IE', 'IT', 'PT'],
                  'Population': [8527230, 11215442, 8183800, 85800, 80767000, 5639719, 46507760, 5468609, 66048000,
                                 64105654, 10992589, 4267558, 9879000, 4609600, 60780377, 10477800]}

    def __init__(self, dir_path, pattern, skiprows=9, maxcolumns=26, hourchange='3B:00:00'):
        if os.path.isfile(os.path.join(dir_path, 'hconsum')):
            self.df = pd.load(os.path.join(dir_path, 'hconsum'))
        else:
            self.load_dataframe(dir_path, pattern, skiprows, maxcolumns, hourchange)

    def load_dataframe(self, dir_path, pattern, skiprows=9, maxcolumns=26, hourchange='3B:00:00'):
        """
        This function parses hourly (1:24) consumption data from all countries and returns a Pandas DataFrame with the
        following schema: (country, day, 1,2,3...,24,total)
        :param file_path: The file path where to search for the files
        :param pattern: The pattern of the files to be searched
        :return: A Pandas DataFrame object with the hourly consumption for all countries and all dates
        :param skiprows: rows to skip from the excel file
        :param maxcolumns: max number of columns that should appear on the excel file
        :param hourchange: label showing the hour change that makes a new column on the worksheet
        """
        print('_' * 80)
        self.df = pd.DataFrame()

        # search for the files to load
        for file in glob.glob(dir_path + pattern):
            print file
            # read excel file
            wb = pd.read_excel(file, 'Statistics', skiprows=skiprows, na_values=[u'n.a.'], keep_default_na=False)
            # check if there are more than maxcolumns since it is the change of hour
            if len(wb.columns) != maxcolumns:
                # take out the change of hour
                del wb[hourchange]
            # change columns names
            #wb.columns = ['country', 'day'] + range(1,25)

            weekdays = []
            months = []
            years = []
            for index, row in wb.iterrows():
                date = datetime.datetime.strptime(row['Day'], '%Y-%m-%d')
                weekdays.append(str(date.weekday()))
                months.append(str(date.month))
                years.append(str(date.year))

            wb['weekday'] = pd.Series(weekdays)
            wb['month'] = pd.Series(months)
            wb['year'] = pd.Series(years)

            # Append to the self.df data frame
            self.df = pd.concat([self.df, wb])


        self.df.save(os.path.join(dir_path, 'hconsum'))
