import pandas as pd

# TEST DF
# df0 = pd.DataFrame({'a': [20, 24, 22, 19, 29, 18, 4300, 30, 18],
#                     'b': [-400, 24, 22, 19, 29, 18, 300, 30, 18]})


class Outliers:
    """
    X
    """

    def __init__(self, df):
        self.df = df

    def _interquartile_range(self):
        """
        :return:
        """
        q1 = self.df.quantile(0.25)
        q3 = self.df.quantile(0.75)
        iqr = q3 - q1

        low_boundary = (q1 - 1.5 * iqr)
        high_boundary = (q3 + 1.5 * iqr)

        return low_boundary, high_boundary, iqr

    def show_interquartile_outliers(self):
        """
        :return:
        """
        low_boundary, high_boundary, iqr = self._interquartile_range()

        num_of_outliers_l = (self.df[iqr.index] < low_boundary).sum()
        num_of_outliers_h = (self.df[iqr.index] > high_boundary).sum()
        num_of_outliers = pd.DataFrame(
            {'lower_boundary': low_boundary, 'upper_boundary': high_boundary, 'num_of_outliers_L': num_of_outliers_l,
             'num_of_outliers_U': num_of_outliers_h})
        return num_of_outliers

    def remove_outliers(self):
        """
        :return:
        """
        outliers = self.show_interquartile_outliers()
        for row in outliers.iterrows():
            df_without_outliers = self.df[
                (self.df[row[0]] >= row[1]['lower_boundary']) & (self.df[row[0]] <= row[1]['upper_boundary'])]
        return df_without_outliers, df_without_outliers.index

    def get_outliers(self):
        _, idx = self.remove_outliers()
        return self.df.drop(index=idx)

    def find_out_3sigma_method(self, parm=3):
        """
        :return:
        """
        outliers = set()

        data = self.df.agg(['std', 'mean'])
        data.loc['std'] = data.loc['std'] * parm

        low_boundary = data.loc['mean'] - data.loc['std']
        high_boundary = data.loc['mean'] + data.loc['std']

        for i in self.df.columns:
            outliers.add(self.df[(self.df[i] <= low_boundary[i]) | (self.df[i] >= high_boundary[i])].index[0])
        return self.df.loc[outliers]
