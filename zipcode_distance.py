import pgeocode
from pandas import DataFrame as frame


class Distance_zipcode:
    """
    X
    """

    def __init__(self, df_code, base_code='51-430'):
        self.df = df_code
        self.dist = pgeocode.GeoDistance('PL')
        self.base = base_code

    def get_dist(self):
        f_df = frame(columns=['id', 'dist_km'])
        for index, row in self.df.iterrows():
            f_df.loc[index] = [row['cus'], int(round(self.dist.query_postal_code(self.base, row['zip_code'])))]
        return f_df


# TEST DF
# df = frame({'cus': [10, 11, 12], 'zip_code': ['59-700', '59-702', '52-200']})
# x = Distance_zipcode(df).get_dist()
# print(x)