import pandas as pd
import numpy as np
import Classifier
class SimpleClassifier(Classifier.Classifier):
    def __init__(self):
        super(SimpleClassifier, self).__init__('SimpleClassifier')
    
    def train(self, reviews, users, businesses):
        self.users = users.loc[users['review_count'] != 0.0]
        self.users = self.users.assign(total_offset = np.zeros(self.users.shape[0]))
        self.businesses = businesses
        grouped_reviews = reviews.groupby('user_id')
        # i = 0
        for name, group in grouped_reviews:
            # i += 1
            self.users.at[name, 'total_offset'] = np.sum([businesses.loc[review.business_id].stars - review.stars for review in group.itertuples()])
            # if i == 200:
            #     print(f"Making progress - user: {users.loc[name]}")
            #     i = 0
        #print(self.users.loc[self.users['total_offset'] != 0])
        self.users = self.users.assign(avg_offset = self.users['total_offset']/self.users['review_count'])

    def classify(self, queries):
        data = []
        #print(self.users.index)
        #print("XEDaNNCTVAqPpvyX2zY03g" in self.users.index)
        for query in queries.itertuples():
            business = self.businesses.loc[query.business_id]
            if query.user_id not in self.users.index:
                data.append(business.stars)
            else:
                user = self.users.loc[query.user_id]
                #print(user)
                data.append(max(1, min(5, round(business.stars + user.avg_offset))))
        return pd.Series(data, name="stars")