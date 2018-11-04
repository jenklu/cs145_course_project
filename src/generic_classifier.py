import pandas as pd
import numpy as np
# Change this to the type of your classifier
import simple_classifier as clf
# Change this as needed to get more cols from the data
reviews = pd.read_csv("../data/train_reviews.csv", usecols=['business_id', 'user_id', 'stars'])
users = pd.read_csv("../data/users.csv", usecols=['user_id', 'review_count'])
users = users.set_index('user_id')
businesses = pd.read_csv("../data/business.csv", usecols=['business_id', 'stars'])
businesses = businesses.set_index('business_id')
# Change this to the model in your file
classifier = clf.SimpleClassifier()
classifier.train(reviews, users, businesses)
queries = pd.read_csv("../data/test_queries.csv")
output = classifier.classify(queries)
classifier.write(output)