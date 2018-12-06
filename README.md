# cs145_course_project
CS145 Course Project to Predict Yelp Ratings

https://www.overleaf.com/2539785618gbmsryyycgpj

# src files
# Classifier.py: the base Classifier class, from which all other classifiers can inherit. 
By inheriting from this class, you are required to override the train method, which takes users, reviews, and businesses dataframes as inputs, and stores the trained model internally. You are also required to implement the classify method, which takes as input a dataframe of queries and outputs a pandas series of classifications for each query. You get the ability to simply use "classifier.write(output_from_classify)" to an output file in the format necessary for the Kaggle submission.

# simple_classifier.py: implements an instance of Classifier.Classifier
As discussed during our first meeting, for each user, we calculate their avg offset from the avg star rating of the business. We then append that value onto the avg star rating of the test business as our classification (and round to the nearest integer, as I think we have to predict integer star ratings? But we should clarify this).

# generic_classifier.py is the runnable file
By changing the import statement to a different implementation of Classifier.Classifier, and maybe updating the "usecols" of each dataframe we read from csv, you should be able to make this run with a different implementation of Classifier.Classifier relatively easily.