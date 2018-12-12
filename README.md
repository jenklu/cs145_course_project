# CS145 Final Course Project: Help Yelp!
CS145 Course Project to Predict Yelp Ratings

# Workflow

Our data processing involved trying many different models, and manipulating the parameters of these models in varying ways. This lead to a development structure with many separate tasks which could be done independently of each other, which was perfect for our group work structure. It allowed us to work separately, and collaborate through git. This structure lead us to make a few choices about the structure of our project. We have some utility functions used by many models, which are detailed in the Shared Files section. We also have some files which amount to implementations of different ML models to process the data.

## Necessary Libraries
Besides the standard Python libraries, we make heavy use of NumPy, Pandas, PyTorch, Scikit Learn, and to a lesser extend MatPlotLib. We also use Python 3.6. 

## Shared Files
There are some files that were reused in training many of the models we developed. These mainly hold utility functions which manipulate and pre/postprocess the design matrices used for scikit-learn-type classifiers. These files are all in the src/ directory, and include preprocessing.py, get_data.py, create_output.py, visualize.py.

## File Unique to Specific Models
Note that most of the development for different models was done in Jupyter notebooks (.ipynb files) using Python 3.6, which are labeled appropriately. These files need to be run in Jupyter using Python 3.6. These files include all of the .ipynb files in the main directory of the .zip file. The names of these files correspond to which model we were investigating with them.
A couple of the models we tested were implemented in simple python files. These files include nn.py, which implements a neural net similar to "Deep neural network.ipynb" - note that this was an earlier implementation, and the .ipynb file is the one we used for our calculations. 

# Note that our files expect a data/ directory, set up in the manner that the one in our .zip file is.

## Simple Classifier files
### Classifier.py: the base Classifier class, from which all other classifiers can inherit. 
By inheriting from this class, you are required to override the train method, which takes users, reviews, and businesses dataframes as inputs, and stores the trained model internally. You are also required to implement the classify method, which takes as input a dataframe of queries and outputs a pandas series of classifications for each query. You get the ability to simply use "classifier.write(output_from_classify)" to an output file in the format necessary for the Kaggle submission.

## simple_classifier.py: implements an instance of Classifier.Classifier
As discussed during our first meeting, for each user, we calculate their avg offset from the avg star rating of the business. We then append that value onto the avg star rating of the test business as our classification (and round to the nearest integer, as I think we have to predict integer star ratings? But we should clarify this).

## generic_classifier.py: the runnable file to create submittable output
By changing the import statement to a different implementation of Classifier.Classifier, and maybe updating the "usecols" of each dataframe we read from csv, you should be able to make this run with a different implementation of Classifier.Classifier relatively easily.
