import pandas as pd
import os
import numpy as np
import ast

_DEFINITES_ = [
    'business_id',
    'attributes_GoodForKids',
    'attributes_OutdoorSeating',
    'attributes_RestaurantsDelivery',
    'attributes_RestaurantsGoodForGroups',
    'attributes_RestaurantsPriceRange2',
    'attributes_RestaurantsReservations',
    'attributes_WiFi',
    'attributes_AgesAllowed',
    'stars'
]

_MAYBES_ = [
    'attributes_Alcohol',
    'attributes_AcceptsInsurance',
    'attributes_Ambience',                 # one hot encode -> PCA
    'attributes_BusinessParking',
    'attributes_Caters',
    'attributes_DogsAllowed',
    'attributes_GoodForDancing',
    'attributes_GoodForMeal',
    'attributes_HasTV',
    'attributes_NoiseLevel',
    'attributes_RestaurantsAttire',
    'attributes_RestaurantsTableService',
    'latitude',
    'longitude',
    'review_count'
]
    
_USER_DEFAULTS_ = [
    'average_stars', 
    'compliment_cool', 
    'compliment_cute',
    'compliment_funny', 
    'compliment_hot', 
    'compliment_list',
    'compliment_more', 
    'compliment_note', 
    'compliment_photos',
    'compliment_plain', 
    'compliment_profile', 
    'compliment_writer', 
    'cool',
    'fans', 
    'funny', 
    'review_count', 
    'useful',
    'user_id', 
]

def get_business_data(feats='definite', verbose=False):
    if feats == 'definite':
        b_features = _DEFINITES_
    elif feats == 'maybe':
        b_features = _DEFINITES_ + _MAYBES_
    elif feats == 'full':
        # include all
        b_features = None
    
    fpath = 'data/business.csv'
    if os.getcwd()[-3:] == 'src':
        fpath = '../' + fpath
        
    business_data = pd.read_csv(fpath, usecols=b_features)
    
    return clean_business_data(business_data, verbose=verbose)
    
def clean_business_data(business_data, verbose=False):
    """ 
    TODO: documentation
    """
    
    ignore = ['business_id', 'stars']
    numeric_dtypes = [np.int64, np.float64, np.float64, float]
    
    b_columns = business_data.columns
    
    for col_name in b_columns:
        if col_name in ignore:
            continue
            
        if verbose:
            print('='*10, "Feature '%s'" % col_name, '='*10)
        
        # unique values
        u_vals = business_data[col_name].unique()
        # unique types
        u_types = list(set([type(v) for v in u_vals]))
        
        # determine data type(s) of column
        if bool in u_types:
            if verbose:
                print('TYPE: boolean. Changing False -> 0, True -> 1.')
            to_replace = {True: 1, False: 0}
            business_data[col_name].replace(to_replace, inplace=True)
            
            # replace NaN's with average non-NaN value
            if pd.isnull(u_vals).any():
                r_val = np.mean(business_data[col_name].notnull())
                business_data[col_name].fillna(value=r_val, inplace=True)
    
                if verbose:
                    print('Detected NaN in column. Replacing with mean of non-NaN values.')
        
        # if column contains categorical string data OR dictionary data
        elif str in u_types:
            
            # Check if dictionary
            try:
                v0 = ast.literal_eval(business_data[col_name][0])
            except ValueError:
                v0 = 'string'
                
            if type(v0) is dict:
                if verbose:
                    print('TYPE: dict. Creating new features and doing one-hot encoding.')
                    
                business_data = process_dictionary_data(business_data, col_name)
            else:
                if verbose:
                    print('TYPE: string. Doing one-hot encoding.')
                    
                # If the column takes string values, then we one-hot encode the column.
                # For lack of a better way, I basically include NaN as a class when one-hot encoding
#                 to_replace = {u_v: i for (i, u_v) in enumerate(u_vals)}
#                 business_data[col_name].replace(to_replace, inplace=True)
                business_data = one_hot_encode(business_data, col_name)
        
        # if column contains only numeric data
        elif all([dtype in numeric_dtypes for dtype in u_types]):
            if verbose:
                print('TYPE: numeric.')
            # replace NaN's with average non-NaN value
            if pd.isnull(u_vals).any():
                r_val = np.mean(business_data[col_name].notnull())
                business_data[col_name].fillna(value=r_val, inplace=True)
                
                if verbose:
                    print('Detected NaN in column. Replacing with mean of non-NaN values.')
        else:
            print('u_vals: ', u_vals)
            print('u_types: ', u_types)
            raise NotImplementedError('?? for %s' % col_name)
            
        if verbose:
            print('')
            
    return business_data

def get_training_data(b_cols='definite', verbose=False):
    """ 
    Retrieve training data.
    Returns a 3-tuple (business_data, user_data, review_data), where each element
        is a Pandas dataframe.
    
    The argument 'b_cols' indicates which columns/features of the business data to retrieve:
         'definite':  Include only the 8 features we marked as definite
            'maybe':  Include the definite features, plus the 15 we marked "Maybe".
             'full':  Include all features.
             
    NOTE: for now, I've left out the feature "categories".
    """
        
    business_data = get_business_data(feats=b_cols, verbose=verbose)
    business_data.set_index('business_id', inplace=True)
    
    pfx = '..' if os.getcwd()[-3:] == 'src' else '.'
    user_data = pd.read_csv(pfx + '/data/users.csv', usecols=_USER_DEFAULTS_)
    user_data.set_index('user_id', inplace=True)
    
    reviews = pd.read_csv(pfx + '/data/train_reviews.csv')
                                
    return (business_data, user_data, reviews)

def get_validation_reviews():
    
    pfx = '..' if os.getcwd()[-3:] == 'src' else '.'
    valid_queries = pd.read_csv(pfx + '/data/validate_queries.csv')
                                
    return valid_queries    
    
def process_dictionary_data(business_data, col_name):
    """ 
    Turn a dictionary column into a number of one-hot encoded features.
    
    Assumes dict values are of type BOOLEAN.
    """
    d = ast.literal_eval(business_data[col_name][0])
    N, M = len(business_data), len(d.keys())
    
    keys = list(d.keys())
    new_features = np.zeros((N, M), dtype=np.float64)
    
    for i, value in enumerate(business_data[col_name]):
        try:
            float(value)
            new_features[i] = np.full(new_features[i].shape, fill_value=np.nan)
            continue
        except ValueError:
            pass
        
        v_dict = ast.literal_eval(value)
        assert type(v_dict) is dict, "ERROR: %s" % value
    
        new_features[i] = np.array([int(v_dict[k]) for k in keys])
    
    business_data = business_data.drop(col_name, axis=1)
    
    # Now process NaN values
    for m in range(M):
        nan_idx = np.isnan(new_features[:, m])
        non_nan_idx = np.invert(nan_idx)
        
        avg_val = np.mean(new_features[non_nan_idx, m])
        new_features[nan_idx, m] = avg_val
        
        c_name = col_name + keys[m].upper()
        business_data.insert(loc=len(business_data.columns), column=c_name, value=new_features[:, m])
    
    return business_data
    
def one_hot_encode(business_data, col_name):
    """
    Take a column containing STRING categorical data and return a 
      DataFrame with that column replaced with one-hot-encoded columns.
    """
    u_vals = business_data[col_name].unique()
    
    N, M = len(business_data), len(u_vals)
    new_features = np.zeros((N, M))
    
    for i, value in enumerate(business_data[col_name]):
        new_features[i, u_vals == value] = 1
        
    business_data = business_data.drop(col_name, axis=1)
    
    for m in range(M):
        c_name = col_name + str(u_vals[m]).upper()
        business_data.insert(
            loc=len(business_data.columns), column=c_name, value=new_features[:, m]
        )
    
    return business_data

def construct_design_matrix(business_data, user_data, reviews):
    """
    Construct a (np.ndarray) design matrix of business-user-review data, and also the target
      array y of star ratings.
    """
    N = len(reviews['stars'])
    Db = len(business_data.columns)
    Du = len(u_data.columns)
    D = Db + Du

    X = np.zeros((N, D))
    y = np.zeros(N)

    for i, review in reviews.iterrows():
    if (i % 20000) == 0:
        print('%d/%d done' % (i, N))
    
    u_id = review['user_id']
    b_id = review['business_id']
    y[i] = review['stars']
    
    X[i, :Db] = b_data.loc[b_id].values
    X[i, Db:] = u_data.loc[u_id].values
    
    print('Done')

