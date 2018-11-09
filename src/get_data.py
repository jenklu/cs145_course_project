import pandas as pd

_DEFINITES_ = [
    'business_id',
    'attributes_GoodForKids',
    'attributes_OutdoorSeating',
    'attributes_RestaurantsDelivery',
    'attributes_RestaurantsGoodForGroups',
    'attributes_RestaurantsPriceRange2',
    'attributes_RestaurantsReservations',
    'attributes_WiFi',
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


def get_business_data(feats='definite'):
    if feats == 'definite':
        b_features = _DEFINITES_
    elif feats == 'maybe':
        raise NotImplementedError('haven\'t finished writing the cleaning-up code for these')
        b_features = _DEFINITES_ + _MAYBES_
    elif feats == 'full':
        # include all
        b_features = None
        
    business_data = pd.read_csv('../data/business.csv', usecols=b_features)
    
def clean_business_data(business_data):
    """ 
    WARNING: UNTESTED 
    
    TODO: documentation
    """
    
    ignore = ['business_id', 'stars']
    
    for col_name in business_data.columns:
        if col_name in ignore:
            continue
        
        # unique values
        u_vals = set(business_data[col_name])
        # unique types
        u_types = set([type(v) for v in u_vals])
        
        # determine data type(s) of column
        if bool in u_types:
            to_replace = {True: 1, False: 0}
            business_data = business_data.replace(to_replace)
            
            # replace NaN's with average non-NaN value
            if np.isnan(u_vals).any():
                r_val = np.mean(business_data[c_name].notnull())
                business_data = business_data.fillna(value=r_val)
        
        # if column contains categorical string data
        elif str in u_types:
            # If the column takes string values, then we one-hot encode the column.
            # For lack of a better way, I basically include NaN as a class when one-hot encoding
            to_replace = {i, u_v for i, u_v in enumerate(u_vals)}
            business_data = business_data.replace(to_replace)
        
        # if column contains numeric data
        elif len(u_types) == 1 and list(u_types)[0] == float:
            # replace NaN's with average non-NaN value
            if np.isnan(u_vals).any():
                r_val = np.mean(business_data[c_name].notnull())
                business_data = business_data.fillna(value=r_val)
            
    return business_data

def get_training_data(b_cols='definite'):
    """ 
    WARNING: UNTESTED
    
    Retrieve training data.
    Returns a 3-tuple (business_data, user_data, review_data), where each element
        is a Pandas dataframe.
    
    The argument 'b_cols' indicates which columns/features of the business data to retrieve:
         'definite':  Include only the 8 features we marked as definite
            'maybe':  Include the definite features, plus the 15 we marked "Maybe".
             'full':  Include all features.
             
    NOTE: for now, I've left out the features "attributes_AgesAllowed" and "categories".
    """
        
    business_data = get_business_data(feats=b_cols)
    
    user_data = pd.read_csv('../data/users.csv', usecols=_USER_DEFAULTS_)
                                
    reviews = pd.read_csv('../data/train_reviews.csv')
                                
    return (business_data, user_data, reviews)
                                
                                
                        