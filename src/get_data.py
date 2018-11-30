import pandas as pd
import os
import numpy as np
import torch.utils.data 

_DEFINITES_ = [
    'business_id',
    #'attributes_GoodForKids',
    'attributes_OutdoorSeating',
    #'attributes_RestaurantsDelivery',
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

_USER_SELECTED_FEATURES_ = [
    'average_stars', 
    'review_count', 
    'user_id', 
]

_REVIEW_DEFAULT_FEATURES_ = [
    'business_id', 
    'cool', 
    'date', 
    'funny', 
    'review_id', 
    'stars', 
    'text',   
    'useful', 
    'user_id'
]

_REVIEW_SELECTED_FEATURES_ = [
    'business_id',  
    'review_id', 
    'stars',  
    'user_id'
]
def get_business_data(feats='definite', verbose=False):
    if feats == 'definite':
        b_features = _DEFINITES_
    elif feats == 'maybe':
        raise NotImplementedError('haven\'t finished writing the cleaning-up code for these')
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
    
    for col_name in business_data.columns:
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
        
        # if column contains categorical string data
        elif str in u_types:
            if verbose:
                print('TYPE: string. Doing one-hot encoding.')
            # If the column takes string values, then we one-hot encode the column.
            # For lack of a better way, I basically include NaN as a class when one-hot encoding
            to_replace = {u_v: i for (i, u_v) in enumerate(u_vals)}
            business_data[col_name].replace(to_replace, inplace=True)
        
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

def get_training_data(b_cols='definite', user_features = _USER_DEFAULTS_, rev_features = _REVIEW_DEFAULT_FEATURES_, verbose=False):
    """ 
    Retrieve training data.
    Returns a 3-tuple (business_data, user_data, review_data), where each element
        is a Pandas dataframe.
    
    The argument 'b_cols' indicates which columns/features of the business data to retrieve:
         'definite':  Include only the 8 features we marked as definite
            'maybe':  Include the definite features, plus the 15 we marked "Maybe".
             'full':  Include all features.
             
    NOTE: for now, I've left out the features "attributes_AgesAllowed" and "categories".
    """
        
    business_data = get_business_data(feats=b_cols, verbose=verbose)
    business_data.set_index('business_id', inplace=True)
    
    pfx = '..' if os.getcwd()[-3:] == 'src' else '.'
    user_data = pd.read_csv(pfx + '/data/users.csv', usecols=user_features)
    user_data.set_index('user_id', inplace=True)
    
    reviews = pd.read_csv(pfx + '/data/train_reviews.csv', usecols=rev_features)
    return (business_data, user_data, reviews)


def get_validation_reviews():
    pfx = '..' if os.getcwd()[-3:] == 'src' else '.'
    valid_queries = pd.read_csv(pfx + '/data/validate_queries.csv')   
    return valid_queries    
    

class YelpTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        training_data_sets = get_training_data( b_cols='definite',
                                                user_features=_USER_SELECTED_FEATURES_,
                                                rev_features=_REVIEW_SELECTED_FEATURES_)
        #Merge reviews from to combine business and user giving review of business
        dataset = training_data_sets[2].merge(right=training_data_sets[0], on='business_id', how='inner')
        dataset = dataset.merge(right=training_data_sets[1], on='user_id', how='inner')
        self.data_frame = dataset.drop(['review_id','business_id','user_id'],1)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        expected_stars = self.data_frame.iloc[idx].stars_x.astype(np.int32) - 1
        one_h_enc_expected_stars = np.zeros((1, 5))
        one_h_enc_expected_stars[0,expected_stars] = 1
        features = self.data_frame.iloc[idx].drop(['stars_x']).values.astype(np.float32)
        sample = {'features': features, 'expect': one_h_enc_expected_stars[0].astype(np.float32)}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    class YelpTestingDataset(torch.utils.data.Dataset):
        def __init__(self, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """

            training_data_sets = get_training_data( b_cols='definite',
                                                    user_features=_USER_SELECTED_FEATURES_,
                                                    rev_features=_REVIEW_SELECTED_FEATURES_)
            #Merge reviews from to combine business and user giving review of business
            dataset = training_data_sets[2].merge(right=training_data_sets[0], on='business_id', how='inner')
            dataset = dataset.merge(right=training_data_sets[1], on='user_id', how='inner')
            self.data_frame = dataset.drop(['review_id','business_id','user_id'],1)
            self.transform = transform

        def __len__(self):
            return len(self.data_frame)

        def __getitem__(self, idx):
            expected_stars = self.data_frame.iloc[idx].stars_x.astype(np.int32) - 1
            one_h_enc_expected_stars = np.zeros((1, 5))
            one_h_enc_expected_stars[0,expected_stars] = 1
            features = self.data_frame.iloc[idx].drop(['stars_x']).values.astype(np.float32)
            sample = {'features': features, 'expect': one_h_enc_expected_stars[0].astype(np.float32)}
            if self.transform:
                sample = self.transform(sample)
            return sample
        
        