import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

def normalize_columns(X):
    """
    Make each column have a mean of 0 and a variance of 1.
    """
    X = X - np.mean(X, axis=0)
    f_vars = np.var(X, axis=0)  # feature variances
    if any(f_vars == 0.):
        print(np.argwhere(f_vars == 0))
        raise ValueError('One or more columns (above) have zero variance...')
        
    X = X / np.var(X, axis=0)
    return X


def get_PCA_matrix(
    X, explained_variance=0.9, n_components=None, plot_expl_variance=False, verbose=False
):
    """
    Do principal component analysis to reduce design matrix dimensionality.
    
    IMPORTANT: only do PCA on the TRAINING data - do NOT include the validation or test data when
      constructing your transformation matrix.
      
    Args:
      (np.array)                 X: design matrix
      (float)   explained_variance: proportion of variance we want to keep.
       (int)          n_components: alternatively, how many components to use (overrides 
                                      explained_variance if specified).
      (bool)    plot_expl_variance: whether to plot amount of explained variance versus
                                      number of components.
      (bool)               verbose: print information or not
      
    Returns the (n_components, M) matrix Z which you multiply your design matrix by.
        (where M = number of features in untransformed data).
    """
    
    pca = PCA()
    pca.fit(X)
        
    evr = np.copy(pca.explained_variance_ratio_)
    for i in range(1, len(evr)):
        evr[i] += evr[i-1]
        
    if plot_expl_variance:
        f, ax = plt.subplots(figsize=(12, 8))
        ax.plot(range(X.shape[1]), evr)
        ax.scatter(range(X.shape[1]), evr)
        
        if n_components is None:
            ax.plot(
                [0, X.shape[1]-1], [explained_variance]*2, color='red', linestyle='--'
            )
        else:
            ev = evr[n_components]
            ax.plot(
                [0, X.shape[1]-1], [ev]*2, color='red', linestyle='--'
            )
#             ax.scatter([n_components], [ev], color='red', s='10', zorder=5)
        
        ax.set_xlabel('Number of components')
        ax.set_ylabel('Percent of variance explained')
        plt.show()
        
    
    n_comps = np.argwhere(evr >= explained_variance).min() + 1
    if n_components is not None:
        n_comps = n_components
    
    if verbose:
        print('Using %d components - %f of variance explained.' % (n_comps, evr[n_comps]))
        
    Z = pca.components_[:n_comps, :]
    return Z
















