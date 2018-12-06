import numpy as np
import matplotlib.pyplot as plt

def visualize_weight_vector(w, column_names=None):
    """
    Plot weights and annotate biggest magnitude weights if column_names is provided.
    """
    xs = range(len(w))
    
    tiny_weights = np.argwhere(abs(w) <= 10**-4)
    print('num weights ~= 0: %d' % len(tiny_weights))
    
    f, ax = plt.subplots(figsize=(12, 8))
    ax.plot(xs, w)
    ax.scatter(xs, w)
    ax.scatter(tiny_weights, w[tiny_weights], color='red', zorder=5, label='|w| < 10**-4')

    if column_names is not None:
        ws_idx = np.argsort(abs(w))[::-1][:5]
        
        for w_i in ws_idx:
            c_n = column_names[w_i]
            attr_name = c_n[11:] if c_n[:11] == 'attributes_' else c_n
            
            text = '%s: %.3f' % (attr_name, w[w_i])
            
            ax.annotate(s=text, xy=(w_i, w[w_i]))

    plt.legend()
    plt.show()
    
    
    