import pandas as pd
import numpy as np

from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO 
from IPython.display import Image

def split_df(df, num_splits:int):
    '''
    df - pandas DataFrame object
    num_splits - int - number of equal parts to split the DataFrame into

    returns - list[DataFrame] - returns the split DataFrame objects in a list
    '''
    if num_splits <= 0:
        raise ValueError('Number of splits cannot be less than or equal to zero.')
        
    N = df.shape[0]
    split_ends = np.linspace(0, N, num_splits + 1, dtype=np.int32)
    parts = []
    for i in range(1, len(split_ends)):
        start = split_ends[i - 1]
        end = split_ends[i]
        parts.append(df.iloc[start:end])
        
    return parts

def tree_viz(model: 'Decision Tree model', 
             class_names: 'list(str) - list of label names', 
             feature_names: 'list(str)', 
             out_fname:'if specified, graph is saved to this path'=None) -> Image:
    dot_data = StringIO()
    export_graphviz(model, 
     out_file=dot_data, 
     class_names=class_names, 
     feature_names=feature_names,
     filled=True,
     rounded=True,
     special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    if out_fname:
        graph.write_png(out_fname)
    return Image(graph.create_png())