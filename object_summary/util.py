import pandas as pd
import numpy as np

from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO 
import IPython.display as display
from PIL import Image
import os
from tqdm import tqdm
import cv2

def resize_img(img, resize_to=720):
    '''
    img : np.ndarray - image array of size (height, width, channels)
    resize_to : int - the larger among height and width will be resized to "resize_to". 
                    The other dimension will be scaled to preserve the original aspect ratio.
                    
    returns : np.ndarray - resized image
    '''
    h, w, ch = img.shape
    if h <= resize_to and w <= resize_to:
        return img

    if h >= w:
        new_h = resize_to
        new_w = int((new_h / h) * w)
    else:
        new_w = resize_to
        new_h = int((new_w / w) * h)

    return cv2.resize(img, (new_w, new_h))


def verify_image(path):
    '''
    path : str - path to the image

    returns True if valid image file. returns False otherwise.

    Reference - https://opensource.com/article/17/2/python-tricks-artists
    '''
    try:
      img = Image.open(path)
      img.verify()
      return True
    except (IOError, SyntaxError) as e:
      return False

def verify_images(paths, delete=False):
    res = []
    for path in tqdm(paths):
        res.append(verify_image(path))

    bad_paths = []
    for p, r in zip(paths, res):
        if r == False:
            bad_paths.append(p)
            if delete:
                os.remove(p)
    return bad_paths

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
             out_fname:'if specified, graph is saved to this path'=None,rotate=False) -> Image:
    dot_data = StringIO()
    export_graphviz(model, 
     out_file=dot_data, 
     class_names=class_names, 
     feature_names=feature_names,
     filled=True,
     rounded=True,
     special_characters=True, rotate=rotate)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
    if out_fname:
        graph.write_png(out_fname)
    return display.Image(graph.create_png())