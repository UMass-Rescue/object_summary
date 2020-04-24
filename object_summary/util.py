import pandas as pd
import numpy as np

from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO 
import IPython.display as display
from PIL import Image
import os
from tqdm import tqdm
from tinydb import TinyDB
import cv2
from .analysis import objects_in_categories_df
from .scene import scene_res_to_df

def flatten_fmap_res(fmap_res:'list(dict) - results from db.all() on TinyDB'):
    fmap_di = {}
    for e in fmap_res:
        k, v = next(iter(e.items()))
        fmap_di[k] = v
    return fmap_di

def merge_scene_and_obj_results(obj_db:TinyDB, obj_fmap_db:TinyDB, 
    scene_db:TinyDB, scene_fmap_db:TinyDB, scene_threshold:float=0.3):
    '''
    Merges the results of object detection and scene detection into a single
    DataFrame (merges the results using file_id and category columns)
    '''
    obj_df = objects_in_categories_df(obj_db.all())
    scene_df = scene_res_to_df(scene_db.all(), scene_threshold=scene_threshold)

    scene_fmap = flatten_fmap_res(scene_fmap_db.all())
    obj_fmap = flatten_fmap_res(obj_fmap_db.all())

    obj_df.file_id = obj_df.file_id.map(obj_fmap)
    scene_df.file_id = scene_df.file_id.map(scene_fmap)

    merged_df = obj_df.merge(scene_df, on=('file_id', 'category'), 
        how='inner', suffixes=('_obj', '_scene') )
    return merged_df

def np_to_list(di:dict):
    '''
    Converts all values in "di" dictionary that are numpy ndarrays to a python list

    WARNING - The conversion is done inplace. i.e. this function modifies "di".
    '''
    for k, v in di.items():
        if isinstance(v, np.ndarray):
            di[k] = v.tolist()

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

def verify_images(paths:'list(str) - list of paths', 
                delete:'Boolean, if True, deletes corrupt images'=False) -> 'list(str): list of paths of corrupt images':
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
             class_names: 'list(str) - list of label (class) names', 
             feature_names: 'list(str) - list of names of features of the independent variable', 
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