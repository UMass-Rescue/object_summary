import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from PIL import Image
from functools import partial
import pdb

def get_non_nan(ser:pd.Series) -> pd.Series:
    '''
    returns non nan values in pandas Series "ser"
    '''
    return ser[~ser.isna()]

def get_unique_objects(ser:pd.Series) -> set:
    '''
    Returns unique objects in series "ser". The method relies on the convention of object summary
    to name number of occurances of an object under the column "{object_name}_count"
    '''
    return set(map( lambda s: s[:-6]  , filter(lambda f: f.endswith('count'), ser.keys())))

def get_coordinates(res:pd.Series, obj:str, idx:int) -> np.ndarray:
    '''
    returns the coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y) of
    the "idx"th occurance of object "obj" in "res"
    '''
    width = res.img_width
    height = res.img_height
    x0, y0, x1, y1 = (res[f'{obj}_tl_x_{idx}']*width, res[f'{obj}_tl_y_{idx}']*height, 
                    res[f'{obj}_br_x_{idx}']*width, res[f'{obj}_br_y_{idx}']*height)
    return np.array((x0, y0, x1, y1))

def get_all_coordinates(ser:pd.Series, obj:str) -> np.ndarray:
    '''
    Returns an array of coordinates of occurances of object "obj" in series "ser"
    '''
    num_objs = ser[f'{obj}_count']
    return np.array([get_coordinates(ser, obj, i) for i in range(int(num_objs))])

def object_distance(obj1_coordinates:np.ndarray, obj2_coordinates:np.ndarray) -> np.float32:
    '''
    calculates the euclidian distance between two object coordinates
    '''
    return np.linalg.norm(obj1_coordinates - obj2_coordinates)

def single_unique_obj_distance(res1:pd.Series, res2:pd.Series, max_penalty:float) -> float:
    '''
    Calculates the distance penalty for all the unique objects in res1. Unique objects in 
    res1 refers to the objects in res1 that aren't present in res2
    '''
    penalty = 0.
    unq_objs = get_unique_objects(res1) - get_unique_objects(res2)
    for obj in unq_objs:
        penalty += (res1[f'{obj}_count'] * max_penalty)
    return penalty
        
def unique_obj_distance(res1:pd.Series, res2:pd.Series, m:float=2.) -> float:
    '''
    Calculates the penalty for the unique objects in both "res1" and "res2"
    m - m is a parameter that controls how much higher unique objects will be penalized as 
        opposed to non unique objects that don't get paired. By default, m = 2.0. This means that
        if a unique object is encountered, it receives 2.0 as much penalty as a non unique un-paired object 
        would.
    '''
    max_penalty = (m * object_distance(np.array((0, 0, 0, 0)), 
               np.array((res1.img_width, res1.img_height, res1.img_width, res1.img_height)))) + 1
    penalty = (single_unique_obj_distance(res1, res2, max_penalty) + 
               single_unique_obj_distance(res2, res1, max_penalty))
    return penalty

def common_obj_distance(res1:pd.Series, res2:pd.Series) -> float:
    '''
    calculates the distance penalty due to common objects in both res1 and res2
    '''
    max_distance = object_distance(np.array((0, 0, 0, 0)), 
               np.array((res1.img_width, res1.img_height, res1.img_width, res1.img_height))) + 1
    
    common_objects = list(get_unique_objects(res1).intersection(get_unique_objects(res2)))
    
    matched_bbox_penalty = 0.0
    unmatched_bbox_penalty = 0.0

    # TODO - use reduce instead of explicit loops
    for obj in common_objects:
        res1_objs = get_all_coordinates(res1, obj)
        res2_objs = get_all_coordinates(res2, obj)
        all_dists = list(np.ndenumerate(euclidean_distances(res1_objs, res2_objs)))
        all_dists = sorted(all_dists, key=lambda e: e[1])

        res1_obj_paired = np.array([False for i in range(res1_objs.shape[0])])
        res2_obj_paired = np.array([False for i in range(res2_objs.shape[0]) ])

        for (res1_idx, res2_idx), dis in all_dists:
            if res1_obj_paired.sum() == res1_obj_paired.shape[0] or res2_obj_paired.sum() == res2_obj_paired.shape[0]:
                break
            if res1_obj_paired[res1_idx] == False and res2_obj_paired[res2_idx] == False:
                res1_obj_paired[res1_idx], res2_obj_paired[res2_idx] = True, True
                matched_bbox_penalty += dis
        
        res_not_fully_paired = res1_obj_paired if res1_obj_paired.shape[0] > res2_obj_paired.shape[0] else res2_obj_paired
        unmatched_bbox_penalty += (~res_not_fully_paired).sum() * (max_distance + 1)

    return (matched_bbox_penalty) + unmatched_bbox_penalty

def obj_det_coordinate_dist(res1:pd.Series, res2:pd.Series, m:float=2.0) -> float:
    '''
    calculates the distance between two object detection results using the object coordinates
    '''
    return unique_obj_distance(res1, res2, m=m) + common_obj_distance(res1, res2)


def visualize_res(res, inf, fmap):
    '''
    res - a single entry from the results obtained from "objects_in_categories" function
    inf - tf_object_detection_util.inference_api.TFInference object. Required in order to draw visualizations.
    fmap - mapping from file_id to file path
    '''
    plt.figure(figsize=(12,8))
    plt.imshow(inf.visualize_pred(res, np.asarray(Image.open(fmap[res['file_id']]))))
    
def proc_entry_for_distance(ser:pd.Series, query_point:pd.Series, m=2.0) -> float:
    compare_to = get_non_nan(ser)
    dist = obj_det_coordinate_dist(query_point, compare_to, m=m)
    return dist

def get_similar_images(df:pd.DataFrame, query_idx:int, m:float=2.0):
    '''
    df - pd.DataFrame obtained from "res_to_df" function in object_summary.util.
    query_idx - the index of the query image in the DataFrame
    Returns a DataFrame where indicies are the index of the image result in "df" and value
    is the distance of the result from the entry at the query index. The DataFrame will be sorted 
    in ascending order.
    '''
    # TODO - change this to get query as a Series directly. Currently, we require that
    # the query image be in the "df" DataFrame.
    query_point = get_non_nan(df.iloc[query_idx])
    dists = []
    proc_ser = partial(proc_entry_for_distance, query_point=query_point, m=m)
    return df.apply(proc_ser, axis=1).sort_values()

def visualize_similarity_results(dist_idxs:'iterable(int)', res_json:'list(dict)', 
                                 inf:'TFInference', fmap:dict, top:int=10):
    '''
    dist_idxs - list of indicies of the items in "res_json". 
    res_json - list of dictionaries. results obtained from "objects_in_categories" function.
    inf - tf_object_detection_util.inference_api.TFInference object. Required in order to draw visualizations.
    top - the number specified in "top" will be the number of visualizations produced (from the start of the list)
    fmap - mapping from file_id to file path
    
    The visualizations will be produced in the order specified in "dist_idxs".
    '''
    for idx in dist_idxs[:top]:
        visualize_res(res_json[idx], inf, fmap)
        plt.show()