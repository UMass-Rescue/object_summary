from collections import defaultdict
import pandas as pd
import numpy as np
import pdb

def counts_in_single_res(res:'list(str)', key_suffix:str='') -> 'dict(int)':
    '''
    res - list of category strings
    returns - dictionary of form {objectname:count}
    '''
    object_counter = defaultdict(int)
    for e in res:
        object_counter[e+key_suffix] += 1
    return dict(object_counter)

def objects_in_categories_df(res:'results obtained from "objects_in_categories function"',
                            object_list_key:'key (string) to access the list of detected class names' = 'detection_classes_translated',
                            id_col:'key (string) to access the unique ID for each entry in "res"'='file_id',
                            cat_str:'key (string) to access the category in "res" entries'='category') \
                            -> pd.DataFrame:
    counts = []
    for r in res:
        di = counts_in_single_res(r[object_list_key])
        di[cat_str] = r[cat_str]
        di[id_col] = r[id_col]
        counts.append(di)
    count_df = pd.DataFrame(counts)
    count_df = count_df.fillna(0.0)
    return count_df

def get_counts_df(res:'results obtained from "objects_in_categories function"',
                    object_list_key:'key (string) to access the list of detected class names' = 'detection_classes_translated',
                    id_col:'key (string) to access the unique ID for each entry in "res"'='file_id',
                    cat_str:'key (string) to access the category in "res" entries'='category') \
                    -> pd.DataFrame:
    '''
    res - list of dictionaries. Result obtained from "objects_in_categories" function
    returns - pandas DataFrame where rows are categories and columns are objects. a cell contains the number of 
    objects that have been found in a category
    '''
    df = objects_in_categories_df(res)
    df = df.drop(id_col, axis=1)
    count_df = df.groupby(by=cat_str).sum()
    count_df = count_df.sort_index(axis=1)
    return count_df

def detection_box_to_dict(detection_box:list, key_prefix:str=''):
    return {
        key_prefix + 'tl_y':detection_box[0],
        key_prefix + 'tl_x':detection_box[1],
        key_prefix + 'br_y':detection_box[2],
        key_prefix + 'br_x':detection_box[3]
    }

def extract_scores_and_bbox(res:'dict - single result', 
        object_list_key:'key (string) to access the list of detected class names' = 'detection_classes_translated',
        scores_key:str='detection_scores',
        detection_boxes_key:str = 'detection_boxes',):
    res_di = {}
    counts = defaultdict(int)
    all_objs = res[object_list_key]
    for i in range(len(all_objs)):
        cur_obj = all_objs[i]
        cur_idx = counts[cur_obj]
        counts[cur_obj] += 1
        res_di[f'{cur_obj}_score_{cur_idx}'] = res[scores_key][i]
        bbox = res[detection_boxes_key][i]
        res_di[f'{cur_obj}_tl_y_{cur_idx}'] = bbox[0]
        res_di[f'{cur_obj}_tl_x_{cur_idx}'] = bbox[1]
        res_di[f'{cur_obj}_br_y_{cur_idx}'] = bbox[2]
        res_di[f'{cur_obj}_br_x_{cur_idx}'] = bbox[3]
    return res_di       

def res_to_df(res:'results obtained from "objects_in_categories function"',
                    object_list_key:'key (string) to access the list of detected class names' = 'detection_classes_translated',
                    detection_boxes_key:str = 'detection_boxes',
                    scores_key:str='detection_scores',
                    img_height_key:str='img_height',
                    img_width_key:str='img_width',
                    num_detections_key:str='num_detections',
                    id_col:'key (string) to access the unique ID for each entry in "res"'='file_id',
                    cat_str:'key (string) to access the category in "res" entries'='category') \
                    -> pd.DataFrame:
    res_li = []
    for r in res:
        di = counts_in_single_res(r[object_list_key], key_suffix='_count')
        di[cat_str] = r[cat_str]
        di[id_col] = r[id_col]
        di[img_height_key] = r[img_height_key]
        di[img_width_key] = r[img_width_key]
        di[num_detections_key] = r[num_detections_key]
        di.update(extract_scores_and_bbox(r, object_list_key, scores_key, detection_boxes_key))
        res_li.append(di)
    df = pd.DataFrame(res_li)
    return df

    
def _get_cooccurance(df:pd.DataFrame, col_one:str, col_two:str):
    '''
    returns the number of rows in df where both "col_one" and "col_two" entries are greater than 0.
    '''
    return ((df[col_one] > 0) & (df[col_two] > 0)).sum()

def object_correlation(df:pd.DataFrame, method:str='pearson', threshold:int=0) -> pd.DataFrame:
    '''
    df - DataFrame where the rows correspond to results on individual images. The columns are object names and the 
        values in each cell is a metric for an object in an image (Ex. number of occurances of an object in an image)
    method - can be one of {‘pearson’, ‘kendall’, ‘spearman’} or callable (callable with input 
        two 1d ndarrays and returning a float.)
    threshold - removes pairs where the number of images where both objects occur is less than the specified threshold.

    returns a pandas DataFrame where each row contains object names and their correlation score. The dataframe will be sorted
        in an ascending order. 
    '''
    corr_res = df.corr(method=method).fillna(0.)
    mask = np.triu(np.ones(corr_res.shape)).astype('bool')
    mask[list(range(mask.shape[0])), list(range(mask.shape[1]))] = False
    mask = mask.reshape(corr_res.size)
    res = corr_res.stack()[mask].reset_index()
    res.columns = ['object_1', 'object_2' , 'correlation']
    res = res.sort_values(by='correlation')
    res['occurance'] = res.apply(lambda s: _get_cooccurance(df, s['object_1'], s['object_2']) ,axis=1)
    res = res[res.occurance > threshold]
    return res
