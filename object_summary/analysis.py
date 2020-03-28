from collections import defaultdict
import pandas as pd
import numpy as np

def counts_in_single_res(res:'list(str)') -> 'defaultdict(int)':
    '''
    res - list of category strings
    returns - dictionary of form {objectname:count}
    '''
    object_counter = defaultdict(int)
    for e in res:
        object_counter[e] += 1
    return object_counter

def objects_in_categories_df(res:'results obtained from "objects_in_categories function"',
                            object_list_key:str = 'detection_classes_translated',
                            cat_str:'key (string) for accessing the category in "res" entries'='category') \
                            -> pd.DataFrame:
    counts = []
    for r in res:
        di = counts_in_single_res(r[object_list_key])
        di[cat_str] = r[cat_str]
        counts.append(di)
    count_df = pd.DataFrame(counts)
    count_df = count_df.fillna(0.0)
    return count_df

def get_counts_df(res:'results obtained from "objects_in_categories function"',
                    object_list_key:str = 'detection_classes_translated',
                    cat_str:'key (string) for accessing the category in "res" entries'='category') \
                    -> pd.DataFrame:
    '''
    res - list of dictionaries. Result obtained from "objects_in_categories" function
    returns - pandas DataFrame where rows are categories and columns are objects. a cell contains the number of 
    objects that have been found in a category
    '''
    df = objects_in_categories_df(res)
    count_df = df.groupby(by=cat_str).sum()
    count_df = count_df.sort_index(axis=1)
    return count_df

def _get_cooccurance(df, col_one, col_two):
    return ((df[col_one] > 0) & (df[col_two] > 0)).sum()

def object_correlation(df, method='pearson',threshold=0):
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
