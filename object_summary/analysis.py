from collections import defaultdict
import pandas as pd

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

