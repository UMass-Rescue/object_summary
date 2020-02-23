from collections import defaultdict
import pandas as pd

def update_class_counts(count_dict, class_list):
    for c in class_list:
        count_dict[c] += 1
        
def get_counts_df(res):
    '''
    res - list of dictionaries. Result obtained from "objects_in_categories" function
    returns - pandas DataFrame where rows are categories and columns are objects. a cell contains the number of 
    objects that have been found in a category
    '''
    counts = defaultdict(lambda : defaultdict(int))
    for e in res:
        update_class_counts(counts[e['category']], e['detection_classes_translated'])
        
    count_df = pd.DataFrame(counts).transpose()
    count_df = count_df.sort_index(axis=1)
    return count_df