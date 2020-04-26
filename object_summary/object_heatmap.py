import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
from functools import partial

def filter_by_hw(df:pd.DataFrame, keep_width:'list(int)', keep_height:'list(int)', img_width:str, img_height:str) -> pd.DataFrame:
    '''
    Keeps items that have width and height among the values mentioned in the "keep_width"
    and "keep_height" parameter. 
    Returns the filtered DataFrame
    '''
    df = df[(df[img_width].isin(keep_width)) & (df[img_height].isin(keep_height) )]
    return df

def object_counts(df:pd.DataFrame, objects:'list(str)'):
    df_counts = df[[o + '_count' for o in objects]].fillna(0.)
    df_counts.columns = objects
    top_object_counts = df_counts.sum().sort_values(ascending=False)
#     objects = list(top_object_counts.index)
    return top_object_counts

def sort_objects_using_counts(df:pd.DataFrame, objects:'list(str)'):
    top_obj_counts = object_counts(df, objects)
    objects = list(top_obj_counts.index)
    return objects

def filter_top(df, objects,top=30):
    return objects[:top]

# def draw_bounding_box_heatmap(df:pd.DataFrame, width:int, height:int, 
#                               objects:'list(str) - list of unique objects in the dataframe',
#                               tl_x = 'tl_x', tl_y = 'tl_y', br_x = 'br_x', br_y = 'br_y', 
#                               separator = '_', img_width = 'img_width', 
#                               img_height = 'img_height', sort_objects=None, 
#                               filter_objects=filter_top, cmap='viridis'):
#     '''
#     df_full - the pandas DataFrame containing the results obtained from "res_to_df"
#     '''
#     if sort_objects:
#         objects = sort_objects(df, objects)
#     if filter_objects:
#         objects = filter_objects(df, objects)
    
#     for obj in objects:
#         print(f'Object = {obj}')
#         agg = np.zeros((df.iloc[0][img_height], df.iloc[0][img_width]))
#         obj_tl_x = obj + separator + tl_x
#         obj_tl_y = obj + separator + tl_y
#         obj_br_x = obj + separator + br_x
#         obj_br_y = obj + separator + br_y
#         df_obj = df[~(df[f'{obj}{separator}count'].isna())]
#         for i in range(df_obj.shape[0]):
#             s = df_obj.iloc[i]
#             for j in range(int(s[f'{obj}{separator}count'])):
#                 tl_y_val = int(round(s[obj_tl_y + f'{separator}{j}'] * height))
#                 tl_x_val = int(round(s[obj_tl_x + f'{separator}{j}'] * width))
#                 br_y_val = int(round(s[obj_br_y + f'{separator}{j}'] * height))
#                 br_x_val = int(round(s[obj_br_x + f'{separator}{j}'] * width))
#                 agg[tl_y_val:br_y_val, tl_x_val:br_x_val] += 1
#         plt.figure(figsize=(9, 14))
#         plt.imshow(agg, cmap=cmap)
#         plt.colorbar()
#         plt.show()

def draw_bounding_box_heatmap(df:pd.DataFrame, width:int, height:int, 
                              objects:'list(str) - list of unique objects in the dataframe',
                              tl_x = 'tl_x', tl_y = 'tl_y', br_x = 'br_x', br_y = 'br_y', 
                              separator = '_', img_width = 'img_width', 
                              img_height = 'img_height', sort_objects=None, 
                              filter_objects=filter_top, cmap='viridis'):
    '''
    df_full - the pandas DataFrame containing the results obtained from "res_to_df"
    '''
    if sort_objects:
        objects = sort_objects(df, objects)
    if filter_objects:
        objects = filter_objects(df, objects)
    
    for obj in objects:
        print(f'Object = {obj}')
        agg = np.zeros((height, width))
        obj_tl_x = obj + separator + tl_x
        obj_tl_y = obj + separator + tl_y
        obj_br_x = obj + separator + br_x
        obj_br_y = obj + separator + br_y
        df_obj = df[~(df[f'{obj}{separator}count'].isna())]
        for i in range(df_obj.shape[0]):
            s = df_obj.iloc[i]
            for j in range(int(s[f'{obj}{separator}count'])):
                tl_y_val = int(round(s[obj_tl_y + f'{separator}{j}'] * height))
                tl_x_val = int(round(s[obj_tl_x + f'{separator}{j}'] * width))
                br_y_val = int(round(s[obj_br_y + f'{separator}{j}'] * height))
                br_x_val = int(round(s[obj_br_x + f'{separator}{j}'] * width))
                agg[tl_y_val:br_y_val, max(min(tl_x_val, width-1)-8, 0): min(tl_x_val, width-1)] += 1
                agg[tl_y_val:br_y_val, max(min(br_x_val, width-1)-8, 0): min(br_x_val, width-1)] += 1
                agg[max(min(tl_y_val,height-1)-8, 0): min(tl_y_val,height-1) , tl_x_val:br_x_val] += 1
                agg[max(min(br_y_val,height-1)-8, 0): min(br_y_val,height-1) , tl_x_val:br_x_val] += 1
#                 agg[tl_y_val:br_y_val, tl_x_val:br_x_val] += 1
        plt.figure(figsize=(9, 14))
        plt.imshow(agg, cmap=cmap)
        plt.colorbar()
        plt.show()
        
def filter_and_draw_bounding_box_heatmap(df:pd.DataFrame, keep_width:'list(int)', keep_height:'list(int)',
                                         width:int, height:int, 
                              objects:'list(str) - list of unique objects in the dataframe',
                              tl_x = 'tl_x', tl_y = 'tl_y', br_x = 'br_x', br_y = 'br_y', 
                              separator = '_', img_width = 'img_width', 
                              img_height = 'img_height', num_top_objects=30,
                             cmap='viridis'):
    '''
    df_full - the pandas DataFrame containing the results obtained from "res_to_df"
    '''
    df = filter_by_hw(df, keep_width, keep_height, img_width, img_height)
    
    draw_bounding_box_heatmap(df, width, height, objects, tl_x, tl_y, br_x, br_y, separator, 
                              img_width, img_height, sort_objects_using_counts, 
                              partial(filter_top, top=num_top_objects), cmap)