from tqdm import tqdm
from tinydb import TinyDB
from pathlib import Path
import uuid
import os
import pandas as pd

def remove_done_files(path_df:pd.DataFrame, filemap_db:TinyDB) -> pd.DataFrame:
    '''
    Removes rows in "path_df" where the column "path" is already present in the "filemap_db" database
    returns a pandas DataFrame with all the rows removed whose "path" is present in "filemap_db".
    '''
    already_done = filemap_db.all()
    already_done_paths = [list(e.values())[0] for e in already_done]
    done = path_df.path.isin(already_done_paths)
    if done.sum() > 0:
        print(f'Found {done.sum()} pre existing results in database. Ignoring these files.')
    return path_df[~done]

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

DATA_DIR = os.path.join(__location__, 'data')

def load_labels():
    with open(os.path.join(DATA_DIR, 'categories_places365.txt')) as f:
        categories = [l.strip().split(' ')[0][3:] for l in f.readlines()]
    
    with open(os.path.join(DATA_DIR, 'labels_sunattribute.txt')) as f:
        scene_attrs = [l.strip() for l in f.readlines()]
    
    return categories, scene_attrs


def scene_detect(path_df:'DataFrame - containing "path" and "category" columns', model : 'SceneDetector',
    out_path:'str or pathlib.Path - path to save the results database', 
    db_name:'str, the name for the db (no file extension. just the name)', 
    filemap_name:'str:Name of the DB for mapping between unique file id and file',
    insert_every:'int: inserts the results into DB after the specified number of scene detections'=1000) -> 'list of dictionary containing the results':
    out_path = Path(out_path) if type(out_path) == str else out_path
    db = TinyDB(str(out_path / (db_name + '.json')))
    filemap_db = TinyDB(str(out_path/ (filemap_name + '.json')))
    path_df = remove_done_files(path_df, filemap_db)

    res_acc = []
    filemap_acc = []
    for i in tqdm(range(path_df.shape[0])):
        ser = path_df.iloc[i]
        img_path, category = ser['path'], ser['category']
        res = model.predict_from_path(img_path)
        file_id = uuid.uuid4().hex
        res['file_id'] = file_id
        res['category'] = category
        
        res_acc.append(res)
        filemap_acc.append({file_id:img_path})

        if len(res_acc) >= insert_every:
            db.insert_multiple(res_acc)
            filemap_db.insert_multiple(filemap_acc)
            res_acc, filemap_acc = [], []
            
    db.insert_multiple(res_acc)
    filemap_db.insert_multiple(filemap_acc)

    return db.all(), filemap_db.all()


def format_scene_res(res, categories, scene_attrs, scene_threshold=None, attr_prefix='attr_'):
    res_cvt = {}
    
    res_cvt['file_id'] = res['file_id']
    res_cvt['category'] = res['category']
    
    res_cvt['indoor'] = 1 if res['type_of_env'] == 'indoor' else 0
    
    res_cats = res['scene_categories']
    for cat in categories:
        if cat in res_cats:
            if scene_threshold:
                res_cvt[cat] = 1 if res_cats[cat] >= scene_threshold else 0
            else:
                res_cvt[cat] = res_cats[cat]
        else:
            res_cvt[cat] = 0

    res_sc_attrs = res['scene_attributes']
    for sc_attr in scene_attrs:
        res_cvt[attr_prefix + sc_attr] = 1 if sc_attr in res_sc_attrs else 0

    return res_cvt    

def scene_res_to_df(all_res, scene_threshold=None, attr_prefix='attr_'):
    categories, scene_attrs = load_labels()
    all_res_cvt = [format_scene_res(res, categories, scene_attrs, scene_threshold, attr_prefix) 
                        for res in all_res]
    df = pd.DataFrame(all_res_cvt)
    return df