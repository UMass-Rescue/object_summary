from pathlib import Path
import pandas as pd
from functools import reduce
from tf_object_detection_util.inference_api import TFInference
import cv2
import json
import numpy as np
from object_detection.utils import label_map_util
from collections import defaultdict
from tinydb import TinyDB
import uuid
from tqdm import tqdm

def ls(path): return [f for f in path.glob('*')]

fmts_to_regex = lambda img_fmts : '|'.join(map(lambda s: f'.({s})', img_fmts))

def ls_images(path:Path, img_fmts:'list(str)' = ['jpg', 'jpeg', 'png']) -> 'list(pathlib.Path)':
    '''
    Returns files (list(pathlib.Path)) in "path" which have format = one of "img_fmts"
    '''
    return [p for p in path.glob(f'*[{fmts_to_regex(img_fmts)}]' )]

def folder_to_cat_df(path:Path) -> pd.DataFrame:
    '''
    path - a folder containing files
    Returns a DataFrame containing two columns - 'path' and 'category' where 'path' column contains
        the absolute path to a file and 'category' column contains the category of the file (this will 
        be the name of the folder)
    '''
    return pd.DataFrame([{'path': str(p.absolute()), 'category': path.stem} for p in ls_images(path)])

def clf_folders_to_df(path:Path) -> pd.DataFrame:
    '''
    path - path to the folder containing other folders that contain files. Each file (not a directory file) 
        will be labelled with category = parent folder of a file.
    returns - pandas DataFrame containing two columns - "path" and "category" where each path is mapped
        to its corresponding category.
    '''
    dirs = ls(path)
    if len(dirs) <= 0:
        return None
    dirs = list(filter(lambda d: d.is_dir(), dirs ))
    return reduce(lambda d1, d2: d1.append(folder_to_cat_df(d2), ignore_index=True), dirs, 
                    pd.DataFrame({'path':[], 'category':[]}))

def cv2_imread_rgb(uri:'str or pathlib.Path') -> np.ndarray:
    '''
    reads an image at the URI specified in "uri" as an RGB image 
    returns the output as a numpy ndarray of shape (height, width, 3)
    '''
    return cv2.cvtColor(cv2.imread(str(uri)), cv2.COLOR_BGR2RGB)

def dump_to_json_file(obj:'any python object. prefferably a list or dict or tuple', out_path:'str or pathlib.Path'):
    '''
    saves "obj" in the specified file at "out_path"

    "read_json_from_file" can be used to read in the saved object at "out_path"
    '''
    with open(out_path, 'w') as f:
        f.write(json.dumps(obj, cls=NumpyEncoder))

def read_json_from_file(in_path:'str or pathlib.Path'):
    '''
    reads the python object at "in_path" and returns the object

    The function expects that "dump_to_json_file" function is used to save the 
    python object to "in_path".
    '''
    with open(in_path) as f:
        return json.loads(f.read())
    
class NumpyEncoder(json.JSONEncoder):
    '''
    Encodes NumPy ndarrays into json.dumps compatible data format
    '''
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def np_to_list(di:dict):
    '''
    Converts all values in "di" dictionary that are numpy ndarrays to a python list

    WARNING - The conversion is done inplace. i.e. this function modifies "di".
    '''
    for k, v in di.items():
        if isinstance(v, np.ndarray):
            di[k] = v.tolist()

def remove_done_files(path_df:pd.DataFrame, filemap_db:TinyDB) -> pd.DataFrame:
    '''
    Removes rows in "path_df" where the column "path" is already present in the "filemap_db" database
    returns a pandas DataFrame with all the rows removed whose "path" is present in "filemap_db".
    '''
    already_done = filemap_db.all()
    already_done_paths = [list(e.values())[0] for e in already_done]
    done = path_df.path.isin(already_done_paths)
    if done.sum() > 0:
        print(f'Found {done.sum()} pre existing results in database. Ignoring these files and resuming the object detection...')
    return path_df[~done]

def objects_in_categories(path_df:'DataFrame - containing "path" and "category" columns', inf:TFInference, 
    out_path:'str or pathlib.Path - path to save the visualizations and results database', db_name:'str, the name for the db (no file extension. just the name)', 
    filemap_name:'str:Name of the DB for mapping between unique file id and file',visualize:bool=False) -> 'list of dictionary containing the results':
    out_path = Path(out_path) if type(out_path) == str else out_path

    db = TinyDB(str(out_path / (db_name + '.json')))
    filemap_db = TinyDB(str(out_path/ (filemap_name + '.json')))
    path_df = remove_done_files(path_df, filemap_db)
    for i in tqdm(range(path_df.shape[0])):
        ser = path_df.iloc[i]
        img_path, category = ser['path'], ser['category']
        res, res_img = inf.predict(img_path, visualize=visualize)
        if visualize:
            cv2.imwrite(str(out_path / f'{i}.jpg'), cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR  ))
#         res['file'] = img_path
        file_id = uuid.uuid4().hex
        res['file_id'] = file_id
        res['category'] = category
        np_to_list(res)
        db.insert(res)
        filemap_db.insert({file_id:img_path})

    return db.all(), filemap_db.all()

def objects_in_folder(folder:'str or pathlib.Path - path to folder to be analyzed', inf:TFInference, 
    out_path:'str or pathlib.Path - path to save the visualizations', visualize=False) -> 'list of dictionary containing the results':
    folder = Path(folder) if type(folder) == str else folder
    imgs = ls(folder)
    paths = {'path':[str(f) for f in imgs],
            'category': [folder for i in range(len(imgs))]}
    res = objects_in_categories(pd.DataFrame(paths), inf, out_path, visualize=visualize)
    return res

def count_objects(raw_res, max_objects, object_names, threshold=0.0):
    di = defaultdict(lambda : np.zeros(max_objects))
    for res in raw_res:
        detection_classes = np.array(res['detection_classes']) - 1
        detection_scores = res['detection_scores']
        cur_cat_count = di[res['category']]
        for i in range(len(detection_classes)):
            if detection_scores[i] > threshold:
                cur_cat_count[detection_classes[i]] += 1
    return pd.DataFrame(di, index=object_names)

def pbtxt_object_list(pbtxt_path):
    object_index = label_map_util.create_category_index_from_labelmap(str(pbtxt_path),use_display_name=True)
    num_objects = max(object_index.keys())
    object_list = [object_index[k]['name'] for k in sorted(object_index.keys())]
    return object_list, num_objects
