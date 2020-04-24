from scene_detection import SceneDetector
from tqdm import tqdm
from tinydb import TinyDB
from .util import remove_done_files
from pathlib import Path
import uuid

def scene_detect(path_df:'DataFrame - containing "path" and "category" columns', model : SceneDetector,
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
