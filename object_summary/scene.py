from scene_detection import SceneDetector
from tqdm import tqdm
from tinydb import TinyDB
from .util import remove_done_files
from pathlib import Path
import uuid

def scene_detect(path_df:'DataFrame - containing "path" and "category" columns', 
    out_path:'str or pathlib.Path - path to save the results database', 
    db_name:'str, the name for the db (no file extension. just the name)', 
    filemap_name:'str:Name of the DB for mapping between unique file id and file') -> 'list of dictionary containing the results':
    out_path = Path(out_path) if type(out_path) == str else out_path
    model = SceneDetector()
    db = TinyDB(str(out_path / (db_name + '.json')))
    filemap_db = TinyDB(str(out_path/ (filemap_name + '.json')))
    path_df = remove_done_files(path_df, filemap_db)
    for i in tqdm(range(path_df.shape[0])):
        ser = path_df.iloc[i]
        img_path, category = ser['path'], ser['category']
        res = model.predict_from_path(img_path)
        file_id = uuid.uuid4().hex
        res['file_id'] = file_id
        res['category'] = category
        db.insert(res)
        filemap_db.insert({file_id:img_path})

    return db.all(), filemap_db.all()
