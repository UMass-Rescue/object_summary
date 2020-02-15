from pathlib import Path
import pandas as pd
from functools import reduce
from tf_object_detection_util.inference_api import TFInference
import cv2
import json
import numpy as np
from object_detection.utils import label_map_util
from collections import defaultdict

def ls(path): return [f for f in path.glob('*')]

fmts_to_regex = lambda img_fmts : '|'.join(map(lambda s: f'.({s})', img_fmts))

def ls_images(path, img_fmts = ['jpg', 'jpeg', 'png']):
    return [p for p in path.glob(f'*[{fmts_to_regex(img_fmts)}]' )]

def category_path_df(path):
    return pd.DataFrame([{'path': str(p.absolute()), 'category': path.stem} for p in ls_images(path)])

def clf_folders_to_df(path):
    dirs = ls(path)
    if len(dirs) <= 0:
        return None
    dirs = list(filter(lambda d: d.is_dir(), dirs ))
    return reduce(lambda d1, d2: category_path_df(d1).append(category_path_df(d2), ignore_index=True), dirs)

def cv2_imread_rgb(uri):
    return cv2.cvtColor(cv2.imread(str(uri)), cv2.COLOR_BGR2RGB)

def dump_to_json_file(obj, out_path):
    with open(out_path, 'w') as f:
        f.write(json.dumps(obj, cls=NumpyEncoder))

def read_json_from_file(in_path):
    with open(in_path) as f:
        return json.loads(f.read())
    
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_faster_rcnn_open_images():
    MODEL_NAME = Path('/home/prasannals/models/research/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12')
    PATH_TO_FROZEN_GRAPH = MODEL_NAME / 'frozen_inference_graph.pb'
    PATH_TO_LABELS = MODEL_NAME / 'oid_v4_label_map.pbtxt'

    inf = TFInference(PATH_TO_FROZEN_GRAPH, PATH_TO_LABELS)
    return inf

def objects_in_categories(path_df, inf, out_path, visualize=False):
    out_path = Path(out_path) if type(out_path) == str else out_path

    all_res = []
    for i in range(path_df.shape[0]):
        ser = path_df.iloc[i]
        img_path, category = ser['path'], ser['category']
        res, res_img = inf.predict(img_path, visualize=visualize)
        if visualize:
            cv2.imwrite(str(out_path / f'{i}.jpg'), res_img)
#         res['file'] = img_path
        res['category'] = category
        all_res.append(res)

    return all_res

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
