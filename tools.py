from collections import defaultdict
from million._config import NULL_VALUE
import cPickle
import json
import numpy as np
import os
import seamless as ss
import logging

def delete_some_outliers(df, targets):
    outlier_ixs = (targets > 0.419) | (targets < -0.4)
    filter_ixs = np.array([(np.random.normal() > 0.5) & o for o in outlier_ixs])
    return df.iloc[~filter_ixs], targets[~filter_ixs]

def ensemble_preds(predictions, weights):
    return np.average(predictions, weights=weights, axis=0)

def get_mae_loss(y, ypred):
    return np.sum([abs(y[i]-ypred[i]) for i in range(len(y))]) / float(len(y))

def get_test_ixs(targets):
    ixs = targets == NULL_VALUE
    return ixs

def ix_to_bool(ix, length):
    boolean_mask = np.repeat(False, length)
    boolean_mask[ix] = True
    return boolean_mask

def get_group_ixs(*group_ids, **kwargs):
    """ Returns a dictionary {groupby_id: group_ix}.

    group_ids:
        List of IDs to groupbyy
    kwargs:
        bools = True or False, if True returns a boolean array
    """
    group_ids = _ensure_group_ids_hashable(group_ids)
    grouped_ixs = _get_group_ixs(group_ids)
    grouped_ixs = _convert_int_indices_to_bool_indices_if_necessary(grouped_ixs, kwargs)
    return grouped_ixs

def _ensure_group_ids_hashable(group_ids):
    if len(group_ids) == 1:
        combined_group_ids = group_ids[0]
    else:
        combined_group_ids = zip(*group_ids)

    is_list_of_list = lambda ids: isinstance(ids[0], list)
    is_matrix = lambda ids: isinstance(ids, np.ndarray) and ids.ndim == 2
    if is_list_of_list(combined_group_ids) or is_matrix(combined_group_ids):
        hashable_group_ids = [tuple(group_id) for group_id in combined_group_ids]
    else:
        hashable_group_ids = combined_group_ids

    return hashable_group_ids

def _convert_int_indices_to_bool_indices_if_necessary(ixs, kwargs):
    bools = kwargs.get('bools', False)
    if bools:
        length = np.sum([len(v) for v in ixs.itervalues()])
        ix_to_bool = lambda v, length: ix_to_bool(v, length)
        ixs = {k: ix_to_bool(v, length) for k, v in ixs.iteritems()}
    return ixs

def get_logger(file_name):
    logger_format = '%(asctime)s - %(filename)s - %(message)s'
    logging.basicConfig(filename=file_name, level=logging.DEBUG, format=logger_format)
    logger = logging.getLogger()
    return logger

def _get_group_ixs(ids):
    """ Returns a dictionary {groupby_id: group_ix}.

    ** Code Hall Of Fame **.
    """
    id_hash = defaultdict(list)
    for j, key in enumerate(ids):
        id_hash[key].append(j)
    id_hash = {k:np.array(v) for k,v in id_hash.iteritems()}
    return id_hash

def read_pickle(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj

def write_pickle(obj, path):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def dropbox():
    path = os.path.expanduser('~/Dropbox/')
    if not os.path.isdir(path):
        path = '/Dropbox/'
    return path

def experiments():
    dropbox_path = dropbox()
    path = dropbox_path + 'experiments/'
    return path

def write_results_to_json(results_dict, path):
    with open(path, 'a') as f:
        json_format_data = json.dumps(results_dict)
        f.write(json_format_data + '\n')

def read_special_json(path):
    with open(path, 'r') as f:
        raw = f.read()
    raw_data = raw.split('\n')
    data = []
    for d in raw_data:
        if d:
            data.append(json.loads(d))
    return ss.DDF(data)

def convert_to_python_types(dic):
    for key in dic:
        dic[key] = dic[key].item()
    return dic

