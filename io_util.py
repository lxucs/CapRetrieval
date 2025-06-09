import logging
import json
import pickle

logger = logging.getLogger(__name__)


def read_jsonlines(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonlines(file_path, instances):
    with open(file_path, 'w') as f:
        for inst in instances:
            f.write(f'{json.dumps(inst, ensure_ascii=False)}\n')


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def write_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def write_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def read_plain(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def write_plain(file_path, instance):
    with open(file_path, 'w') as f:
        f.write(f'{instance}\n')


def get_read_write_fct(ext):
    ext = ext.lower().strip()
    if ext == 'json':
        return read_json, write_json
    elif ext in ('jsonl', 'jsonlines'):
        return read_jsonlines, write_jsonlines
    elif ext in ('bin', 'pickle'):
        return read_pickle, write_pickle
    elif ext in ('txt',):
        return read_plain, write_plain
    else:
        raise ValueError(f'Unknown ext: {ext}')


def read(file_path):
    ext = file_path.split('.')[-1].strip().lower()
    return get_read_write_fct(ext)[0](file_path)


def write(file_path, data):
    ext = file_path.split('.')[-1].strip().lower()
    return get_read_write_fct(ext)[1](file_path, data)


def json_to_jsonl(id2inst):
    instances = []
    for inst_id, inst in id2inst.items():
        if 'id' in inst:
            assert inst_id == inst['id']
        else:
            inst['id'] = inst_id
        instances.append(inst)
    return instances


def jsonl_to_json(insts):
    return {inst['id']: inst for inst in insts}
