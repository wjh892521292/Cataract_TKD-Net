import os
import json

Modal_Dict = {'acid': '_2', 'iodine': '_3'}


def equal_fn(d, key = None):
    if key is not None:
        if key == 'file_name':
            return all(
                [d[modal].removesuffix(Modal_Dict[modal] + '.jpg') == d['acid'].removesuffix(Modal_Dict['acid'] + '.jpg') for modal in
                 Modal_Dict])
        elif key in ['area', 'bbox']:
            return True
    return all(d[modal] == d['acid'] for modal in Modal_Dict)


def is_match(ann_dict, key = None):
    if isinstance(ann_dict['acid'], dict):
        if not all([is_match({modal: ann_dict[modal][k] for modal in Modal_Dict}, k) for k in ann_dict['acid']]):
            return False
    elif isinstance(ann_dict['acid'], list):
        if not all([len(ann_dict[modal]) == len(ann_dict['acid']) for modal in Modal_Dict]):
            return False
        if not all([is_match({modal: ann_dict[modal][i] for modal in Modal_Dict}, key) for i in range(len(ann_dict['acid']))]):
            return False
    else:
        return equal_fn(ann_dict, key)
    return True


def main():
    ann_path = 'data/DualCervixDetection/hsil_rereannos'
    for split in ['train', 'val', 'test']:
        ann = {}
        for modal in Modal_Dict:
            ann[modal] = json.load(open(os.path.join(ann_path, f'{split}_{modal}.json')))
        print(is_match(ann_dict = ann))


if __name__ == '__main__':
    main()
