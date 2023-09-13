import os
import json
import mmcv
import shutil
import pickle
import numpy as np
from tqdm import tqdm


def main():
    modal_dict = {'acid': '_2', 'iodine': '_3'}
    data_root = 'data/DualCervixDetection'
    ann_path = os.path.join(data_root, 'surface/result_{}.pkl')
    image_path = os.path.join(data_root, 'hsil_rereannos')
    image_prefix = os.path.join(data_root, 'img')
    output_prefix = os.path.join(data_root, 'draw_crop_img')
    if os.path.exists(output_prefix):
        shutil.rmtree(output_prefix)
    os.makedirs(output_prefix)

    ann = {}
    for part in modal_dict:
        cur_ann = pickle.load(open(ann_path.format(part), 'rb'))
        ann.update({k + modal_dict[part]: v for k, v in cur_ann.items()})

    for json_file in sorted([f for f in os.listdir(image_path) if f.endswith('.json')]):
        print(json_file)
        image_names = [img['file_name'] for img in json.load(open(os.path.join(image_path, json_file)))['images']]
        no_find = 0
        for img_name in tqdm(image_names):
            cur_name = img_name.split('.')[0]
            img_path = os.path.join(image_prefix, img_name)
            output_path = os.path.join(output_prefix, img_name)
            if cur_name in ann:
                mmcv.imshow_bboxes(img_path, np.array(ann[cur_name])[np.newaxis], show = False, out_file = output_path)
            else:
                no_find += 1
                print(cur_name)
        print(f'{json_file} no find: {no_find}/{len(image_names)}')


if __name__ == '__main__':
    main()
