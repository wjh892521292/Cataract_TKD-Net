import os
import json
import mmcv
import shutil
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def xyxy2lxty(bbox):
    x, y, x2, y2 = bbox
    return [float(t) for t in [x, y, x2 - x, y2 - y]]


def is_bboxes_overlap(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_min = max(x1, x2)
    x_max = min(x1 + w1, x2 + w2)
    y_min = max(y1, y2)
    y_max = min(y1 + h1, y2 + h2)
    if x_max <= x_min or y_max <= y_min:
        return False
    else:
        return [x_min - x2, y_min - y2, x_max - x_min, y_max - y_min]


def bbox_area(bbox):
    x, y, w, h = bbox
    return w * h


def rm_and_create_dir(d):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)


def main():
    modal_dict = {'acid': '_2', 'iodine': '_3'}
    data_root = 'data/DualCervixDetection'
    crop_ann_path = os.path.join(data_root, 'surface/result_{}.pkl')
    ann_path = os.path.join(data_root, 'hsil_rereannos')
    output_ann_path = os.path.join(data_root, 'cropped_annos')
    image_prefix = os.path.join(data_root, 'img')
    output_prefix = os.path.join(data_root, 'cropped_img')

    rm_and_create_dir(output_prefix)
    rm_and_create_dir(output_ann_path)

    crop_ann = {}
    for part in modal_dict:
        cur_crop_ann = pickle.load(open(crop_ann_path.format(part), 'rb'))
        crop_ann.update({k + modal_dict[part]: v for k, v in cur_crop_ann.items()})

    # crop ann
    for json_file in sorted([f for f in os.listdir(ann_path) if f.endswith('.json')]):
        print(json_file)
        ann = json.load(open(os.path.join(ann_path, json_file)))
        res_ann = {
            'images': [],
            'annotations': [],
            'categories': ann['categories']
        }
        image_id = ann_id = 0

        img_id_to_anns = defaultdict(list)
        print('building ann index')
        for a in tqdm(ann['annotations']):
            img_id_to_anns[a['image_id']].append(a)

        for image in tqdm(ann['images']):
            image_name = image['file_name']
            cur_name = image_name.split('.')[0]
            if cur_name not in crop_ann:
                continue

            # crop img and save img
            crop_bbox = crop_ann[cur_name]
            img_path = os.path.join(image_prefix, image_name)
            output_path = os.path.join(output_prefix, image_name)
            img = mmcv.imread(img_path)
            img = mmcv.imcrop(img, np.array(crop_bbox))
            mmcv.imwrite(img, output_path)

            # crop bbox and save bbox
            crop_bbox = xyxy2lxty(crop_bbox)
            for a in img_id_to_anns[image['id']]:
                bbox_overlap = is_bboxes_overlap(a['bbox'], crop_bbox)
                if not bbox_overlap:
                    continue
                a['id'] = ann_id
                a['image_id'] = image_id
                a['bbox'] = bbox_overlap
                a['area'] = bbox_area(bbox_overlap)
                res_ann['annotations'].append(a)
                ann_id += 1

            # save img ann
            image['id'] = image_id
            x, y, w, h = crop_bbox
            image['height'] = h
            image['width'] = w
            res_ann['images'].append(image)
            image_id += 1

        # save ann
        json.dump(res_ann, open(os.path.join(output_ann_path, json_file), 'w'))


if __name__ == '__main__':
    main()
