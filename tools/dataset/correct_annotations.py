import json
import os
import shutil


def main():
    ann_path = 'data/TCTDetection/annotations'
    output_path = 'outputs'
    if os.path.exists(os.path.join(ann_path, output_path)):
        shutil.rmtree(os.path.join(ann_path, output_path))
    os.mkdir(os.path.join(ann_path, output_path))

    for name in [n for n in os.listdir(ann_path) if n.endswith('.json')]:
        ann = json.load(open(os.path.join(ann_path, name)))
        for k in [k for k in ann if k not in ['info', 'licenses', 'images', 'annotations', 'categories']]:
            del ann[k]
        if 'images' in ann:
            image_id_plus_1 = min([int(i['id']) for i in ann['images']]) <= 0
            for i in range(len(ann['images'])):
                if 'filename' in ann['images'][i] and 'file_name' not in ann['images'][i]:
                    ann['images'][i]['file_name'] = ann['images'][i]['filename']
                    del ann['images'][i]['filename']
                if image_id_plus_1:
                    ann['images'][i]['id'] = int(ann['images'][i]['id']) + 1
        else:
            image_id_plus_1 = False

        if 'categories' in ann:
            category_id_plus_1 = min([int(i['id']) for i in ann['categories']]) <= 0
            for i in range(len(ann['categories'])):
                if category_id_plus_1:
                    ann['categories'][i]['id'] = int(ann['categories'][i]['id']) + 1
        else:
            category_id_plus_1 = False

        if 'annotations' in ann:
            annotation_id_plus_1 = min([int(i['id']) for i in ann['annotations']]) <= 0
            for i in range(len(ann['annotations'])):
                if image_id_plus_1:
                    ann['annotations'][i]['image_id'] = int(ann['annotations'][i]['image_id']) + 1
                if category_id_plus_1:
                    ann['annotations'][i]['category_id'] = int(ann['annotations'][i]['category_id']) + 1
                if annotation_id_plus_1:
                    ann['annotations'][i]['id'] = int(ann['annotations'][i]['id']) + 1
        json.dump(ann, open(os.path.join(ann_path, output_path, name), 'w'), indent = 4)


if __name__ == '__main__':
    main()
