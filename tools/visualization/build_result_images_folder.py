import os
from tqdm import tqdm
import shutil


def main():
    data_path = '/data/zhengwenhao/Datasets/object_detection/TCTDataSet/middle_results/outputs/final_images/feature_map'
    output_path = 'images'
    output_path = os.path.join(data_path, output_path)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for dir_name in tqdm(os.listdir(data_path), desc = 'dir'):
        dir_path = os.path.join(data_path, dir_name)
        if os.path.isdir(dir_path) and dir_path != output_path:
            for file_name in tqdm(os.listdir(dir_path), desc = 'file', leave = False):
                if file_name.endswith(('.jpg', '.png')):
                    output_name = os.path.splitext(file_name)
                    output_name = output_name[0] + f'_{dir_name}' + output_name[1]
                    shutil.copy2(os.path.join(dir_path, file_name), os.path.join(output_path, output_name))


if __name__ == '__main__':
    main()
