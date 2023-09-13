import os
from tqdm import tqdm
import subprocess


def main():
    run_ids = ['1unrxvkz',
               '39g7zsyp',
               '35ap3ocp',
               '13de8eld',
               '77zv06xe']
    processes = []
    for run_id in tqdm(run_ids):
        src_path = f'~/Project/image_registration/dual_cervix_registration/work_dirs/pix2pix_UnetGenerator_NLayerDiscriminator_color/{run_id}/visualization'
        output_path = f'~/Downloads/lab/visualization/GAN/Run_19/visualization_{run_id}'
        if not os.path.exists(os.path.expanduser(os.path.dirname(output_path))):
            os.makedirs(os.path.expanduser(os.path.dirname(output_path)))
        cmd = f'scp -r lab_243:{src_path} {output_path}'
        processes.append(subprocess.Popen(cmd, stdout = subprocess.PIPE, shell = True))

    for p in tqdm(processes):
        res = p.communicate()
        print(res)


if __name__ == '__main__':
    main()
