from collections import defaultdict

from mmdet.datasets.pipelines import Compose, LoadImageFromFile
from tqdm import trange

from utils import CLI


def get_pipeline(pipeline):
    transforms = pipeline.transforms
    for i in range(len(transforms)):
        if isinstance(transforms[i], LoadImageFromFile) or transforms[i].__class__.__name__ == 'LoadImageFromFile':
            break
    return Compose(transforms[:i + 1])


def calculate_on_part(get_data_fn, dataset):
    img_shape = defaultdict(int)
    for i in trange(len(dataset)):
        res = get_data_fn(dataset, i)
        img_shape[res['img_shape'][:2][::-1]] += 1
    return img_shape


def main():
    cli = CLI(run = False)
    cli.datamodule._setup_dataset('train')
    dataset = cli.datamodule.datasets['train']

    pipeline = dataset.pipeline
    if isinstance(pipeline, list):
        dataset.pipeline = [get_pipeline(p) for p in pipeline]
        img_shape = [calculate_on_part(lambda d, i: d[i][index], dataset) for index in range(len(pipeline))]
    elif isinstance(pipeline, dict):
        dataset.pipeline = {k: get_pipeline(pipeline[k]) for k in pipeline}
        img_shape = {k: calculate_on_part(lambda d, i: d[i][k], dataset) for k in pipeline}
    else:
        dataset.pipeline = get_pipeline(pipeline)
        img_shape = calculate_on_part(lambda d, i: d[i], dataset)

    print(img_shape)


if __name__ == '__main__':
    main()
