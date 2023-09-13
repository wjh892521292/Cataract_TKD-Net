import torch
from mmdet.datasets.pipelines import Compose, Normalize
from tqdm import trange

from utils import CLI


def add_item(sum_dict, value, num):
    sum_dict['value'] = sum_dict['value'] * sum_dict['num'] / (sum_dict['num'] + num)
    sum_dict['value'] = sum_dict['value'] + value * (num / (sum_dict['num'] + num))
    sum_dict['num'] += num


def get_pipeline(pipeline):
    transforms = pipeline.transforms
    for i in range(len(transforms)):
        if isinstance(transforms[i], Normalize) or transforms[i].__class__.__name__ == 'Normalize':
            break
    transforms[i].mean[...] = 0
    transforms[i].std[...] = 1
    return Compose(transforms[:i + 1])


def calculate_on_part(get_data_fn, dataset, device = 'cpu'):
    mean, std = [{'value': 0, 'num': 0} for _ in range(2)]
    with torch.no_grad():
        for i in trange(len(dataset)):
            res = get_data_fn(dataset, i)
            img = torch.tensor(res['img'], dtype = torch.float, device = device)
            m = torch.mean(img, dim = [0, 1])
            n = img.shape[0] * img.shape[1]
            add_item(mean, m, n)

        for i in trange(len(dataset)):
            res = get_data_fn(dataset, i)
            img = torch.pow(torch.tensor(res['img'], dtype = torch.float, device = device) - mean['value'], 2)
            s = torch.mean(img, dim = [0, 1])
            n = img.shape[0] * img.shape[1]
            add_item(std, s, n)
        std['value'] = torch.sqrt(std['value'])
    return mean['value'].cpu().numpy().tolist(), std['value'].cpu().numpy().tolist()


def main():
    device = 'cpu'
    cli = CLI(run = False)
    cli.datamodule._setup_dataset('train')
    dataset = cli.datamodule.datasets['train']
    pipeline = dataset.pipeline

    if isinstance(pipeline, list):
        dataset.pipeline = [get_pipeline(p) for p in pipeline]

        mean, std = [], []
        for index in range(len(pipeline)):
            m, s = calculate_on_part(lambda d, i: d[i][index], dataset, device)
            mean.append(m)
            std.append(s)
    elif isinstance(pipeline, dict):
        dataset.pipeline = {k: get_pipeline(pipeline[k]) for k in pipeline}

        mean, std = {}, {}
        for k in pipeline:
            mean[k], std[k] = calculate_on_part(lambda d, i: d[i][k], dataset, device)
    else:
        dataset.pipeline = get_pipeline(pipeline)
        mean, std = calculate_on_part(lambda d, i: d[i], dataset, device)

    print(f'mean: {mean}')
    print(f'std: {std}')


if __name__ == '__main__':
    main()
