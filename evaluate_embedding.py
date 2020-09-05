from __future__ import print_function
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist

from Datasets import cluster_year_built_dataset
import moco.loader
import moco.builder
from torchvision import models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
def plot_t_sne(data_subset, filename):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    data_subset['tsne-2d-one'] = tsne_results[:, 0]
    data_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    # sns.set()

    sns_plot = sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="year_built",
        palette=sns.color_palette("hls", 10),
        data=data_subset,
        legend="full",
        alpha=0.3
    )
    sns_plot.savefig(filename)
def main():
    parser = argparse.ArgumentParser(description='PyTorch year_built Training Validation')
    parser.add_argument('--validate-data', type=str, help='the data to be visualized', required=True)
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data', default='', type=str,
                        help='path to dataset')
    parser.add_argument('--to-csv', action='store_true', help='save the embeddin as file in /home/shuai/MoCo_stats/embedding')
    parser.add_argument('--embedding-file', type=str, help='if provide the embedding file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')

    args = parser.parse_args()
    # prepare dataset
    normalize = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    augmentation_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    val_dataset = cluster_year_built_dataset(args.batch_size,'year_built', args.validate_data, args.data, transform=augmentation_val, )
    dataloader = DataLoader(val_dataset,shuffle=False,
            num_workers=args.workers, pin_memory=True)
    embeddings = {'embedding':[], 'year_built':[]}

    # parallel training
    # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
    #                         world_size=args.world_size, rank=args.rank)
    # load model
    model = models.__dict__[args.arch]()
    # model = torch.nn.parallel.DistributedDataParallel(model)
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]



        print("=> loaded pre-trained model '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=[0,1]).to(device)
    if args.embedding_file:
        pass
    else:
        if (args.to_csv):
            name = args.resume.split('/')
            if(args.resume.endswith('/')):
                name = name[-3].split('_')[-1] + name[-2].split['.'][0]
                print(name)
            else:
                name = name[-2].split('_')[-1] + name[-1].split['.'][0]
            print(name)
        # compute embeddings
        model.eval()
        labels = None
        embd = None
        print('Dataset length: {}'.format(val_dataset.__len__()))
        iter = val_dataset.__len__()/args.batch_size
        for i, (images, targets, _) in enumerate(dataloader):
            if(i%1000 == 0):
                print('{}/{}'.format(int(i/1000),int(iter/1000)))
            images = images.cuda()
            targets = targets.cuda()
            # print('dim_{}'.format(model.size(1)))
            # assert (model.size(1) == 128)
            output = model(images)
            # print('dim_{}'.format(output.size(1)))
            # assert (output.size(1) == 128)
            if labels == None:
                labels = embd = []
            # else:
            t = targets.cpu()
            o = output.cpu()
            if i == 0:
                print(o.size())
                print(t.size())
            labels.extend(t.tolist())# torch.cat([labels, targets], dim=0)
            embd.extend(o.tolist()) #= torch.cat([embd, output], dim=0)
            # print('dim_{}'.format(output.size(1)))
            # assert(model.size(1) == 128)
        # labels = labels.cpu().tolist()
        # embd = embd.cpu().tolist()
        embeddings['embedding'] = embd
        embeddings['year_built'] = labels

        df = pd.DataFrame(embeddings)
        if (args.to_csv):
            df.to_csv('/home/shuai/MoCo_stats/embedding/embeddings_'+name+'.csv')
if __name__ == '__main__':
    if not os.path.exists('/home/shuai/MoCo_stats/embedding'):
        os.mkdir('/home/shuai/MoCo_stats/embedding')
    main()