from __future__ import print_function
import time

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pandas as pd
import os
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributed as dist

from Datasets import Rolling_Window_Year_Dataset
import moco.loader
import moco.builder
from torchvision import models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
# /home/shuai/MoCo_stats/unsupervised_pretrained_20200904035252/checkpoint_0199.pth.tar
def plot_t_sne(data_subset_embd, data_subset_label, filename):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=3000)
    tsne_results = tsne.fit_transform(data_subset_embd)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    plt.figure(figsize=(15, 15))
    # sns.set()

    sns_plot = sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=data_subset_label,
        palette=sns.cubehelix_palette(dark=.1, light=.9, hue=1,as_cmap=True, n_colors=len(np.unique(label))),
        legend="full",
        s=30,
        alpha=0.3
    )
    sns_plot.figure.savefig('/home/shuai/MoCo_stats/embedding/'+filename+'.png')
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
    parser.add_argument('--embedding-file-e', type=str, help='if provide the embedding file')
    parser.add_argument('--embedding-file-l', type=str, help='if provide the label file')
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
    parser.add_argument('--step', default=10, type=int, help='the step of the rolling window')

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

    if args.embedding_file_e or args.embedding_file_l:
        # /home/shuai/MoCo_stats/embedding/09032020checkpoint_0137.csv
        if(args.embedding_file_l and args.embedding_file_e):
            embd = np.loadtxt(args.embedding_file_e)
            label = np.loadtxt(args.embedding_file_l)
            label = np.asarray(label)
            label = pd.Series(label, dtype=int)
        else:
            print('require embedding/label file!')
            exit()

        if(args.embedding_file_e.endswith('/')):
            filename = (args.embedding_file_e.split('/')[-2]).split('.')[0]
        else:
            filename = (args.embedding_file_e.split('/')[-1]).split('.')[0]

        plot_t_sne(embd, label, filename)
    else:
        val_dataset = Rolling_Window_Year_Dataset(args.batch_size, 'year_built', args.validate_data, args.data,
                                                 transform=augmentation_val, step=args.step)
        dataloader = DataLoader(val_dataset, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        embeddings = {'embedding': [], 'year_built': []}

        # parallel training
        # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                         world_size=args.world_size, rank=args.rank)
        # load model
        model = models.__dict__[args.arch]()
        # model = torch.nn.parallel.DistributedDataParallel(model)
        model = model.cuda()
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
        if (args.to_csv):
            name = args.resume.split('/')
            if(args.resume.endswith('/')):
                name = name[-3].split('_')[-1] + name[-2].split('.')[0]
                print(name)
            else:
                name = name[-2].split('_')[-1] + name[-1].split('.')[0]
            print(name)
        # compute embeddings
        model.eval()
        labels = None
        embd = None
        print('Dataset length: {}'.format(val_dataset.__len__()))
        iter = val_dataset.__len__()/args.batch_size
        for i, (images, targets) in enumerate(dataloader):
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
                labels = targets.cpu().detach()
                embd = output.cpu().detach()
            else:
                labels = torch.cat([labels, targets.cpu().detach()], dim=0)
                embd = torch.cat([embd, output.cpu().detach()], dim=0)
        # embeddings['embedding'] = embd
        # embeddings['year_built'] = labels
        # df = pd.DataFrame(embeddings)
        embd = np.array(embd)
        labels = np.array(labels)
        print(embd.shape)
        print(labels.shape)
        if (args.to_csv):
            np.savetxt('/home/shuai/MoCo_stats/embedding/'+name+'_embd.csv', embd)
            np.savetxt('/home/shuai/MoCo_stats/embedding/'+name+'_label.csv', labels)
if __name__ == '__main__':
    if not os.path.exists('/home/shuai/MoCo_stats/embedding'):
        os.mkdir('/home/shuai/MoCo_stats/embedding')
    main()
