import argparse
from sklearn.manifold import t_sne
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
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # load model
    model = models.__dict__[args.arch]()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
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
    if args.embedding_file:
        pass
    else:
        # compute embeddings
        model.eval()
        labels = None
        embd = None
        for i, (images, targets) in enumerate(dataloader):
            images = images.cuda()
            targets = targets.cuda()
            print('dim_{}'.format(model.size(1)))
            assert (model.size(1) == 128)
            output = model(images)
            if labels == None:
                labels = targets
                embd = output
            else:
                labels = torch.cat([labels, targets], dim=0)
                embd = torch.cat([embd, output], dim=0)
            print('dim_{}'.format(model.size(1)))
            assert(model.size(1) == 128)
        labels = labels.cpu().tolist()
        embd = embd.cpu().tolist()
        embeddings['embedding'] = embd
        embeddings['year_built'] = labels

        df = pd.DataFrame(embeddings)
        if (args.to_csv):
            df.to_csv('/home/shuai/MoCo_stats/embedding/embeddings_'+args.resume+'.csv')
if __name__ == '__main__':
    if not os.path.exists('/home/shuai/MoCo_stats/embedding'):
        os.mkdir('/home/shuai/MoCo_stats/embedding')
    main()