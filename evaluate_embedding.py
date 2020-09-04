import argparse
from sklearn.manifold import t_sne
import pandas as pd
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from Datasets import cluster_year_built_dataset
import moco.loader
import moco.builder
from torchvision import models
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
    val_dataset = cluster_year_built_dataset(args.batch_size,'year_built', args.train_data, args.data, transform=augmentation_val, )
    dataloader = DataLoader(val_dataset,shuffle=False,
            num_workers=args.workers, pin_memory=True)
    embeddings = {'embedding':[], 'year_built':[]}
    # load model
    model = models.__dict__[args.arch]()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]



        print("=> loaded pre-trained model '{}'".format(args.pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(args.pretrained))
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