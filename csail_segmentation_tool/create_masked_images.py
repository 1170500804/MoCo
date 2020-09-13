from csail_segmentation_tool import csail_segmentation
import argparse
import pandas as pd
from PIL import Image
from torchvision import transforms
import os

parser = argparse.ArgumentParser('creating images')
parser.add_argument('--in-images', type=str, help='the file describing raw images')
parser.add_argument('--out-dir', type=str, help='the directory to which the program outputs')

args = parser.parse_args()
Mask = csail_segmentation.MaskBuilding()
in_df = pd.read_csv(args.in_images)
out = {'old_path':[], 'new_path':[]}
for i in range(len(in_df)):
    type = in_df.iloc[i, 2]
    path = in_df.iloc[i, 1]
    if path.endswith('/'):
        img_name = path.split('/')[-2]
    else:
        img_name = path.split('/')[-1]
    image = Image.open(path)
    image = transforms.ToPILImage()(image).convert("RGB")
    new_path = os.path.join(args.out_dir, img_name)
    image.save(new_path, "JPG")
    out['new_path'].append(new_path)
    out['old_path'].append(path)
df_out = pd.DataFrame(out)
df_out.to_csv('./mask_lookup.csv')