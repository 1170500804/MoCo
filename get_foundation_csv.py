import pandas as pd
import os
import random
image_folder = '/home/saschaho/Simcenter/Foundation_Images_Orig'
dataset = {}
dataset['filename'] =  []
dataset['year_built'] = []
min_year = 1913
max_year = 2013
for type_ in os.listdir(image_folder):
    for img in os.listdir(os.path.join(image_folder, type_)):
        if(img.endswith('.jpg')):
            dataset['filename'].append(os.path.join(type_,img))
            dataset['year_built'].append(random.randint(1913, 2013))
dataset_df = pd.DataFrame(dataset)
dataset_df.to_csv('foundation_type.csv')