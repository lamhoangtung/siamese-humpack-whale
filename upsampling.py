from tqdm import tqdm
import imgaug as ia
from albumentations import *
import pandas as pd
import numpy as np
import cv2
import os
from config import *
import pickle

# For each images id, select the prefered image

train_root = TRAIN
upsampling_root = "./upsampling/"
os.makedirs(upsampling_root, exist_ok=True)

train_df = pd.read_csv("/Users/lamhoangtung/whale/data/train.csv")
count = train_df['Id'].value_counts()
train_df['count'] = train_df['Id'].apply(lambda x: count[x])


def train_aug(scale, shear, p=1.0):
    return Compose([
        IAAAffine(
            scale=(scale, scale),
            shear=(shear, shear),
            mode=ia.ALL,
            p=1,
            cval=255,
            fit_output= True
        )
    ], p=p)


need_to_up_df = train_df[train_df["count"] < 4]

scales = [1.0, 1.2]
shears = [-30, 30, -20, 20, -10, 10]

augs = [(1.0, -30), (1.0, 30), (1.0, -20), (1.0, 20), (1.2, -30), (1.2, 30),
        (1.2, -20), (1.2, 20), (1.2, 10), (1.2, -10), (1.0, -10), (1.0, 10)]

upsampling_images = []
upsampling_ids = []

p2bb = pd.read_csv(BB_DF).set_index("Image")

if os.path.isfile(P2SIZE):
    print("P2SIZE exists.")
    with open(P2SIZE, 'rb') as f:
        p2size = pickle.load(f)

for id in tqdm(need_to_up_df["Id"].unique()):
    tmp_df = need_to_up_df[need_to_up_df["Id"] == id]
    all_images = tmp_df["Image"].values
    n_upsampling = 4 - len(all_images)
    count = 0
    if n_upsampling > len(all_images):
        rnd_images = np.random.choice(all_images, n_upsampling, replace=True)
    else:
        rnd_images = np.random.choice(all_images, n_upsampling, replace=False)
    for i, sample_image in enumerate(rnd_images):
        splits = sample_image.split(".")
        image_name = splits[0]

        aug_param = augs[i]
        upsampling_aug = train_aug(scale=aug_param[0], shear=aug_param[1])

        sample_image = train_root + sample_image
        sample_image = cv2.imread(sample_image)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        row = p2bb.loc[image_name+'.jpg']
        size_x, size_y = p2size[image_name+'.jpg']
        x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
        add_amount_x = int(size_x * crop_margin / 2)
        add_amount_y = int(size_y * crop_margin / 2)
        x0 -= add_amount_x
        y0 += add_amount_y
        x1 += add_amount_x
        y1 -= add_amount_y
        if x0 < 0:  x0 = 0
        if y0 > size_y: y0 = size_y
        if x1 > size_x: x1 = size_x
        if y1 < 0: y1+=add_amount_y
        # import pdb; pdb.set_trace()
        # Determine the region of the original image we want to capture based on the bounding box.

        sample_image = sample_image[int(y0):int(y1), int(x0):int(x1)]

        try:
            transform_image = upsampling_aug(image=sample_image)["image"]
        except Exception as ex:
            print(ex)
            import pdb; pdb.set_trace()
        image_name = f"{image_name}_{i}" + "." + splits[1]

        upsampling_images.append(image_name)
        upsampling_ids.append(id)

        upsampling_img = os.path.join(upsampling_root, image_name)

        cv2.imwrite(upsampling_img, transform_image)

up_df = pd.DataFrame({
    "Image": upsampling_images,
    "Id": upsampling_ids
})

train_df = pd.read_csv("/Users/lamhoangtung/whale/data/train.csv")

train_df["is_upsampling"] = False
up_df["is_upsampling"] = True

train_df = pd.concat([train_df, up_df], axis=0)

train_df.to_csv("train_up_df.csv", index=False)
