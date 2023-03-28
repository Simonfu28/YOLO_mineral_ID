import os
import cv2
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
                help = 'path to raw images')
args = ap.parse_args()

img_name = 0
out_dir = 'Training-split/'

if os.path.exists(out_dir) is False:
    os.mkdir(out_dir)

for files in tqdm(os.listdir(args.path)):
    src_img = cv2.imread(args.path + files)
    src_height, src_width = src_img.shape[0], src_img.shape[1]

    cutoff = [(src_height//2), (src_width//2)]

    cut_img = [
            src_img[:cutoff[0], :cutoff[1]], 
            src_img[:cutoff[0], cutoff[1]:],
            src_img[cutoff[0]:, :cutoff[1]],
            src_img[cutoff[0]:, cutoff[1]:]
            ]

    UL_img = src_img[:cutoff[0], :cutoff[1]]
    UR_img = src_img[:cutoff[0], cutoff[1]:]
    LL_img = src_img[cutoff[0]:, :cutoff[1]]
    LR_img = src_img[cutoff[0]:, cutoff[1]:]

    for i in range(0,4):
        cv2.imwrite(out_dir + str(img_name).zfill(5) + '.jpg', cut_img[i])
        img_name = img_name + 1

