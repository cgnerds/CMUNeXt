import os
import cv2
import time
import torch
import datetime
import argparse
import numpy as np
import albumentations as albu
from skimage import morphology
from albumentations.core.composition import Compose
from network.CMUNeXt import cmunext, cmunext_s, cmunext_l

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="CMUNeXt-L", choices=["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L"], help='model')
parser.add_argument('--folder', type=str, default="data/test", help='dir')
parser.add_argument('--device', type=str, default="cuda", help='dir')
parser.add_argument('--output', default='outputs', help='output dir')

args = parser.parse_args()

def get_model(args):
    if args.model == "CMUNeXt":
        model = cmunext()
    elif args.model == "CMUNeXt-S":
        model = cmunext_s()
    elif args.model == "CMUNeXt-L":
        model = cmunext_l()
    else:
        model = None
        print("model err")
        exit(0)
    return model.to(args.device)

def infer_folder(args):
    # get model
    model = get_model(args)
    model_path = os.path.join('checkpoint', 'CMUNeXt_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    
    # create folder to save combined images
    combined_folder = os.path.join(args.output,  args.folder, 'combined')
    if not os.path.exists(combined_folder):
        os.makedirs(combined_folder)
    
    
    # transform
    img_size = 256
    val_transform = Compose([
        albu.Resize(img_size, img_size),
        albu.Normalize(),
    ])
    
    frames = os.listdir(args.folder)
    for frame in frames:
        print(frame)
        img = cv2.imread(os.path.join(args.folder, frame))
        
        # image shape
        img_h, img_w, _ = img.shape

        image = val_transform(image=img)['image']
        image = image.astype('float32') / 255
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        
        with torch.no_grad():
            image = image.to(args.device)
            output = model(image)
            # [1, 1, 256, 256]
            output = torch.sigmoid(output).cpu().numpy()
            output[output > 0.5] = 1
            output[output <= 0.5] = 0
            
            for i in range(len(output)):
                viz_mask_bgr = np.zeros((img_size, img_size, 3))
                # num_classes 
                for c in range(output.shape[1]):
                    if (np.count_nonzero(output[i, c]) > 100):
                        viz_mask_bgr[np.where(output[i, c]>0)] = [0,255,255]
                    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(output[i, c].astype('uint8'), connectivity=4, ltype=None)
                    # regoins = morphology.remove_small_objects(ar=labels, min_size=threshold, connectivity=1)
                    # viz_mask_bgr[regoins>0] = [0,0,200]
                    
                    
                opacity = 0.8 # 透明度越大，可视化效果越接近原图
                ret_img = img.copy()
                viz_mask_bgr = viz_mask_bgr.astype('uint8')
                viz_mask_bgr = cv2.resize(viz_mask_bgr, dsize=(img_w, img_h))
                ret_img = cv2.addWeighted(ret_img, opacity, viz_mask_bgr, 1-opacity, 0)
                
                # save combined image with mask
                ret_file = os.path.join(combined_folder, frame)
                cv2.imwrite(ret_file, ret_img)
                                
                # display image with mask
                cv2.imshow('ret_img', ret_img)

        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    infer_folder(args)