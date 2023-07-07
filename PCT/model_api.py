from models import PCT

from types import SimpleNamespace
import os
import warnings
from ultralytics import YOLO
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import torch 
import sys

import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import (inference_top_down_pose_model, process_mmdet_results)
from mmpose.datasets import DatasetInfo
from mmdet.apis import inference_detector, init_detector
from models.builder import build_posenet
from tqdm import tqdm

import io
from PIL import Image
import datetime
import matplotlib.patches as patches
from enum import Enum
from pathlib import Path
import contextlib
import shutil
import requests
import logging

import mmtrack
import mmdet
from mmtrack.apis import inference_sot,inference_mot, init_model
import tempfile
import mmcv
import tempfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

import argparse
from collections import OrderedDict
from torch.utils.data import DataLoader
import tarfile


from RVRT.models.network_rvrt import RVRT as net
from RVRT.utils import utils_image as util
from RVRT.data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, SingleVideoRecurrentTestDataset
from RVRT.main_test_rvrt import prepare_model_dataset,test_video,test_clip
###rewritten version of main in RVRT/main_test_rvrt.py

class ColorStyle:
    """
    Small helper class copied from PCT library, helps define colors for the
    visualization of the bones in the predicted pose
    """
    def __init__(self, color, link_pairs, point_color):
        self.color = color
        self.link_pairs = link_pairs
        self.point_color = point_color

        for i in range(len(self.link_pairs)):
            self.link_pairs[i].append(tuple(np.array(self.color[i])/255.))

        self.ring_color = []
        for i in range(len(self.point_color)):
            self.ring_color.append(tuple(np.array(self.point_color[i])/255.))
class ChunhuaStyle(ColorStyle):
    def __init__(self):
        color2 = [(252,176,243),(252,176,243),(252,176,243),
        (0,176,240), (0,176,240), (0,176,240),
        (255,255,0), (255,255,0),(169, 209, 142),
        (169, 209, 142),(169, 209, 142),
        (240,2,127),(240,2,127),(240,2,127), (240,2,127), (240,2,127)]
        link_pairs2 = [
                [15, 13], [13, 11], [11, 5], 
                [12, 14], [14, 16], [12, 6], 
                [9, 7], [7,5], [5, 6], [6, 8], [8, 10],
                [3, 1],[1, 2],[1, 0],[0, 2],[2,4],
                ]
        point_color2 = [(240,2,127),(240,2,127),(240,2,127), 
                    (240,2,127), (240,2,127), 
                    (255,255,0),(169, 209, 142),
                    (255,255,0),(169, 209, 142),
                    (255,255,0),(169, 209, 142),
                    (252,176,243),(0,176,240),(252,176,243),
                    (0,176,240),(252,176,243),(0,176,240),
                    (255,255,0),(169, 209, 142),
                    (255,255,0),(169, 209, 142),
                    (255,255,0),(169, 209, 142)]
        super().__init__(color2, link_pairs2, point_color2)

    
    def map_joint_dict(self,joints):
        joints_dict = {}
        for i in range(joints.shape[0]):
            x = int(joints[i][0])
            y = int(joints[i][1])
            id = i
            joints_dict[id] = (x, y)
            
        return joints_dict
class VisType(Enum):
    """
    Simple enum for selecting what type of visualization to do
    Visualization type used when we do inference
    -SKELETON paints the visualization with the joints and bones on every frame
    -BBOX creates red boxes demarkating where the bounding box model detected a box of the specific category, i.e. human
    -NONE, dont visualize the result just get the values
    """
    SKELETON = 1
    BBOX = 2
    NONE = 3
class UpscaleType(Enum):
    """
    used for selecting what type of upscaling to use in superres_video
    """
    BICUBIC=1
    RVRT=2
class PoseExtraction:
    def check_env(self):
        """
        Checks that the versions of the imported libraries are the
        same as the one i (Axel) verified to work
        """
        assert mmcv.__version__ == "1.7.0"
        assert mmdet.__version__ == "2.26.0"
        assert mmtrack.__version__ == "0.14.0"
        #assert mmengine.__version == "0.7.4"
        assert torch.version.cuda == "11.1"
        assert torch.__version__ == "1.8.0+cu111"
        assert sys.version_info.major ==3
        assert sys.version_info.minor == 8
    #sets self.det_model,self.pose_model,self.dataset and self._dataset_info
    def init_models(self,detec_config,detec_checkpoint,pose_config,pose_checkpoint,device):
        self.det_model = init_detector(
        detec_config, detec_checkpoint, device=device.lower())
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())
        #TODO download the dataset
        self.dataset = self.pose_model.cfg.data['test']['type']
        self._dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self._dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self._dataset_info = DatasetInfo(self._dataset_info)
    #visualizes the first frame of the video with the bbox on top to check if its a good fit
    #init_bbox, initial bounding box [x1,y1,x2,y2]
    #input_video, absolute path to the input video
    def vis_bbox_first_frame(self,init_bbox,input_video):        
        # Load the video
        video = cv2.VideoCapture(input_video)
        # Get the first frame of the video
        ret, frame = video.read()
        # Convert to RGB for matplotlib
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Prepare figure
        fig,ax = plt.subplots(1)
        # Display the image
        ax.imshow(frame)
        rectangle = Rectangle((init_bbox[0],init_bbox[1]),(init_bbox[2]-init_bbox[0]),(init_bbox[3]-init_bbox[1]),linewidth=1,edgecolor='r',facecolor='none')
        # Draw the rectangle on the image
        ax.add_patch(rectangle)
        video.release()
        return plt
    #rewrite of main in main_test_rvrt
    def run_rvrt(self,
        task='001_RVRT_videosr_bi_REDS_30frames', sigma=0,
         folder_lq='temp/raw_frames',save_dir="temp/upscaled",
         folder_gt=None,
         tile=[100,128,128], tile_overlap=[2,20,20],
         num_workers=16, save_result=False):
        # define model
        device = torch.device('cuda')
        args = SimpleNamespace(task=task, sigma=sigma, folder_lq=folder_lq, folder_gt=folder_gt,
                                tile=tile, tile_overlap=tile_overlap, num_workers=num_workers,
                                save_result=save_result)
        print("the task is: "+str(args.task))
        model = prepare_model_dataset(args)
        model.eval()
        model = model.to(device)

        if 'vimeo' in args.folder_lq.lower():
            test_set = VideoTestVimeo90KDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                            'meta_info_file': "data/meta_info/meta_info_Vimeo90K_test_GT.txt",
                                                'mirror_sequence': True, 'num_frame': 7, 'cache_data': False})
        elif args.folder_gt is not None:
            test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                                'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
        else:
            test_set = SingleVideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                                'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})

        test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=1, shuffle=False)

        #save_dir = f'results/{args.task}'
        if args.save_result:
            os.makedirs(save_dir, exist_ok=True)
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []
        test_results['psnr_y'] = []
        test_results['ssim_y'] = []

        assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

        for idx, batch in enumerate(test_loader):
            lq = batch['L'].to(device)
            folder = batch['folder']
            gt = batch['H'] if 'H' in batch else None

            # inference
            with torch.no_grad():
                output = test_video(lq, model, args)

            if 'vimeo' in args.folder_lq.lower():
                output = (output[:, 3:4, :, :, :] + output[:, 10:11, :, :, :]) / 2
                batch['lq_path'] = batch['gt_path']

            test_results_folder = OrderedDict()
            test_results_folder['psnr'] = []
            test_results_folder['ssim'] = []
            test_results_folder['psnr_y'] = []
            test_results_folder['ssim_y'] = []

            for i in range(output.shape[1]):
                # save image
                img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img.ndim == 3:
                    img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                if args.save_result:
                    seq_ = os.path.basename(batch['lq_path'][i][0]).split('.')[0]
                    os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                    cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}.png', img)

                # evaluate psnr/ssim
                if gt is not None:
                    img_gt = gt[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if img_gt.ndim == 3:
                        img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                    img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                    img_gt = np.squeeze(img_gt)

                    test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                    if img_gt.ndim == 3:  # RGB image
                        img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                        img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                        test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                        test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                    else:
                        test_results_folder['psnr_y'] = test_results_folder['psnr']
                        test_results_folder['ssim_y'] = test_results_folder['ssim']

            if gt is not None:
                psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
                test_results['psnr'].append(psnr)
                test_results['ssim'].append(ssim)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                print('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                        format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
            else:
                print('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

        # summarize psnr/ssim
        if gt is not None:
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

    ###1
    #calculates the bounding box using the SOT model from mmtracking
    def calculate_sot_bbox(self,frames,init_bbox,start_frame_number,visualize=False):
        sot_config = Path("mmtracking_configs") / "sot" / "siamese_rpn" / "siamese_rpn_r50_20e_lasot.py"
        sot_config=str(sot_config.absolute())
        sot_checkpoint = Path("mmtracking_checkpoints") / "siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth"
        sot_checkpoint=str(sot_checkpoint.absolute())
        # build the model from a config file and a checkpoint file
        sot_model = init_model(sot_config, sot_checkpoint, device='cuda:0')
        #xstart ystart xend yend?

        prog_bar = mmcv.ProgressBar(len(frames))
        result_lst = []
        for i, frame in enumerate(frames):
            result = inference_sot(sot_model, frame, init_bbox, frame_id=start_frame_number+i)
            result_lst.append(result)
            prog_bar.update()
            

        #rename the dict keys to match the other model, "bbox" is what PCT expects
        for elem in result_lst:
            elem["bbox"] = elem.pop("track_bboxes")
        return result_lst

    ###2
    def crop_video_bbox(self,frames,bboxes,visualize=False):
        #since the video needs to be constant in resolution this
        #calculates the largest width i.e. the width we want to set the new video to
        def calc_max_box_width(bboxes):
            max_dx = 0
            for bbox in bboxes:
                x1, y1, x2, y2,probs = bbox
                if((x2-x1) > max_dx):
                    max_dx= x2-x1
            return max_dx
        #same as calc_max_box_width but for height
        def calc_max_box_height(bboxes):
            max_dy = 0
            for bbox in bboxes:
                x1, y1, x2, y2,probs = bbox
                if((y2-y1) > max_dy):
                    max_dy= y2-y1
            return max_dy

        bboxes = [box["bbox"] for box in bboxes]
        width = int(calc_max_box_width(bboxes))
        height = int(calc_max_box_height(bboxes))
        print("width,height:"+str((width,height)))
        bboxes = np.round(bboxes).astype(np.int32)
        prog_bar = mmcv.ProgressBar(len(frames))
        cropped_frames=[]
        for i,frame in enumerate(frames): 
            x1, y1, x2, y2, prob = bboxes[i]
            # Extract the region inside the bounding box
            region = frame[y1:y2, x1:x2]
            rescaled = cv2.resize(region, (width, height), interpolation=cv2.INTER_CUBIC)
            cropped_frames.append(rescaled)
            prog_bar.update()
        return cropped_frames
    ###3
    #takes a cropped video and returns a scaled up version of the same video
    #NOTE: the license for rvrt is non-commercial
    def superres_video(self,frames,upscale_type=UpscaleType.BICUBIC):
        if upscale_type == UpscaleType.BICUBIC:
            resized_images = []
            #temporary code that resizes it to the correct resolution 1920x1080 but w/o deep learning
            for img in frames:
                # cv2.resize expects the new size in (width, height) format
                # img should be in numpy array format. If img is a PIL image, use np.array(img) to convert.
                new_img = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_CUBIC)
                resized_images.append(new_img)
        elif upscale_type == UpscaleType.RVRT:
            #create the directory
            raise NotImplementedError
        return resized_images
    ###4
    #takes a scaled up video and returns bboxes/segmentation of all persons
    def yolo_segmentation(self,frames): 
        # Load a model

        model = YOLO('yolov8m-seg.pt')  # load an official segmentation model
        #model = YOLO('path/to/best.pt')  # load a custom model

        # Track with the model
        results = model.track(source=frames,
                            show=False,
                            persist=True,
                            tracker="bytetrack.yaml",
                            classes=0,
                            iou=0.3,
                            conf=0.25,
                            retina_masks=True,
                            augment=True,
                            max_det=2)
        #reformat them on the format PCT wants
        all_frames_boxes = []
        for res in results:
            boxes = np.array(res.boxes.xyxy.data.cpu()).round().astype(np.int32)  # Boxes object for bbox outputs
            single_frame_boxes = []
            for box in boxes:
                single_frame_boxes.append({"bbox":box})
            all_frames_boxes.append(single_frame_boxes)
        return all_frames_boxes
    ###5
    #modified version of PCT/vis_tools/demo_img_with_mmdet.py
    #calls the functions multiple time with a single frame
    def pct_pose_estimation(self,
                        video_frames,
                        bboxes_lst,
                        bbox_threshold=None,
                        vis_type=VisType.NONE,
                        thickness=2,
                        ):
            #helper function to make the bboxes all 2 subarrays with 4 elements each
            def pad_bbox_list(bboxes):
                bboxes=[x["bbox"] for x in bboxes]
                padded_lst = []
                for arr in bboxes:
                    if arr.size == 0:
                        # if array is empty, create a new array of shape (2,4) filled with zeros
                        arr = np.zeros((2, 4), dtype=int)
                    else:
                        # if array is not empty, pad it with zeros
                        padding = ((0, 2 - arr.shape[0]), (0, 0))
                        arr = np.pad(arr, pad_width=padding, mode='constant', constant_values=0)
                    padded_lst.append(arr)

                padded_lst = [{"bbox":x} for x in padded_lst]
                return padded_lst
            human_poses = []
            output_frames = []
            framecount = 0
            # optional
            return_heatmap = False
            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None
            
            
            #for each frame get the poses for all the people that were detected in that frame
            #done automatically in the inference_top_down
            prog_bar = mmcv.ProgressBar(len(video_frames))
            for i,frame in enumerate(video_frames):
                #only run the pose detection if there is atleast one person detected
                bboxes=bboxes_lst[i]
                if(bboxes == []):
                    human_poses.append([])#keeps the frame number and poses aligned
                    continue
                #can be given a list of images from memory
                pose_results, returned_outputs = inference_top_down_pose_model(
                    self.pose_model,
                    frame,
                    bboxes,
                    bbox_thr=bbox_threshold,
                    format='xyxy',
                    dataset=self.dataset,
                    dataset_info=self._dataset_info,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names)
                
                human_poses.append(pose_results)
                prog_bar.update() 
                # show the results depending on vis
                """
                if vis_type==VisType.SKELETON:
                    out_frame = vis_pose_result_np(
                        frame,
                        pose_results,
                        thickness=thickness)
                    output_frames.append(out_frame)
                elif vis_type==VisType.BBOX:
                    out_frame = vis_bbox_result_np(
                    frame,
                    person_results,
                    thickness=thickness)
                    output_frames.append(out_frame)

                    framecount += 1
                if vis_type != VisType.NONE:
                    self.save_output_video(output_dir,output_frames,framerate)
                print("Total frames processed: "+str(framecount)
                """
            return human_poses
    #called if visualize=True of a certain step 1-5
    def write_current_video():
        sot_model.show_result(
                    img,
                    result,
                    wait_time=int(1000. / imgs.fps),
                    out_file=f'{out_path}/{i:06d}.jpg')
        output = Path("output") / 'sot.mp4'
        print(f'\n making the output video at {output} with a FPS of {imgs.fps}')
        mmcv.frames2video(out_path, str(output.absolute()), fps=imgs.fps, fourcc='mp4v')
        out_dir.cleanup() #removes the temporary .jpg files
    #writes each frame of the video to the temp folder where we store the images while we process them
    def setup_frames_data_dir(self,video_path):
        #video_path = Path(video_folder) / video_name
        print(video_path)
        #we assume the file sits in ./uploaded/video_name
        # Open the video
        cap = cv2.VideoCapture(str(video_path))

        # Check if camera opened successfully
        if (cap.isOpened() == False): 
            print("Unable to read camera feed")

        frame_count = 0
        print("writing each frame of video to " + str(self.image_folder))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        with tqdm(total=total_frames, unit='frame', ncols=70, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            while(True):
                ret, frame = cap.read()

                if ret == True: 
                    # Write the frame into the file 'self.image_folder/frame_count.jpg'
                    frame_name =  f'frame_{str(frame_count).zfill(6)}.jpg'
                    frame_path = os.path.join(self.image_folder,frame_name)
                    cv2.imwrite(frame_path, frame)
                    frame_count += 1
                    pbar.update(1)
                # Break the loop if video is ended
                else:
                    break 

        # When everything done, release the video capture and video write objects
        cap.release()
    #just deletes the temp folder where we stored the images during processing
    def delete_frames_data_dir(self):
        shutil.rmtree(self.image_folder)
    
    def write_cropped_images(self,cropped_batch,start_frame):
        for i,frame in enumerate(cropped_batch):
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
                        pil_image = Image.fromarray(frame_rgb)
                        pil_image.save(str(Path("temp") / Path("upscaled") / Path(f"img_{start_frame+i:06}.jpg")))
    #returns up to batch_size number of images from the temp directory of image frames
    #all n-1 batches is 100 long and last batch is the remaining images
    def load_image_batch(self,batch_size,filetype="jpg",path=None):
        if path == None:
            all_image_files = glob.glob(os.path.join(self.image_folder, f'*.{filetype}'))
        else:
            all_image_files = glob.glob(os.path.join(path, f'*.{filetype}'))
        all_image_files.sort()
        for i in range(0, len(all_image_files), batch_size):
            batch_files = all_image_files[i:i + batch_size]
            batch_images = []
            
            for file in batch_files:
                img = cv2.imread(file)
                batch_images.append(img)
            
            yield batch_images
    #creates the black lines
    def new_vis_pose_result_np(self,data_numpy, pose_results, thickness):
        # Copy the image so we don't draw on top of the original
        img = data_numpy.copy()
        
        # ChunhuaStyle() equivalent in cv2
        chunhua_style = ChunhuaStyle()
        
        for i, dt in enumerate(pose_results):
            dt_joints = np.array(dt['keypoints']).reshape(17,-1)
            joints_dict = chunhua_style.map_joint_dict(dt_joints)
            
            # draw sticks/lines
            for k, link_pair in enumerate(chunhua_style.link_pairs):
                if k in range(11,16):
                    lw = thickness
                else:
                    lw = thickness * 2
                    
                # map color to BGR for cv2
                color_bgr = (link_pair[2][2], link_pair[2][1], link_pair[2][0])
                
                pt1 = tuple(int(x) for x in joints_dict[link_pair[0]])
                pt2 = tuple(int(x) for x in joints_dict[link_pair[1]])
                cv2.line(img, pt1, pt2, color_bgr, thickness=lw)

            # draw circles
            for k in range(dt_joints.shape[0]):
                if k in range(5):
                    radius = thickness
                else:
                    radius = thickness * 2
                    
                center = tuple(int(x) for x in dt_joints[k,:2])
                
                # map color to BGR for cv2
                ring_color_bgr = (chunhua_style.ring_color[k][2], chunhua_style.ring_color[k][1], chunhua_style.ring_color[k][0])
                
                cv2.circle(img, center, radius, (0,0,0), thickness=-1) # black border
                cv2.circle(img, center, radius-1, ring_color_bgr, thickness=-1) # fill
                
        return img
    def create_video(self,human_poses):
        img_files = glob.glob(os.path.join(Path("./temp"), '*.jpg')) #only take jpg files
        img_files.sort()  # make sure that the images are in order

        # Read the first file to get the size and color information
        img = cv2.imread(img_files[0])
        height, width, layers = img.shape
        size = (width,height)

        # Create a VideoWriter object
        out = cv2.VideoWriter('output_3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
        prog_bar = mmcv.ProgressBar(len(img_files))
        prev_pose = []#no pose until first frame that has a pose
        for i,img_file in enumerate(img_files):
            img = cv2.imread(img_file)
            if(human_poses[i] != []):
                processed_img = cv2.resize(self.new_vis_pose_result_np(img,human_poses[i],1),(1920,1080))
                out.write(processed_img)
                prev_pose=human_poses[i]
            else:
                #use prev frames skeleton
                processed_img = cv2.resize(self.new_vis_pose_result_np(img,prev_pose,1),(1920,1080))
                out.write(img)
            prog_bar.update()

        out.release()
    #does the entire transformation we want, from video to poses
    #crop_video_bbox crops to one size per batch but thats okay since we upscale them to 1080x1920,note: causes visualizations of this step to be different size
    #pct_pose_estimation uses the upscaled frames so we dont need to translate the postion of the bbox to the original video
    def end_to_end(self,video,init_bbox,image_batch_size=100,debug=False):
        original_stdout = sys.stdout
        if not debug:
            sys.stdout = open(os.devnull, 'w')
        self.setup_frames_data_dir(video)
        #num_batches = len(glob.glob(str(self.image_folder / Path("*.jpg")))) // image_batch_size
        #prog_bar = mmcv.ProgressBar(num_batches +1)
        human_poses=[]
        upscaled_frames = []
        start_frame = 0
        human_bboxes_all=[]
        for batch in self.load_image_batch(image_batch_size):
            #1-5
            sot_bbox_batch=self.calculate_sot_bbox(batch,init_bbox,0)
            cropped_batch=self.crop_video_bbox(batch,sot_bbox_batch)
            upscaled_batch=self.superres_video(cropped_batch)
            human_bboxes_batch=self.yolo_segmentation(upscaled_batch)
            human_bboxes_all.append(human_bboxes_batch)
            human_poses_batch=self.pct_pose_estimation(upscaled_batch,human_bboxes_batch)
            
            #append and set init_bbox for next iteration, and increment starting frame for next batch
            for item in human_poses_batch:
                human_poses.append(item)
            if not os.path.exists(str((Path("temp") / Path("upscaled")))):
                    os.makedirs(str((Path("temp") / Path("upscaled"))))
            self.write_cropped_images(cropped_batch,start_frame)
            init_bbox=sot_bbox_batch[-1]["bbox"][:4] #[]"bbox"] because its a map, [:4] to ditch the probability
            start_frame+=len(batch)
        #p.delete_frames_data_dir()
        sys.stdout = original_stdout
        return human_poses
    def __init__(self,
                 parent_path,
                 debug=False,
                 detec_config="vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py",
                 detec_checkpoint="https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth",
                 model_size="base",
                 device = "cuda:0"
                 ) -> None:
        #make sure all packages are the right version
        self.check_env()
        
        if not os.path.exists("./videos"):
            os.makedirs("./videos")
        self.parent_path=parent_path
        self.image_folder = parent_path / Path("temp") / Path("raw_frames") / Path("000")

        pose_config=Path("configs") / f"pct_{model_size}_classifier.py"
        pose_config=str(pose_config.absolute())
        pose_checkpoint=Path("weights") / "pct" / f"swin_{model_size}.pth"
        pose_checkpoint=str(pose_checkpoint.absolute())
        self.cat_id = 1 #Category id for bounding box detection model, 1 corresponds to person?
        
        #download the SOT and PCT weights if needed
        self.check_sot_weights_present()
        self.check_pct_weights_present()
        #if debug is false silence all the outputs of the models when they init
        original_stdout = sys.stdout 
        if debug==False:
            logging.getLogger().setLevel(logging.CRITICAL)
            sys.stdout = open(os.devnull, 'w')
        #initialize the models
        self.init_models(detec_config,detec_checkpoint,pose_config,pose_checkpoint,device)
        #restore prints
        sys.stdout = original_stdout
    def check_pct_weights_present(self):
        """
        same as check_sot_weights_present but with a loop and downloads a tar from google drive,
        couldnt download from the paper link in python
        """
        def weights_folder_correct():
            files = ["./weights/pct/swin_base.pth","./weights/heatmap/swin_base.pth","./weights/simmim/swin_base.pth","./weights/tokenizer/swin_base.pth"]
            for file in files:
                if not os.path.isfile(file):
                    return False
            return True
        if(not weights_folder_correct()):
            #create folder
            if not os.path.exists("./weights"):
                os.makedirs("./weights")
            #download tar file
            url=r"https://drive.google.com/u/0/uc?id=12pxN3W2UTl7jRSlAu4wHDJkZaPG1oCDB&export=download&confirm=t&uuid=ada72c11-0255-4279-8283-dee8d324b339&at=AKKF8vzTGxHbGT2OYJdDym0HqElj:1688581673447"
            response = requests.get(url, allow_redirects=True)
            open('weights.tar', 'wb').write(response.content)

            #untar it
            tar = tarfile.open("weights.tar")
            tar.extractall()
            tar.close()
            #clean up tar file
            os.remove("weights.tar")
    def check_sot_weights_present(self):
        """
        checks if the checkpoint files for the pretrained models are in the correct place
        if not downloads them
        """
        if not os.path.exists("./mmtracking_checkpoints"):
                    os.makedirs("./mmtracking_checkpoints")
        if not os.path.isfile(Path("mmtracking_checkpoints") / Path("selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth")):
            print("downloading selsa_faster_rcnn_r50")
            download_and_write_file("https://download.openmmlab.com/mmtracking/vid/selsa/selsa_faster_rcnn_r50_dc5_1x_imagenetvid/selsa_faster_rcnn_r50_dc5_1x_imagenetvid_20201227_204835-2f5a4952.pth","mmtracking_checkpoints")
        if not os.path.isfile(Path("mmtracking_checkpoints") / Path("siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth")):
            print("downloading siamese rpn")
            download_and_write_file("https://download.openmmlab.com/mmtracking/sot/siamese_rpn/siamese_rpn_r50_1x_lasot/siamese_rpn_r50_1x_lasot_20211203_151612-da4b3c66.pth","mmtracking_checkpoints")
        if not os.path.isfile(Path("mmtracking_checkpoints") / Path("masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth")):
            print("downloading masktrack rcnn")
            download_and_write_file("https://download.openmmlab.com/mmtracking/vis/masktrack_rcnn/masktrack_rcnn_r50_fpn_12e_youtubevis2019/masktrack_rcnn_r50_fpn_12e_youtubevis2019_20211022_194830-6ca6b91e.pth","mmtracking_checkpoints")
    #CHANGE OR MOVE
    def save_output_video(self,dir,frames,framerate):
        #make sure directory exists
        os.makedirs(dir, exist_ok=True)
        filename="out_" + datetime.datetime.now().strftime("%H_%M") + ".mp4"
        output_path = os.path.join(dir,filename)
        #convert the output frames to a video and save it
        images_to_video(frames, output_path, framerate)
        print("Wrote visualization video to " + str(output_path))
    def zoom(self,image, zoom_factor):
        height, width, _ = image.shape

        # Getting the center of the image
        center_height = height // 2
        center_width = width // 2

        # Calculating the new dimensions of the image
        new_height = int(height // zoom_factor)
        new_width = int(width // zoom_factor)

        # Getting the region of interest
        roi = image[center_height - new_height // 2: center_height + new_height // 2,
                    center_width - new_width // 2: center_width + new_width // 2, :]
        
        # Resizing the ROI to the original size to achieve zoom effect using bicubic interpolation
        zoomed_image = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)

        return zoomed_image
    def zoom_video(self,input_video, output_video_path, zoom_factor):
        imgs = mmcv.VideoReader(input_video)
        prog_bar = mmcv.ProgressBar(len(imgs))
        frames_out = []
        for i,frame in enumerate(imgs): 
            out = self.zoom(frame,zoom_factor)
            out = out.astype(np.uint8)
            out=cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            frames_out.append(out) #inefficient but easy solution
            prog_bar.update()
        self.save_output_video("output",frames_out,imgs.fps)

##Below functions are outside of the class since they are not meant to be called directly  
def download_and_write_file(url,destination_folder):
    response = requests.get(url, stream=True)

    file_size = int(response.headers.get('Content-Length', 0))
    file_name = url.split("/")[-1]
    destination_path = os.path.join(destination_folder, file_name)

    with open(destination_path, 'wb') as f:
        for data in response.iter_content(1024):
            f.write(data)
def init_pose_model(config, checkpoint=None, device='cuda:0'):
    """Initialize a pose model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    model = build_posenet(config.model)
    if checkpoint is not None:
        # load model checkpoint
        load_checkpoint(model, checkpoint, map_location='cpu')
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model
#same as vis_pose_result but takes an np array and returns an np array
def vis_pose_result_np(data_numpy, pose_results, thickness):
    
    #data_numpy = cv2.imread(image_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    h = data_numpy.shape[0]
    w = data_numpy.shape[1]
        
    # Plot
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)
    #helps define colors for the visualization of bones
    chunhua_style=ChunhuaStyle()
    
    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17,-1)
        joints_dict = chunhua_style.map_joint_dict(dt_joints)
        
        # stick 
        for k, link_pair in enumerate(chunhua_style.link_pairs):
            if k in range(11,16):
                lw = thickness
            else:
                lw = thickness * 2

            line = mlines.Line2D(
                    np.array([joints_dict[link_pair[0]][0],
                                joints_dict[link_pair[1]][0]]),
                    np.array([joints_dict[link_pair[0]][1],
                                joints_dict[link_pair[1]][1]]),
                    ls='-', lw=lw, alpha=1, color=link_pair[2],)
            line.set_zorder(0)
            ax.add_line(line)

        # black ring
        for k in range(dt_joints.shape[0]):
            if k in range(5):
                radius = thickness
            else:
                radius = thickness * 2

            circle = mpatches.Circle(tuple(dt_joints[k,:2]), 
                                        radius=radius, 
                                        ec='black', 
                                        fc=chunhua_style.ring_color[k], 
                                        alpha=1, 
                                        linewidth=1)
            circle.set_zorder(1)
            ax.add_patch(circle)
        
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()

    # Load buffer as an image
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    
    return img_arr
def vis_bbox_result_np(data_numpy, bbox_results, thickness):
    
    h = data_numpy.shape[0]
    w = data_numpy.shape[1]
        
    # Plot
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = plt.subplot(1,1,1)
    bk = plt.imshow(data_numpy[:,:,::-1])
    bk.set_zorder(-1)
    
    for i, dt in enumerate(bbox_results[:]):
        bbox = dt['bbox']
        x1, y1, x2, y2 = bbox[:4]

        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=thickness, edgecolor='r', facecolor='none')
        rect.set_zorder(1)
        ax.add_patch(rect)
        
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.axis('off')
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)        
    plt.margins(0,0)

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()

    # Load buffer as an image
    buf.seek(0)
    img_arr = np.array(Image.open(buf))
    
    return img_arr
def images_to_video(image_arrays, output_file, fps):
    # Get the shape of the images
    h, w, _ = image_arrays[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a codec suitable for mp4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

    for image in image_arrays:
        # Convert the image from RGB to BGR format
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Write the frame to the video file
        out.write(image_bgr)

    # Release the VideoWriter
    out.release()


