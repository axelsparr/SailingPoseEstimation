from models import PCT

from argparse import ArgumentParser
import os
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
            
#from PCT library, helps define colors for the visualization of bones        
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
chunhua_style = ColorStyle(color2, link_pairs2, point_color2)
def map_joint_dict(joints):
    joints_dict = {}
    for i in range(joints.shape[0]):
        x = int(joints[i][0])
        y = int(joints[i][1])
        id = i
        joints_dict[id] = (x, y)
        
    return joints_dict

#Visualization type used when we do inference
#SKELETON paints the visualization with the joints and bones on every frame
#BBOX creates red boxes demarkating where the bounding box model detected a box of the specific category, i.e. human
#NONE, dont visualize the result just get the values
class VisType(Enum):
    SKELETON = 1
    BBOX = 2
    NONE = 3
class PoseExtraction:
    def check_env(self):
        """
        Checks that the versions of the imported libraries are the
        same as the one i (Axel) verified to work
        """
        assert mmcv.__version__ == "1.7.0"
        assert torch.version.cuda == "11.1"
        assert torch.__version__ == "1.8.0+cu111"
        assert sys.version_info.major ==3
        assert sys.version_info.minor == 8
    def save_output_video(self,dir,frames,framerate):
        #make sure directory exists
        os.makedirs(dir, exist_ok=True)
        filename="out_" + datetime.datetime.now().strftime("%H_%M") + ".mp4"
        output_path = os.path.join(dir,filename)
        #convert the output frames to a video and save it
        images_to_video(frames, output_path, framerate)
        print("Wrote visualization video to " + str(output_path))
        
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
    
    #modified version of PCT/vis_tools/demo_img_with_mmdet.py
    #calls the functions multiple time with a single frame
    def video_inference(self,
                        video_name,
                        vis_type=VisType.NONE,
                        framerate=1,#how many times a second to run the pose estimation
                        video_folder="videos/",
                        output_dir="output/",
                        thickness=2,
                        bbox_threshold=0.3,
                        ):
            video_path = os.path.join(video_folder, video_name)
            
            video_frames = extract_frames(video_path,sample_rate=framerate)
            
            output_data = []
            output_frames = []
            framecount = 0
            for frame in tqdm(video_frames, desc="Processing frames", unit="frames"):
                # test a single image, the resulting box is (x1, y1, x2, y2)
                mmdet_results = inference_detector(self.det_model, frame)
                # keep the person class bounding boxes.
                person_results = process_mmdet_results(mmdet_results, self.cat_id)

                # test a single image, with a list of bboxes.

                # optional
                return_heatmap = False

                # e.g. use ('backbone', ) to return backbone feature
                output_layer_names = None

                #can be given a list of images from memory
                pose_results, returned_outputs = inference_top_down_pose_model(
                    self.pose_model,
                    frame,
                    person_results,
                    bbox_thr=bbox_threshold,
                    format='xyxy',
                    dataset=self.dataset,
                    dataset_info=self._dataset_info,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names)
                
                output_data.append(pose_results)
                
                # show the results depending on vis
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
            print("Total frames processed: "+str(framecount))
            #return output_data
    def __init__(self,
                 detec_config="vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py",
                 detec_checkpoint="https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth",
                 model_size="base",
                 device = "cuda:0"
                 ) -> None:
        #make sure all packages are the right version
        self.check_env()

        pose_config=Path("configs") / f"pct_{model_size}_classifier.py"
        pose_config=str(pose_config.absolute())
        pose_checkpoint=Path("weights") / "pct" / f"swin_{model_size}.pth"
        pose_checkpoint=str(pose_checkpoint.absolute())
        self.cat_id = 1 #Category id for bounding box detection model, 1 corresponds to person?
        
        #initialize the models
        self.init_models(detec_config,detec_checkpoint,pose_config,pose_checkpoint,device)

##Below functions are outside of the class since they are not meant to be called directly  
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

def get_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Read the first frame
    ret, frame = cap.read()

    if ret:
        # Save the result
        cv2.imwrite('first_frame.png', frame)
    else:
        print("Error getting frame")

    # When everything done, release the video capture object
    cap.release()

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
    
    for i, dt in enumerate(pose_results[:]):
        dt_joints = np.array(dt['keypoints']).reshape(17,-1)
        joints_dict = map_joint_dict(dt_joints)
        
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

def extract_frames(video_path, sample_rate):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Could not open video")
        return []

    # Get the video's default frame rate
    default_fps = video.get(cv2.CAP_PROP_FPS)
    if default_fps == None or default_fps == 0:
        default_fps = 1
    frame_count = 0
    frames = []

    while True:
        ret, frame = video.read()

        # If the frame was not successfully read then break the loop
        if not ret:
            break

        # If this frame number is divisible by the sample rate, store it
        if frame_count % int(default_fps / sample_rate) == 0:
            frames.append(np.array(frame))

        frame_count += 1

    video.release()

    return frames

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
