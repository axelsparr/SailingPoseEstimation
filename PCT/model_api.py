from models import PCT

from argparse import ArgumentParser
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

import mmtrack
import mmdet
from mmtrack.apis import inference_sot,inference_mot, init_model
import tempfile
import mmcv
import tempfile
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

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
    def superres_video(self,frames):
        print("NOTE: NO SUPERRESOLUTION IS BEING APPLIED USES BICUBIC INTERPOLATION")
        resized_images = []
        #temporary code that resizes it to the correct resolution 1920x1080 but w/o deep learning
        for img in frames:
            # cv2.resize expects the new size in (width, height) format
            # img should be in numpy array format. If img is a PIL image, use np.array(img) to convert.
            new_img = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_CUBIC)
            resized_images.append(new_img)
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
    
    #returns up to batch_size number of images from the temp directory of image frames
    #all n-1 batches is 100 long and last batch is the remaining images
    def load_image_batch(self,batch_size):
        all_image_files = glob.glob(os.path.join(self.image_folder, '*.jpg'))
        
        for i in range(0, len(all_image_files), batch_size):
            batch_files = all_image_files[i:i + batch_size]
            batch_images = []
            
            for file in batch_files:
                img = cv2.imread(file)
                batch_images.append(img)
            
            yield batch_images
    #does the entire transformation we want, from video to poses
    #crop_video_bbox crops to one size per batch but thats okay since we upscale them to 1080x1920,note: causes visualizations of this step to be different size
    #pct_pose_estimation uses the upscaled frames so we dont need to translate the postion of the bbox to the original video
    def end_to_end(self,video,init_bbox):
        self.setup_frames_data_dir(video)
        human_poses=[]
        for batch in self.load_image_batch(100):
            #1-5
            sot_bbox_batch=self.calculate_sot_bbox(batch,init_bbox)
            cropped_batch=self.crop_video_bbox(batch,sot_bbox_batch)
            upscaled_batch=self.superres_video(cropped_batch)
            human_bboxes_batch=self.yolo_segmentation(upscaled_batch)
            human_poses_batch=self.pct_pose_estimation(upscaled_batch,human_bboxes_batch)
            
            #append and set init_bbox for next iteration
            human_poses.append(human_poses_batch)
            init_bbox=sot_bbox_batch[-1]["bbox"] #[]"bbox"] because its a map
        self.delete_frames_data_dir()
        return human_poses
    def __init__(self,
                 parent_path,
                 debug=True,
                 detec_config="vis_tools/cascade_rcnn_x101_64x4d_fpn_coco.py",
                 detec_checkpoint="https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth",
                 model_size="base",
                 device = "cuda:0"
                 ) -> None:
        #make sure all packages are the right version
        self.check_env()

        self.parent_path=parent_path
        self.image_folder = parent_path / Path("temp")

        pose_config=Path("configs") / f"pct_{model_size}_classifier.py"
        pose_config=str(pose_config.absolute())
        pose_checkpoint=Path("weights") / "pct" / f"swin_{model_size}.pth"
        pose_checkpoint=str(pose_checkpoint.absolute())
        self.cat_id = 1 #Category id for bounding box detection model, 1 corresponds to person?
        
        #initialize the models
        self.init_models(detec_config,detec_checkpoint,pose_config,pose_checkpoint,device)

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

#get just the inside, extracts what is inside the bbox in each frame, then scales it up with bicubic to the specified width height
#the idea is that width,height come from the largest x_delta and y_delta of all the frames
#useful since instead of getting a normal size video with smeared pixels it gets a small video with normal "size pixels"
#although cant tell the difference when you watch it in media player as they get treated the same
#offset is a "padding" in all directions
def crop_image_bbox(image,bbox,offset=0):
    bbox = bbox.astype(np.int32)
    #bbox = [int(coord) for coord in bbox]
    x1, y1, x2, y2, prob = bbox+offset

    # Extract the region inside the bounding box
    region = image[y1:y2, x1:x2]
    # Get the original image dimensions
    height, width = image.shape[:2]

    # Rescale the region to fit the original image size
    rescaled = cv2.resize(region, (width, height), interpolation=cv2.INTER_CUBIC)
    return rescaled
#applies gaussian blur to all pixels outside the specified box and returns the image,opencv can handle np arrays directly
def blur_all_pixels_outside_bbox(image,bbox,blur_amount=30):
    bbox = bbox.astype(np.int32)
    #bbox = [int(coord) for coord in bbox]
    x1, y1, x2, y2, prob = bbox
    blurred_image = image.copy()
    # Apply a Gaussian blur to the copy of the image
    blurred_image = cv2.GaussianBlur(blurred_image, (99,99), blur_amount)

    # Paste the non-blurred region back onto the blurred image
    blurred_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return blurred_image

#takes in a video frame as an np array and sets all pixels outside the bbox in x1,y1,x2,y2 format to 0,0,0
def set_pixels_outside_bbox_black(image, rectangle):
        # Ensure value is an array
    value = np.array([0,0,0])
    image= image.astype(np.int32)
    # Convert rectangle coordinates to integers
    rectangle = [int(coord) for coord in rectangle]
        
    # Create mask with ones
    mask = np.ones(image.shape, dtype=bool)
    
    # Set rectangle area to False
    mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = False

    # Assign the value to the pixels outside the rectangle
    np.putmask(image, mask, value)

    return image
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
    print(video_path + " has " + str(default_fps) + " fps")
    if default_fps == None or default_fps == 0:
        default_fps = 1
    #if we specify a too high framerate just use the highest we can
    if sample_rate > default_fps:
        sample_rate = default_fps
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

