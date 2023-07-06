import numpy as np
import cv2
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
