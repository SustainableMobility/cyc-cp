import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import cv2
import subprocess
import shutil
from PIL import Image


def ffmpeg(filename, outfile, fps):
    """
    Extract frames from a video with a given FPS.
    """
    command = ["ffmpeg", "-i", filename, "-s", "1600*900", "-r", str(fps), outfile]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    pipe.communicate()


def extract_frames_from_rgb_and_opt_video_clip(csv_data_path=None,
                                               new_video_clip_rgb_path=None,
                                               new_video_clip_opt_path=None,
                                               video_rgb_frame_root_path=None,
                                               video_opt_frame_root_path=None,
                                               video_fused_frame_root_path=None,
                                               extract_fps=25,
                                               process_start_id=None,
                                               process_end_id=None,
                                               key_frame_front_num=30,
                                               key_frame_rear_num=30,
                                               key_frame_redundant=5,
                                               opt_flow_fusion_length=4):
    """
    key_frame_front_num = 30    # The number of frame that will be kept before key frame (need to be larger than that for training data)
    key_frame_rear_num = 30     # The number of frame that will be kept after key frame (need to be larger than that for training data)
    key_frame_redundant = 5     # Keep a few redundant frames to avoid round to a larger frame id and cannot find the frame
    """
    #
    csv_data_df = pd.read_csv(csv_data_path)
    #
    new_video_clip_rgb_list = os.listdir(new_video_clip_rgb_path)
    new_video_clip_rgb_list.sort()
    for video_clip_rgb in tqdm(new_video_clip_rgb_list[process_start_id: process_end_id]):
        video_clip_name = video_clip_rgb.split('rgb_')[1]
        print('Processing {}'.format(video_clip_name))
        # # Only extract frames from video clips included in dataset to save space
        # if video_clip_name in csv_data_df['ClipName'].values:
        video_clip_opt = 'opt_{}'.format(video_clip_rgb.split('rgb_')[1])
        video_clip_rgb_path = os.path.join(new_video_clip_rgb_path, video_clip_rgb)
        video_clip_opt_path = os.path.join(new_video_clip_opt_path, video_clip_opt)

        # Check basic info
        rgb_cap = cv2.VideoCapture(video_clip_rgb_path)
        opt_cap = cv2.VideoCapture(video_clip_opt_path)

        if not rgb_cap.isOpened():
            print("could not open : {}".format(video_clip_rgb_path))

        if not opt_cap.isOpened():
            print("could not open : {}".format(video_clip_opt_path))

        rgb_length = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        rgb_width = int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        rgb_height = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rgb_fps = rgb_cap.get(cv2.CAP_PROP_FPS)
        print('RGB: length={}, width={}, height={}, fps={}'.format(rgb_length, rgb_width, rgb_height, rgb_fps))

        opt_length = int(opt_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        opt_width = int(opt_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        opt_height = int(opt_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        opt_fps = opt_cap.get(cv2.CAP_PROP_FPS)
        print('OPT: length={}, width={}, height={}, fps={}'.format(opt_length, opt_width, opt_height, opt_fps))


        # Specify frame folder name
        video_clip_frame_folder = '{}'.format(video_clip_rgb.split('.mp4')[0].split('rgb_')[1])
        video_clip_rgb_frame_folder_path = os.path.join(video_rgb_frame_root_path, video_clip_frame_folder)
        video_clip_opt_frame_folder_path = os.path.join(video_opt_frame_root_path, video_clip_frame_folder)
        video_clip_fused_frame_folder_path = os.path.join(video_fused_frame_root_path, video_clip_frame_folder)
        if not os.path.exists(video_clip_rgb_frame_folder_path):
            os.mkdir(video_clip_rgb_frame_folder_path)
        if not os.path.exists(video_clip_opt_frame_folder_path):
            os.mkdir(video_clip_opt_frame_folder_path)
        if not os.path.exists(video_clip_fused_frame_folder_path):
            os.makedirs(video_clip_fused_frame_folder_path)

        ###################################################
        # Extract frames
        ###################################################
        # Specify output image name
        rgb_outputfile = os.path.join(video_clip_rgb_frame_folder_path, "image_%06d.jpg")
        opt_outputfile = os.path.join(video_clip_opt_frame_folder_path, "image_%06d.jpg")
        # Start extraction all frames
        start_time = time.time()
        ffmpeg(video_clip_rgb_path, rgb_outputfile, extract_fps)
        ffmpeg(video_clip_opt_path, opt_outputfile, extract_fps)

        ###################################################
        # Delete redundant frames to save space
        ###################################################
        # Get video clip duration in seconds from clip name
        clip_duration = int(video_clip_name.split('_')[1].split('Duration')[1].split('s')[0])
        event_time = clip_duration / 2  # passing event is in the exactly middle of the clip
        # Get the index of frame that will be kept
        event_frame_id = int(event_time * extract_fps)  # Calculate the frame id where an event is detected and FPS = 25
        start_frame_id = event_frame_id - key_frame_front_num - key_frame_redundant
        end_frame_id = event_frame_id + key_frame_rear_num + key_frame_redundant
        print('\tstart_frame_id={},end_frame_id={}'.format(start_frame_id, end_frame_id))

        # Remove frames that are useless
        rgb_frame_list = os.listdir(video_clip_rgb_frame_folder_path)            # List all frames of the video
        opt_frame_list = os.listdir(video_clip_opt_frame_folder_path)
        print('rgb_frame_num={}, opt_frame_num={}'.format(len(rgb_frame_list), len(opt_frame_list)))
        for tmp_rgb_frame in rgb_frame_list:
            tmp_rgb_frame_id = int(tmp_rgb_frame.split('.jpg')[0].split('_')[-1])
            if tmp_rgb_frame_id < start_frame_id or tmp_rgb_frame_id > end_frame_id:
                os.remove(os.path.join(video_clip_rgb_frame_folder_path, tmp_rgb_frame))    # Remove frames
        for tmp_opt_frame in opt_frame_list:
            tmp_opt_frame_id = int(tmp_opt_frame.split('.jpg')[0].split('_')[-1])
            if tmp_opt_frame_id < start_frame_id-opt_flow_fusion_length or tmp_opt_frame_id > end_frame_id:
                os.remove(os.path.join(video_clip_opt_frame_folder_path, tmp_opt_frame))    # Remove frame

        ###################################################
        # Fuse RGB and OPT frames
        ###################################################
        rgb_frame_list = os.listdir(video_clip_rgb_frame_folder_path)  # List all frames of the video
        rgb_frame_list.sort()
        opt_frame_list = os.listdir(video_clip_opt_frame_folder_path)
        opt_frame_list.sort()
        # Get the maximum and minimum frame id
        rgb_frame_min_id = int(rgb_frame_list[0].split('_')[1].split('.')[0])
        rgb_frame_max_id = int(rgb_frame_list[-1].split('_')[1].split('.')[0])
        opt_frame_min_id = int(opt_frame_list[0].split('_')[1].split('.')[0])
        opt_frame_max_id = int(opt_frame_list[-1].split('_')[1].split('.')[0])
        print('rgb_frame_min_id={}, opt_frame_min_id={}, rgb_frame_max_id={}, opt_frame_max_id={}'.format(
            rgb_frame_min_id, opt_frame_min_id, rgb_frame_max_id, opt_frame_max_id))

        # Check if rgb and opt frames have matched id
        if not (rgb_frame_min_id - opt_frame_min_id) == opt_flow_fusion_length and rgb_frame_max_id == opt_frame_max_id:
            raise ValueError('RGB and OPT are not matched!')

        # Fuse
        for tmp_rgb_id in range(rgb_frame_min_id, rgb_frame_max_id + 1):
            # Read the current rgb frame
            tmp_rgb_img_path = os.path.join(video_clip_rgb_frame_folder_path, 'image_{:06d}.jpg'.format(tmp_rgb_id))
            tmp_rgb_img = Image.open(tmp_rgb_img_path)
            tmp_rgb_img_arr = np.array(tmp_rgb_img, dtype=float)
            # Read the current and the previous (opt_flow_fusion_length-1) optical flow frames
            tmp_opt_list = []
            tmp_opt_arr_list = []
            for tmp_opt_id in range(tmp_rgb_id, tmp_rgb_id - opt_flow_fusion_length, -1):
                tmp_opt_img_path = os.path.join(video_clip_opt_frame_folder_path, 'image_{:06d}.jpg'.format(tmp_opt_id))
                tmp_opt_img = Image.open(tmp_opt_img_path)
                tmp_opt_img_arr = np.array(tmp_opt_img, dtype=float)
                tmp_opt_list.append(tmp_opt_img)
                tmp_opt_arr_list.append(tmp_opt_img_arr)
            # Average RGB and OPT frames
            opt_avg_arr = np.array(np.mean(tmp_opt_arr_list, axis=0), dtype=np.uint8)  # Average optical flow
            fused_img_arr = np.array(np.mean([tmp_rgb_img_arr, opt_avg_arr], axis=0),
                                     dtype=np.uint8)  # Average rgb and opt
            fused_img = Image.fromarray(fused_img_arr)
            fused_img_path = os.path.join(video_clip_fused_frame_folder_path, 'image_{:06d}.jpg'.format(tmp_rgb_id))
            fused_img.save(fused_img_path)
            print('Fused RGB: {}, and OPT: {}'.format(tmp_rgb_id,
                                                      np.arange(tmp_rgb_id, tmp_rgb_id - opt_flow_fusion_length,
                                                                -1).tolist()))

        ###################################################
        # Archive the extracted frames into a .zip file and delete the unzipped files to further reduce space usage.
        #   (This is especially useful for computing system with file number and/or storage space limitation.)
        ###################################################
        rgb_zip_file_path = os.path.join(video_rgb_frame_root_path, '{}'.format(video_clip_frame_folder))
        opt_zip_file_path = os.path.join(video_opt_frame_root_path, '{}'.format(video_clip_frame_folder))
        fused_zip_file_path = os.path.join(video_fused_frame_root_path, '{}'.format(video_clip_frame_folder))
        shutil.make_archive(rgb_zip_file_path, 'zip', video_clip_rgb_frame_folder_path)
        shutil.make_archive(opt_zip_file_path, 'zip', video_clip_opt_frame_folder_path)
        shutil.make_archive(fused_zip_file_path, 'zip', video_clip_fused_frame_folder_path)
        shutil.rmtree(video_clip_rgb_frame_folder_path)
        shutil.rmtree(video_clip_opt_frame_folder_path)
        shutil.rmtree(video_clip_fused_frame_folder_path)

        print('\t\t Extracting frames costs: {:.2f}s'.format(time.time() - start_time))
        print('*******************************************************************')


if __name__ == "__main__":
    # Parsing arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_data_path', type=str, default=None, help='The path the csv file saving dataset.')
    parser.add_argument('--new_video_clip_rgb_path', type=str, default=None, help='The directory path saving all rgb video clips.')
    parser.add_argument('--new_video_clip_opt_path', type=str, default=None, help='The directory path saving all opt video clips.')
    parser.add_argument('--video_rgb_frame_root_path', type=str, default=None, help='The directory to save rgb frames.')
    parser.add_argument('--video_opt_frame_root_path', type=str, default=None, help='The directory to save opt frames.')
    parser.add_argument('--video_fused_frame_root_path', type=str, default=None, help='The directory to save fused frames.')
    parser.add_argument('--extract_fps', type=int, default=25, help='The fps used to extract frames.')
    parser.add_argument('--process_start_id', type=int, default=0, help='The start id of video clip list that will be processed.')
    parser.add_argument('--process_end_id', type=int, default=0, help='The end id of video clip list that will be processed.')
    parser.add_argument('--key_frame_front_num', type=int, default=30, help='')
    parser.add_argument('--key_frame_rear_num', type=int, default=30, help='')
    parser.add_argument('--key_frame_redundant', type=int, default=5, help='')
    parser.add_argument('--opt_flow_fusion_length', type=int, default=4, help='')

    args = parser.parse_args()

    if not os.path.exists(args.video_rgb_frame_root_path):
        os.mkdir(args.video_rgb_frame_root_path)
    if not os.path.exists(args.video_opt_frame_root_path):
        os.mkdir(args.video_opt_frame_root_path)

    extract_frames_from_rgb_and_opt_video_clip(csv_data_path=args.csv_data_path,
                                               new_video_clip_rgb_path=args.new_video_clip_rgb_path,
                                               new_video_clip_opt_path=args.new_video_clip_opt_path,
                                               video_rgb_frame_root_path=args.video_rgb_frame_root_path,
                                               video_opt_frame_root_path=args.video_opt_frame_root_path,
                                               video_fused_frame_root_path=args.video_fused_frame_root_path,
                                               extract_fps=args.extract_fps,
                                               process_start_id=args.process_start_id,
                                               process_end_id=args.process_end_id,
                                               key_frame_front_num=args.key_frame_front_num,
                                               key_frame_rear_num=args.key_frame_rear_num,
                                               key_frame_redundant=args.key_frame_redundant,
                                               opt_flow_fusion_length=args.opt_flow_fusion_length)

