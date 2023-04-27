import os
import numpy as np
from tqdm import tqdm
import time
import cv2


def rewrite_video_clip_and_generate_optical_flow(raw_video_clip_path=None,
                                                 new_video_clip_rgb_path=None,
                                                 new_video_clip_opt_path=None,
                                                 process_start_id=None,
                                                 process_end_id=None):
    """
    Raw video clips have various frame rates that are normally lower than 25, so this function is used to rewrite the
    clips at a given frame rate and generate the corresponding optical flow from the raw video clip.
    (Note: using raw video clip for optical flow is necessary, because if using rewritten video clip, there will be two
    consecutive frames are the same and cause 0 valued optical flow.)
    :param raw_video_clip_path:
    :param new_video_clip_rgb_path:
    :param new_video_clip_opt_path:
    :param process_start_id:
    :param process_end_id:
    :return:
    """
    video_clip_list = os.listdir(raw_video_clip_path)
    video_clip_list.sort()
    for video_clip in tqdm(video_clip_list[process_start_id: process_end_id]):
        start_time = time.time()
        video_clip_path = os.path.join(raw_video_clip_path, video_clip)

        # The video feed is read in as a VideoCapture object
        video_cap = cv2.VideoCapture(video_clip_path)

        # Write frames and optical flow at the same fps, so we can keep the curract correspondance.
        frame_width = int(video_cap.get(3))
        frame_height = int(video_cap.get(4))
        # Important note: cv2 can only write with the same fps as original video
        new_video_clip_fps = video_cap.get(cv2.CAP_PROP_FPS)
        clip_rgb_out_name = os.path.join(new_video_clip_rgb_path, 'rgb_{}'.format(video_clip))
        clip_opt_out_name = os.path.join(new_video_clip_opt_path, 'opt_{}'.format(video_clip))
        video_frame_rgb_out = cv2.VideoWriter(clip_rgb_out_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                              new_video_clip_fps, (frame_width, frame_height))
        video_frame_opt_out = cv2.VideoWriter(clip_opt_out_name, cv2.VideoWriter_fourcc(*'mp4v'),
                                              new_video_clip_fps, (frame_width, frame_height))

        # ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
        ret, first_frame = video_cap.read()
        video_frame_rgb_out.write(first_frame)  # wirte the first frame
        video_frame_opt_out.write(first_frame)  # the 1st frame does not have optical flow so just wirte the first frame

        # Read grayscale image because we only need the luminance channel for detecting edges - less computationally expensive
        prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Creates an image filled with zero intensities with the same dimensions as the frame
        mask = np.zeros((prev_frame_gray.shape[0], prev_frame_gray.shape[1], 3), dtype=np.uint8)

        # Set image saturation to maximum
        mask[..., 1] = 255
        while video_cap.isOpened():
            # Read a frame
            read_success, curr_frame = video_cap.read()
            if not read_success:
                print('Reach the end of video!')
                break
            # print(ret)
            # Converts each frame to grayscale - we previously only converted the first frame to grayscale
            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # Calculates dense optical flow by Farneback method
            flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Sets image hue according to the optical flow direction
            mask[..., 0] = angle * 180 / np.pi / 2

            # Sets image value according to the optical flow magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            # Converts HSV to RGB (BGR) color representation
            rgb_opt = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

            # Save
            # cv2.imwrite(os.path.join(video_clip_frame_optical_flow_folder_path, 'optical_flow_{}'.format(frame_name)), rgb_opt)
            video_frame_rgb_out.write(curr_frame)
            video_frame_opt_out.write(rgb_opt)

            prev_frame_gray = curr_frame_gray

        # When everything done, release the video capture and video write objects
        video_frame_rgb_out.release()
        video_frame_opt_out.release()
        print('Processed {} costs {}s.'.format(video_clip, time.time()-start_time))


if __name__ == "__main__":
    # Parsing arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_video_clip_path', type=str, default=None, help='The directory path saving all video clips.')
    parser.add_argument('--new_video_clip_rgb_path', type=str, default=None, help='The directory path saving new video clips.')
    parser.add_argument('--new_video_clip_opt_path', type=str, default=None, help='The directory path saving optical flow of new video clips.')
    parser.add_argument('--process_start_id', type=int, default=0, help='The start id of video clip list that will be processed.')
    parser.add_argument('--process_end_id', type=int, default=0, help='The end id of video clip list that will be processed.')

    args = parser.parse_args()

    # Create directories
    if not os.path.exists(args.new_video_clip_rgb_path):
        os.makedirs(args.new_video_clip_rgb_path)
    if not os.path.exists(args.new_video_clip_opt_path):
        os.makedirs(args.new_video_clip_opt_path)
    rewrite_video_clip_and_generate_optical_flow(raw_video_clip_path=args.raw_video_clip_path,
                                                 new_video_clip_rgb_path=args.new_video_clip_rgb_path,
                                                 new_video_clip_opt_path=args.new_video_clip_opt_path,
                                                 process_start_id=args.process_start_id,
                                                 process_end_id=args.process_end_id)

