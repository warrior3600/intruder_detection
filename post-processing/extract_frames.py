import os
import cv2
import pandas as pd
import argparse

args_parser = argparse.ArgumentParser(description='Trim Clips from Videos.')
args_parser.add_argument('--path_to_csv',type=str,help='path to csv file which contains clips video path.')
args_parser.add_argument('--path_to_save_frames',type=str,help='path where extracted frames will be stored.')
args_parser.add_argument('--skip_frame',type=int,help='skip every nth frame.',default=10)
args_parser.add_argument('--frame_width',type=int,help='Width of the frame in pixels.')
args_parser.add_argument('--frame_height',type=int,help='Height of the frame in pixels.')

def extract_frames(clip_file_path,path_to_save,skip_frame,frame_width,frame_height):
    clip_cap = cv2.VideoCapture(clip_file_path)
    count = 0
    
    while True:
        flag,frame = clip_cap.read()
        if flag:
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            #frame = frame.resize(frame,(frame_height,frame_width))

            if (count % skip_frame) == 0:
                filename = str(clip_file).split(".")[0] + "_img_" + str(count) + ".jpg"
                
                cv2.imwrite(os.path.join(path_to_save,filename),frame)
                print("[INFO] : Frames being storing in '%s' as filename : '%s'" %(path_to_save,filename))
            count += 1

        if clip_cap.get(cv2.CAP_PROP_POS_FRAMES) == clip_cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break

args = args_parser.parse_args()
path_to_csv = args.path_to_csv
path_to_save = args.path_to_save_frames
skip_frame = args.skip_frame
frame_width = args.frame_width
frame_height = args.frame_height

df = pd.read_csv(path_to_csv)

for index,row in df.iterrows():
    clip_file = df.loc[index,'clip_video_filename']
    clip_filename = str(clip_file).split(".")[0]
    
    if not os.path.exists(os.path.join(path_to_save,clip_filename)):
        try:
            os.makedirs(os.path.join(path_to_save,clip_filename))
            print("[INFO] : New directory create as '%s'" % clip_filename)
        except OSError as error:
            print("[ERROR] : Error while creating new directory as '%s'" % clip_filename)
    
    path_to_save_frame = os.path.join(path_to_save,clip_filename)
    extract_frames(clip_file,path_to_save_frame,skip_frame,frame_width,frame_height)
    

