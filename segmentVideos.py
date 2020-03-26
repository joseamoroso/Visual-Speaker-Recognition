#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 01:31:45 2020

@author: jose
"""

import ffmpeg
import pandas as pd
import glob
import time
import os

def cut_video(in_path,out_path,start,end):
    in_file = ffmpeg.input(in_path)
    (
        ffmpeg
        .trim(in_file,start=start,end=end)
        .setpts ('PTS-STARTPTS')
        .output(out_path)
        .run()
    )
    


path = "AVOriginalDataset/Phrases/*" 
video_folders = glob.glob(path)
for video_folder in video_folders:
    v_path = glob.glob(video_folder+"/*.mp4")
    out_string = v_path[0].split("/")
    out_string = out_string[3].split("_")
    out_string = out_string[0]+"_"+out_string[4]+"_"
    csv_paths = glob.glob(video_folder+"/*.csv")
    for csv_path in csv_paths:
        path_len=len(csv_path.split('_'))
        if path_len == 10:
            data = pd.read_csv(csv_path) 
            for i in range(len(data)):
                start = data.iloc[i][0] / 10000000
                end = data.iloc[i][1] / 10000000
                utterance = data.iloc[i][4]
                out_string_f = "AVSegmentedDataset/"+ out_string + utterance + ".mp4"
                cut_video(v_path[0],out_string_f,start,end)
                print("Video " + out_string + utterance + ".mp4 created")

#ELIMINA G AUDIOS
result_video = glob.glob("AVSegmentedDataset/*")
for elem in result_video:
    elem2 = elem.split("/")
    if 'g' in elem2[1]:
        os.remove(elem)
        print(elem + " DELETED \n")

            
            
