#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 01:31:45 2020

@author: jose
"""

import ffmpeg
import pandas as pd
import glob
import os

def cut_video(input_path, output_path, start, end):
    input_stream = ffmpeg.input(input_path)

    vid = (
        input_stream.video
        .trim(start=start, end=end)
        .setpts('PTS-STARTPTS')
    )
    aud = (
        input_stream.audio
        .filter_('atrim', start=start, end=end)
        .filter_('asetpts', 'PTS-STARTPTS')
    )

    joined = ffmpeg.concat(vid, aud, v=1, a=1).node
    output = ffmpeg.output(joined[0], joined[1], output_path)
    output.run()
    
modes = ["Normal","Silent","Whispered"]

path = "AVOriginalDataset\Phrases\*" 
video_folders = glob.glob(path)
for video_folder in video_folders:
    v_path = glob.glob(video_folder+"\*.mp4")
    out_string = v_path[0].split("\\")
    out_string = out_string[3].split("_")
    out_string = out_string[0]+"_"+out_string[4]+"_"
    csv_paths = glob.glob(video_folder+"\*.csv")
    for csv_path in csv_paths:
        path_len=len(csv_path.split('_'))
        if path_len == 9:
            data = pd.read_csv(csv_path) 
            for i in range(len(data)):
                start = data.iloc[i][0] / 10000000
                end = data.iloc[i][1] / 10000000
                if "Digits" in path:
                    utterance = "p"+str(data.iloc[i][4])
                else:
                    utterance = data.iloc[i][4]
                
                if "C01" in v_path[0]:
                    out_string_f = r"AVSegmentedDataset\Phrases\Normal\\"+ out_string + utterance + ".mp4"
                
                if "C02" in v_path[0]:
                    out_string_f = "AVSegmentedDataset\\Phrases\Whispered\\"+ out_string + utterance + ".mp4"

                if "C03" in v_path[0]:
                    out_string_f = "AVSegmentedDataset\\Phrases\Silent\\"+ out_string + utterance + ".mp4"
                
                if os.path.exists(out_string_f):
                    print("Already")
                    continue
                cut_video(v_path,out_string_f,start,end)
                print("Video " + out_string + utterance + ".mp4 created \n")



#ELIMINA G AUDIOS
for mod in modes:                
    result_video = glob.glob(r"AVSegmentedDataset\\Phrases\\"+ mod +"\\*")
    for elem in result_video:
        elem2 = elem.split("\\")
        if 'g' in elem2[5]:
            os.remove(elem)
            print(elem + " DELETED \n")

            
            
