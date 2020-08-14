import numpy as np

from hmmlearn import hmm
import json 
#import matplotlib.pyplot as plt

from featuresProcessing import normalize_coordinates, loop_over_static, derivate, normalize_coordinates_2
import pandas as pd
import sys
from auxiliars.generateMatrixTransi import genTransMatrix  
from auxiliars.hmmModelGen import HMMTrainer
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
import scikitplot as skplt


    
##############################################################################################    

if __name__=='__main__':

    f=open("LipsCoordinates_Normal_20coor_Phrases_ViolaJ.txt", "r")    
    contents = json.loads(f.read())
    f.close()
    mSize = 44
    #diccionario con fshape para cada frame de todos los videos
    normalized = normalize_coordinates_2(contents)
    # normalized = derivate(contents) #comentar
    # normalized = contents
    y_true = []
    y_pred = []
      
    keys = []
    for key in normalized.keys():
        if key[9:] == "p0":
            keys.append(key)         
    keys.sort()
    #############################

    
    #############################

    keys_train = [keys[i] for i in range(0,len(keys),5)]      
    keys_test = []
    for key in keys:
        if key not in keys_train:
            keys_test.append(key)
        ################################################ USING STATIC ########################################

    hmm_models = []

    ########## Training ################3   
    t_c = 0
    for key in keys_train:
        t_c+=1 
        X = loop_over_static(normalized,key,mSize)          
        hmm_trainer = HMMTrainer(n_components=4)
#                    print("\n Entrenando... ")
#                    print(str(t_c) + " de " + str(len(keys_train)) )
    
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, key))
        hmm_trainer = None
                        
    ########### Testing #################
    test_count = 0
    for key in keys_test:
        X = loop_over_static(normalized,key,mSize)
        max_score = [float("-inf")]
        output_label = None
        model_count = 1
        for model in hmm_models:
            hmm_model, label = model
            # score = hmm_model.get_score(X)
            model_count+=1
            print(key,label,score)
            if score > max_score:
                max_score = score
                output_label = label
    
        key = key.split('_')
        output = output_label.split('_')
        y_true.append(key[0])
        y_pred.append(output[0])
        if(key[0] == output[0]):
            test_count+=1
        # print( "\nTrue:", key[0])
        # print("Predicted:", output[0])
        # print('-'*50)
            
    result_t = test_count/len(keys_test)
    print("The accuracy is: " + str(result_t))
    conf =confusion_matrix(y_true, y_pred) 
    print(classification_report(y_true,y_pred,digits=3))
 
    
    
    ##############################Print confusion matrix
    # skplt.metrics.plot_confusion_matrix(
    # y_true, 
    # y_pred,
    # figsize=(20,20))

   
    ##############################  Print scale different users  ###############################
    # utt="p1"
    # s1_r1=[]
    # for frame in normalized["S001_R01_"+utt]:
    #     s1_r1.append(normalized["S001_R01_"+utt][frame][13][0])
        
    # s1_r2=[]
    # for frame in normalized["S001_R02_"+utt]:
    #     s1_r2.append(normalized["S001_R02_"+utt][frame][13][0])
    # s1_r3=[]
    # for frame in normalized["S001_R03_"+utt]:
    #     s1_r3.append(normalized["S001_R03_"+utt][frame][13][0])
    # s1_r4=[]
    # for frame in normalized["S001_R04_"+utt]:
    #     s1_r4.append(normalized["S001_R04_"+utt][frame][13][0])
    # s1_r5=[]
    # for frame in normalized["S001_R05_"+utt]:
    #     s1_r5.append(normalized["S001_R05_"+utt][frame][13][0])
    # s2_r1=[]
    # for frame in normalized["S021_R01_"+utt]:
    #     s2_r1.append(normalized["S021_R01_"+utt][frame][13][0])
    # s3_r1=[]
    # for frame in normalized["S014_R01_"+utt]:
    #     s3_r1.append(normalized["S014_R01_"+utt][frame][13][0])
    # s4_r1=[]
    # for frame in normalized["S016_R01_"+utt]:
    #     s4_r1.append(normalized["S016_R01_"+utt][frame][13][0])        
  
    # s1_r1=[x / s1_r1[0] for x in s1_r1]
    # s1_r2=[x / s1_r2[0] for x in s1_r2]
    # s1_r3=[x / s1_r3[0] for x in s1_r3]
    # s1_r4=[x / s1_r4[0] for x in s1_r4]
    # s1_r5=[x / s1_r5[0] for x in s1_r5]
    # s2_r1=[x / s2_r1[0] for x in s2_r1]
    # s3_r1=[x / s3_r1[0] for x in s3_r1]
    # s4_r1=[x / s4_r1[0] for x in s4_r1]


    # frames=50
    # s1_r1=s1_r1[:frames]
    # s1_r2=s1_r2[:frames]
    # s1_r3=s1_r3[:frames]
    # s1_r4=s1_r4[:frames]
    # s1_r5=s1_r5[:frames]
    # s2_r1=s2_r1[:frames]
    # s3_r1=s3_r1[:frames]
    # s4_r1=s4_r1[:frames]


    # plt.ylim(0.8, 1.2)

    # y = np.arange(0,frames)
    
    # plt.plot( y,s1_r1, label = "Person 1") 
    # # plt.plot( y,s1_r2, label = "line 2") 
    # # plt.plot( y,s1_r3, label = "Repetition 2") 
    # # plt.plot( y,s1_r4, label = "Repetition 3") 
    # # plt.plot( y,s1_r5, label = "Repetition 4") 
    # plt.plot( y,s2_r1, label = "Person 2")
    # plt.plot( y,s3_r1, label = "Person 3") 
    # plt.plot( y,s4_r1, label = "Person 4") 

    # plt.xlabel('Frame number') 
    # plt.ylabel('Scale')
    
    # plt.legend() 
    # plt.show() 



      



