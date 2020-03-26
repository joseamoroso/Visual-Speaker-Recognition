import os
import argparse
import warnings
import numpy as np

from hmmlearn import hmm
import json 
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing


NUM_CORD = 12
 
def normalize_coordinates(coord_list):
    normalize_coord = {}
    features_param ={}
    for speaker in coord_list:
        normalize_coord[speaker]={}
        features_param[speaker]={}
        for speaker_frame in coord_list[speaker]:
            normalize_coord[speaker][speaker_frame] = []
            x_l = coord_list[speaker][speaker_frame][0][0]
            x_r = coord_list[speaker][speaker_frame][6][0]
            y_l = coord_list[speaker][speaker_frame][0][1]
            y_r = coord_list[speaker][speaker_frame][6][1]    
            x_c = (x_l + x_r) / 2
            y_c = (y_l + y_r) / 2
            alpha = math.atan2(y_r-y_l,x_r-x_l)
            s = math.sqrt((((x_l-x_r)**2)+((y_l-y_r)**2))/2)
            features_param[speaker][speaker_frame] =[x_c,y_c,alpha,s]

            for coord in coord_list[speaker][speaker_frame]:
                xi = coord[0]
                yi = coord[1]
                xi_prime = (((xi - x_c) * math.cos(alpha)) + ((yi - y_c)* math.sin(alpha))) / s
                yi_prime = ((-(xi - x_c) * math.sin(alpha)) + ((yi - y_c)* math.cos(alpha))) / s
                normalize_coord[speaker][speaker_frame].append([xi_prime,yi_prime])
    return normalize_coord,features_param


#Recibe un videos y calcula derivadas entre las coordenadas de frames consecutivos
def derivate (video_frames):
    f_temp_shape = {}
    num_frames = len(video_frames)
    frames_keys = list(video_frames.keys())
    for i in range(num_frames-1):
        f_temp_shape["f_temp_shape"+str(i)]=[]
        for coor in range(NUM_CORD):
            x_act = video_frames[frames_keys[i]][coor][0]
            x_next = video_frames[frames_keys[i+1]][coor][0]
            y_act = video_frames[frames_keys[i]][coor][1]
            y_next = video_frames[frames_keys[i+1]][coor][1]
            x_derivate = (x_next - x_act) / 0.033333
            y_derivate =  (y_next - y_act) / 0.033333
            f_temp_shape["f_temp_shape"+str(i)].append((x_derivate,y_derivate))
    return f_temp_shape

#Recibe un videos y calcula derivadas entre las coordenadas de frames consecutivos
def derivate_f (features_dict):
    f_temp_shape = {}
    num_frames = len(features_dict)
    frames_keys = list(features_dict.keys())
    for i in range(1,num_frames-1):
        

        x_bef = features_dict[frames_keys[i-1]][0]
        x_next = features_dict[frames_keys[i+1]][0]
        y_bef = features_dict[frames_keys[i-1]][1]
        y_next = features_dict[frames_keys[i+1]][1]
        alp_bef = features_dict[frames_keys[i-1]][2]
        alp_next = features_dict[frames_keys[i+1]][2]
        scale_bef = features_dict[frames_keys[i-1]][3]
        scale_next =features_dict[frames_keys[i+1]][3]
        
        
        
        x_derivate = (x_next - x_bef) / 2
        y_derivate =  (y_next - y_bef) / 2
        s_derivate = (scale_next - scale_bef) /2
        apl_derivate = (alp_next - alp_bef)/2
        f_temp_shape["f_temp_shape_"+str(i)] = [x_derivate,y_derivate,apl_derivate,s_derivate]
                    
    return f_temp_shape
 
        
    

# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

def loop_over_static(norm_dict,key):
    frame_features = norm_dict[key] #Lista de diccionarios de frames
    f_temp_shape_dict = list(frame_features)    
    X = np.array([])
    for f_temp_shape_values in f_temp_shape_dict:
        if len(X) == 0:
            X = frame_features[f_temp_shape_values]
        else:
            X = np.append(X,frame_features[f_temp_shape_values],axis=0)
    
#    X_test = np.hsplit(X,2) 
#    a = X_test[0].tolist()
#    b = X_test[1].tolist()    
#    X_test_2 = np.concatenate([a,b])
#    lengths = [len(a), len(b)]
    return X
    

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=6, cov_type='diag', n_iter=3000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)
    def predict_model(self, input_data):
        return self.model.predict(input_data)

#if __name__=='__main__':

f=open("lipsPoints_avDataset_v2.txt", "r")    
contents = json.loads(f.read())
f.close()
#diccionario con fshape para cada frame de todos los videos
normalized,features = normalize_coordinates(contents)
hmm_models = []

#Delete samples with g label -> 2 samples
del_array = []
for elem in normalized:
    if len(elem) < 11:
        del_array.append(elem)

for elem in del_array:
    del normalized[elem]

#Generate train and test labels
#keys = [key for key in normalized.keys()]
    
keys = []
for key in normalized.keys():
    if key[9:] == "p9":
        keys.append(key)


keys.sort()
keys_train = [keys[i] for i in range(0,len(keys),5)]

#########para test 2
#for i in range(1,len(keys),5):
#    keys_train.append(keys[i])
#keys_train.sort()
#
##################33

keys_test = []
for key in keys:
    if key not in keys_train:
        keys_test.append(key)

################################################ USING STATIC ########################################
########### Training ################3   
#t_c = 0
#for key in keys_train:
#    t_c+=1 
#    X, lenght = loop_over_static(normalized)          
#    hmm_trainer = HMMTrainer()
#    print("\n Entrenando... ")
#    print(str(t_c) + " de " + str(len(keys_train)) )
#    
#    hmm_trainer.train(X,lenght)
#    hmm_models.append((hmm_trainer, key))
#    hmm_trainer = None
#                
#        #########3 test 2############
#n_keys_train=[]
#for i in range(len(keys_train)-1):
#    keys_train[i] = [keys_train[i],keys_train[i+1]]
#for i in range(len(keys_train)-1):
#    if i % 2 == 0:
#        n_keys_train.append(keys_train[i])
#    
#t_c = 0
#for key in n_keys_train:
#    hmm_trainer = HMMTrainer()
#    for s_key in key:
#        t_c+=1 
#        X, lenght = loop_over_static(normalized,s_key)          
#        print("\n Entrenando... ")
#        print(str(t_c) + " de " + str(len(keys_train)) )        
#        hmm_trainer.train(X,lenght)
#    hmm_models.append((hmm_trainer, key[0][:4]))
#    hmm_trainer = None
#                    
#            
############ Testing #################
#test_count = 0 
#for key in keys_test:
#    X,lenght = loop_over_static(normalized,key)
#    max_score = [float("-inf")]
#    output_label = None
#    for model in hmm_models:
#        hmm_model, label = model
#        score = hmm_model.get_score(X)
##        print(key,label,score)
#        if score > max_score:
#            max_score = score
#            output_label = label
#
#    key = key.split('_')
#    output = output_label.split('_')
#    if(key[0] == output[0]):
#        test_count+=1
#    print( "\nTrue:", key[0])
#    print("Predicted:", output[0])
#    print('-'*50)
#
#result_t = test_count/len(keys_test)
#print(result_t)
#   
############################################### USING DYNAMIC ########################################

########## Training ################3   

for key in keys_train:
    frame_features = features[key]
    f_temp_shape = derivate_f(frame_features)
    X = []

    for f_temp_shape_values in f_temp_shape:
        if len(X) == 0:
            X = [f_temp_shape[f_temp_shape_values]]
#            print(X )
#            print("\n")
        else:
#            X = np.append(X,f_temp_shape[f_temp_shape_values],axis=0)
            X.append(f_temp_shape[f_temp_shape_values] )
    
    X = np.asarray(X)
    hmm_trainer = HMMTrainer()
    print("entrenando...")
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, key))
    hmm_trainer = None
                
#            
# ########### Testing #################
test_count = 0 
for key in keys_test:
    frame_features = features[key]
    f_temp_shape = derivate_f(frame_features)
    X = []

    for f_temp_shape_values in f_temp_shape:
        if len(X) == 0:
            X = [f_temp_shape[f_temp_shape_values]]
#            print(X )
#            print("\n")
        else:
#            X = np.append(X,f_temp_shape[f_temp_shape_values],axis=0)
            X.append(f_temp_shape[f_temp_shape_values] )
            
    X = np.asarray(X)
    max_score = [float("-inf")]
    output_label = None
    for model in hmm_models:
        hmm_model, label = model
        score = hmm_model.get_score(X)
#        print(key,label,score)
        if score > max_score:
            max_score = score
            output_label = label

    key = key.split('_')
    output = output_label.split('_')
    if(key[0] == output[0]):
        test_count+=1
    print( "\nTrue:", key[0])
    print("Predicted:", output[0])
    print('-'*50)

result_t = test_count/len(keys_test)
print(result_t)


