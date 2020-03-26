import numpy as np

from hmmlearn import hmm
import json 
#import matplotlib.pyplot as plt
#from sklearn import preprocessing

from featuresProcessing import normalize_coordinates, loop_over_static

    

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=6, cov_type='full', n_iter=2):
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
    
##############################################################################################    

if __name__=='__main__':

    f=open("AV_lips_coordinates_v0.txt", "r")    
    contents = json.loads(f.read())
    f.close()
    #diccionario con fshape para cada frame de todos los videos
    normalized = normalize_coordinates(contents)
      
    #Generate train and test labels
    #keys = [key for key in normalized.keys()]        
    keys = []
    for key in normalized.keys():
        if key[9:] == "p9":
            keys.append(key)
     
    keys.sort()
    keys_train = [keys[i] for i in range(0,len(keys),5)]
    
    keys_test = []
    for key in keys:
        if key not in keys_train:
            keys_test.append(key)
    results={}
    ################################################ USING STATIC ########################################
    for i in range (2,11):
        for j in range(1,15):
            try:
                hmm_models = []
            
                ########## Training ################3   
                t_c = 0
                for key in keys_train:
                    t_c+=1 
                    X = loop_over_static(normalized,key)          
                    hmm_trainer = HMMTrainer(n_components=i, n_iter=j)
#                    print("\n Entrenando... ")
#                    print(str(t_c) + " de " + str(len(keys_train)) )
                
                    hmm_trainer.train(X)
                    hmm_models.append((hmm_trainer, key))
                    hmm_trainer = None
                                    
                ########### Testing #################
                test_count = 0
                for key in keys_test:
                    X = loop_over_static(normalized,key)
                    max_score = [float("-inf")]
                    output_label = None
                    model_count = 1
                    for model in hmm_models:
                        hmm_model, label = model
                        score = hmm_model.get_score(X)
                        model_count+=1
                #        print(key,label,score)
                        if score > max_score:
                            max_score = score
                            output_label = label
                
                    key = key.split('_')
                    output = output_label.split('_')
                    if(key[0] == output[0]):
                        test_count+=1
#                        print( "\nTrue:", key[0])
#                        print("Predicted:", output[0])
#                        print('-'*50)
#                
                result_t = test_count/len(keys_test)
                print("Accuracy: " + str(result_t*100) + " %")
                results[str((i,j))] = result_t*100
            except:
                print("ERROR for: " + str((i,j)))
                continue

       
