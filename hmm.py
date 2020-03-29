import numpy as np

from hmmlearn import hmm
import json 
#import matplotlib.pyplot as plt

from featuresProcessing import normalize_coordinates, loop_over_static, derivate
import pandas as pd

    

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=2, cov_type='full', n_iter=1):
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
    # normalized = derivate(contents) #comentar
    # normalized = contents

    
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
    # for i in range (2,11):
    #     for j in range(1,15):
    #         try:
    hmm_models = []

    ########## Training ################3   
    t_c = 0
    for key in keys_train:
        t_c+=1 
        X = loop_over_static(normalized,key)          
        hmm_trainer = HMMTrainer()
#        hmm_trainer = HMMTrainer(n_components=i, n_iter=j)
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
    # results[str((i,j))] = result_t*100
            # except:
            #     print("ERROR for: " + str((i,j)))
            #     continue

# test_1_results =  {'(2, 1)': 10.897435897435898, '(2, 2)': 6.41025641025641, '(2, 3)': 7.051282051282051, '(2, 4)': 7.6923076923076925, '(2, 5)': 7.6923076923076925, '(2, 6)': 7.6923076923076925, '(2, 7)': 8.974358974358974, '(2, 8)': 6.41025641025641, '(2, 9)': 6.41025641025641, '(3, 1)': 9.615384615384617, '(3, 2)': 4.487179487179487, '(3, 3)': 4.487179487179487, '(3, 4)': 2.564102564102564, '(3, 5)': 2.564102564102564, '(3, 6)': 1.9230769230769231, '(3, 7)': 2.564102564102564, '(3, 8)': 1.9230769230769231, '(3, 9)': 2.564102564102564, '(4, 1)': 9.615384615384617, '(4, 2)': 4.487179487179487, '(4, 3)': 3.205128205128205, '(4, 4)': 2.564102564102564, '(4, 5)': 2.564102564102564, '(4, 6)': 0.641025641025641, '(4, 7)': 2.564102564102564, '(4, 8)': 2.564102564102564, '(4, 9)': 2.564102564102564, '(5, 1)': 9.615384615384617, '(5, 2)': 5.128205128205128, '(5, 3)': 3.8461538461538463, '(5, 4)': 3.205128205128205, '(5, 5)': 2.564102564102564, '(5, 6)': 2.564102564102564, '(5, 7)': 2.564102564102564, '(5, 8)': 2.564102564102564, '(5, 9)': 2.564102564102564, '(6, 1)': 8.974358974358974, '(6, 2)': 3.205128205128205, '(6, 3)': 3.205128205128205, '(6, 4)': 1.282051282051282, '(6, 5)': 1.282051282051282, '(6, 6)': 1.9230769230769231, '(6, 7)': 1.282051282051282, '(6, 8)': 2.564102564102564, '(6, 9)': 0.641025641025641, '(7, 1)': 9.615384615384617, '(7, 2)': 3.8461538461538463, '(7, 3)': 4.487179487179487, '(7, 4)': 2.564102564102564, '(7, 5)': 2.564102564102564, '(7, 6)': 0.641025641025641, '(7, 7)': 2.564102564102564, '(7, 8)': 1.282051282051282, '(7, 9)': 2.564102564102564, '(8, 1)': 8.974358974358974, '(8, 2)': 4.487179487179487, '(8, 3)': 2.564102564102564, '(8, 4)': 2.564102564102564, '(8, 5)': 2.564102564102564, '(8, 6)': 1.282051282051282, '(8, 8)': 2.564102564102564, '(8, 9)': 0.0, '(9, 1)': 9.615384615384617, '(9, 2)': 4.487179487179487, '(9, 3)': 3.205128205128205, '(9, 4)': 0.0, '(9, 5)': 0.641025641025641, '(9, 6)': 2.564102564102564, '(9, 7)': 2.564102564102564, '(9, 8)': 2.564102564102564, '(9, 9)': 2.564102564102564}
# df = pd.DataFrame(test_1_results, index=[0])

# df.to_csv('results_test1_derivates.csv', index = False)

