# Class to handle all HMM related processing
import numpy as np

from hmmlearn import hmm
from auxiliars.generateMatrixTransi import genTransMatrix 

# Class to handle all HMM related processing

class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4 ,cov_type='diag'):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.trans_matrix,self.prior_prob = genTransMatrix(self.n_components)
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                    covariance_type=self.cov_type,init_params="cm", params="cmt",
                    transmat_prior=self.trans_matrix,startprob_prior =self.prior_prob)
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