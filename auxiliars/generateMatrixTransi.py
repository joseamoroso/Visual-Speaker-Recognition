import numpy as np

def genTransMatrix(states):
    startprobPrior = np.zeros(states,dtype=np.float)
    startprobPrior[0]= 1   
    
    transmatPrior = np.zeros((states, states),dtype=np.float) 
    tmp_prob = 0.5
    
    for i in range(len(transmatPrior)-1):
        for j in range(len(transmatPrior[i])):
            if i==j:
                transmatPrior[i][j] = tmp_prob
                transmatPrior[i][j+1] = tmp_prob
    
    transmatPrior[states-1][states-1]=1.0
    
    return transmatPrior