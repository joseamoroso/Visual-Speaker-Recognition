# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:11:30 2020

@author: josel
"""
import json
import numpy as np

f=open("exp1Results/results_test_1_normalized.txt", "r")    
results = json.loads(f.read())
f.close()

keysList =["(2, 1)","(3, 1)","(4, 1)","(5, 1)","(6, 1)","(7, 1)","(8, 1)","(9, 1)"] 

listResults = []
for keyL in keysList:
    crossResult={}
    for utter in results:
        crossResult[utter]=[]
        for fold in results[utter]:
            result = results[utter][fold][keyL]
            crossResult[utter].append(result)
        crossResult[utter] = (sum(crossResult[utter])/len(results[utter]), np.std(crossResult[utter]))
    listResults.append(crossResult)
    
f=open("exp1Results/crossExp1.txt", "+w")    
cross_json = json.dumps(listResults)
f.write(cross_json)
f.close() 
