# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:11:30 2020

@author: josel
"""
import json
import numpy as np

f=open("results_test_2_normalized_normal.txt", "r")    
results = json.loads(f.read())
f.close()

keysList =["2","3","4","5"] 

listResults = []
for keyL in keysList:
    crossResult={}
    for utter in results:
        crossResult[utter]=[]
        for fold in results[utter]:
            result = results[utter][fold][keyL]
            crossResult[utter].append(result)
        crossResult[utter] = (sum(crossResult[utter])/len(crossResult[utter]), np.std(crossResult[utter]))
    listResults.append(crossResult)
    
# f=open("crossExp3.txt", "+w")    
# cross_json = json.dumps(listResults)
# f.write(cross_json)
# f.close() 
