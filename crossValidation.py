# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:11:30 2020

@author: josel
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

data_box =[]
datasetMode = ["Normal","Silent","Whispered"]
final_result={}
for mode in datasetMode:
    path = "results_Xnorm_"+mode+"_Digits_DNN.txt"
    f=open(path, "r")    
    results = json.loads(f.read())
    f.close()
    
    keysList =["2","3","4","5"] 
    plotResults = {}
    
    listResults = []
    for keyL in keysList:
        crossResult={}
        plotResults[keyL]={}
        for utter in results:     
            crossResult[utter]=[]
            for fold in results[utter]:
                result = results[utter][fold][keyL]
                crossResult[utter].append(result)
                
            plotResults[keyL][utter]=crossResult[utter]
            
            # crossResult[utter] = str(round((sum(crossResult[utter])/len(crossResult[utter])),2))+"pmsign"+ str(round(np.std(crossResult[utter]),2))
    
            crossResult[utter] = (round((sum(crossResult[utter])/len(crossResult[utter])),2), round(np.std(crossResult[utter]),2))
        data_box.append(plotResults)    
        listResults.append(crossResult)
    final_result[mode]=listResults
    

# ###############################################################################
mResult={}
state=5
for mode in final_result:
    listR = []
    for utterance in final_result[mode][state-2]:
        listR.append(final_result[mode][state-2][utterance][0])
    # mResult[mode] = (sum(listR)/len(listR), np.std(listR))
    mResult[mode] = str(round((sum(listR)/len(listR)),2))+"$pmsign$"+ str(round(np.std(listR),2))
      
        
    

    
###############################################################################
               #        Graficar accuracy vs numero de estados                    #  
###############################################################################
# phrases_ex=["p0","p3"]

# phrases_exDic = {phrases_ex[0]:"Zero",phrases_ex[1]:"Three"}

# xAxis = [2,3,4,5]

# color=['r','g','b']
# for mode in final_result:
#     color_c=0
#     for selecUtter in phrases_ex:
#         yAxis = []  
#         for n_states in final_result[mode]:
#             yAxis.append(n_states[selecUtter][0])
#         if mode == "Normal":
#             plotType = '-'
#         if mode == "Silent":
#             plotType = '--'
#         if mode == "Whispered":
#             plotType = '-.'
        
#         plt.plot(xAxis, yAxis,plotType+color[color_c], label =phrases_exDic[selecUtter]+' ('+mode+')')
#         color_c+=1

    
# ax = plt.subplot(111)
# chartBox = ax.get_position()
# ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])



# plt.xticks(np.arange(2, 6, 1.0))
# # plt.yticks(np.arange(50, 100, 5.0))

# plt.xlabel('Number of states in HMM')
# plt.ylabel('Accuracy (%)')
# plt.legend(bbox_to_anchor=(1, 0.8))
# plt.show()
    

    
          

###############################################################
#########             Boxplots de reusltados            ########
#################################################################
#####################################################
# phrases = ["Excuse me","Goodbye","Hello","How are you","Nice to meet you","See you","I'm sorry","Thank you","Have a good time","You're welcome"]
# digits = ["zero","one","two","three","four","five","six","seven","eight","nine"]
# data_box =[]
# final_result={}
# path = "results_Xnorm_inner_Silent_Digits_ViolaJ.txt"
# f=open(path, "r")    
# results = json.loads(f.read())
# f.close()

# keysList =["2","3","4","5"] 
# plotResults = {}

# listResults = []
# for keyL in keysList:
#     crossResult={}
#     plotResults[keyL]={}
#     for utter in results:     
#         crossResult[utter]=[]
#         for fold in results[utter]:
#             result = results[utter][fold][keyL]
#             crossResult[utter].append(result)
            
#         plotResults[keyL][utter]=crossResult[utter]
        
#         crossResult[utter] = (round((sum(crossResult[utter])/len(crossResult[utter])),2), round(np.std(crossResult[utter]),2))
#     listResults.append(crossResult)
    
# ##Correr data para cada modos del dataset, cada data pertenece a un tipo
# data=[]
# for utterance in plotResults["2"]:
#     utterance_0=plotResults["2"][utterance]
#     data.append(utterance_0)
    
# data2=[]
# for utterance in plotResults["2"]:
#     utterance_0=plotResults["2"][utterance]
#     data2.append(utterance_0)

# data3=[]
# for utterance in plotResults["2"]:
#     utterance_0=plotResults["2"][utterance]
#     data3.append(utterance_0)

# # Create a figure instance
# fig = plt.figure(1, figsize=(10, 6))

# # Create an axes instance
# ax = fig.add_subplot(111)
# plt.ylim(75, 100)
# # Create the boxplot
# bp = ax.boxplot(data,positions=np.array(range(len(data)))*2.0-0.5,widths=0.4)
# blp = ax.boxplot(data2,positions=np.array(range(len(data2)))*2.0+0.0,widths=0.4)
# blb = ax.boxplot(data3,positions=np.array(range(len(data2)))*2.0+0.5,widths=0.4)
# set_box_color(bp, '#D7191C') # colors are from http://colorbrewer2.org/
# set_box_color(blp, '#2C7BB6')
# set_box_color(blb, 'green')
# # draw temporary red and blue lines and use them to create a legend
# plt.plot([], c='#D7191C', label='Normal')
# plt.plot([], c='#2C7BB6', label='Whispered')
# plt.plot([], c='green', label='Silent')

# plt.legend(loc='lower left')


# plt.xlabel("Utterances")
# plt.ylabel("Accuracy (%)")


# plt.xticks(range(0, 10 * 2, 2),digits,rotation = 60)
# plt.tight_layout()

# # Save the figure
# fig.savefig('fig1.png', bbox_inches='tight')
# plt.show()