# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from __future__ import division, print_function
import skfuzzy as fuzz
from sklearn.datasets import load_wine
import numpy as np
#import matplotlib.pyplot as plt
import collections
import random
import math



#Carrega o iris dataset em iris 
wine = load_wine()


dado_randomicos=[]
x_treino=[]
y_treino=[]
x_test=[]
y_test=[]
# armazenar o predict
d=[]

lista_comparacao=[ x for x in range(len(wine.data))]
percentual=0.90
# data=[]
# label=[]
# for n in range(math.trunc(percentual*59)):
    
#     data.append(wine.data[n])
#     label.append(wine.target[n])

# for n in range(math.trunc(percentual*59),math.trunc(percentual*130)):
    
#     data.append(wine.data[n])
#     label.append(wine.target[n])
    
# for n in range(math.trunc(percentual*130),math.trunc(percentual*178)):
    
#     data.append(wine.data[n])  
#     label.append(wine.target[n])
    

# data=np.array(data)
# label=np.array(label)   

# for n in range(math.trunc(percentual*59),59):
    
#     data.append(wine.data[n])
#     label.append(wine.target[n])

# for n in range(math.trunc(percentual*130),130):
    
#     data.append(wine.data[n])
#     label.append(wine.target[n])
    
# for n in range(math.trunc(percentual*178),178):
    
#     data.append(wine.data[n])  
#     label.append(wine.target[n])
    

 
while 1:
    aleatorio=random.randrange(0,len(wine.target))
    if aleatorio not in dado_randomicos:
        dado_randomicos.append(aleatorio)
    if math.trunc(len(wine.target)*percentual)==len(dado_randomicos):
        break
    
    
for n in dado_randomicos:
    x_treino.append(wine.data[n])
    y_treino.append(wine.target[n])

    if n in  lista_comparacao:
        lista_comparacao.remove(n)   


for n in lista_comparacao:
    x_test.append(wine.data[n])
    y_test.append(wine.target[n])

x_test=np.array(x_test)
y_test=np.array(y_test)

alldata3=x_test.transpose()       
    
x_treino=np.array(x_treino)    
xcont=collections.Counter(y_treino)




print(wine.data)
print(wine.feature_names)
print(list(wine.target_names))
alldata = x_treino
# np.vstack(iris.data[50:150,:])

alldata2 = alldata.transpose()
label = wine.target 




# a=np.vstack((iris.data[:coleta1,:],iris.data[50:(50+coleta2),:],iris.data[100:(100+coleta3),:]))
# b=np.hstack((iris.target[:coleta1],iris.target[50:(50+coleta2)],iris.target[100:(100+coleta3)]))
# a1=np.vstack((iris.data[20:50,:],iris.data[50:100,:],iris.data[100:110,:]))
# b1=np.hstack((iris.target[20:50],iris.target[50:100],iris.target[100:110]))
# alldata=a
# alldata2=alldata.transpose()
# label=b

ncenters = 3
#Implementa o Algoritmo Fuzzy C-means
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
alldata2, ncenters, 2, error=0.005, maxiter=1000, init=None)
cluster_membership = np.argmax(u, axis=0)



# c=collections.Counter(Newcluster_membership[:50])
# c1=np.array(c.most_common())
# c=collections.Counter(Newcluster_membership[50:100])
# c2=np.array(c.most_common())
# c=collections.Counter(Newcluster_membership[100:150])
# c3=np.array(c.most_common())

    
# print('Classificação: ',c1[0,0],'\n',(c1[0,1]/50)*100)
# print('Classificação:',c2[0,0],'\n',(c2[0,1]/50)*100)
# print('Classificação:',c3[0,0],'\n',(c3[0,1]/50)*100)
# print(((c1[0,1]+c2[0,1]+c3[0,1])/150)*100)
# """

# Generate uniformly sampled data spread across the range [0, 10] in x and y
# newdata = escolher os dados a serem testados

# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model



u1, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(alldata3
    , cntr, 2, error=0.005, maxiter=1000)
Newcluster_membership=np.argmax(u1,axis=0)

c=collections.Counter(Newcluster_membership[:59-xcont[0]])
print(c)
c1=np.array(c.most_common())
c=collections.Counter(Newcluster_membership[59-xcont[0]:130-xcont[0]-xcont[1]])
print(c)
c2=np.array(c.most_common())
c=collections.Counter(Newcluster_membership[130-xcont[1]-xcont[0]:178-xcont[2]-xcont[1]-xcont[0]])
print(c)
c3=np.array(c.most_common())

  
print('Classificação: ',c1[0,0],'\n',(c1[0,1]/(59-xcont[0]))*100)
print('Classificação:',c2[0,0],'\n',(c2[0,1]/(71-xcont[1]))*100)
print('Classificação:',c3[0,0],'\n',(c3[0,1]/(48-xcont[2]))*100)
print(((c1[0,1]+c2[0,1]+c3[0,1])/(178-xcont[2]-xcont[1]-xcont[0]))*100)
# """


