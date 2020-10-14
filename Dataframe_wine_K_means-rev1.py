# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""

from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
import collections
import random
import numpy as np
import math
from sklearn import metrics
import pandas as pd
 
#Carrega o iris dataset em iris 
wine= load_wine() 
label=wine.target
dado_randomicos=[]
x_treino=[]
y_treino=[]
x_test=[]
y_test=[]
# armazenar o predict
d=[]

lista_comparacao=[ x for x in range(len(wine.data))]
percentual=0.90



while 1:
    aleatorio=random.randrange(0,len(wine.target))
    if aleatorio not in dado_randomicos:
        dado_randomicos.append(aleatorio)
    if math.trunc(len(wine.target)*percentual) ==len(dado_randomicos):
        break
    
    
for n in dado_randomicos:
    x_treino.append(wine.data[n])
    y_treino.append(wine.target[n])

    if n in lista_comparacao:
        lista_comparacao.remove(n)   



for n in lista_comparacao:
    x_test.append(wine.data[n])
    y_test.append(wine.target[n])
    
    
print(x_test)

x_test=np.array(x_test)
y_test=np.array(y_test)
      
print(x_test)   
x_treino=np.array(x_treino)    
xcont=collections.Counter(y_treino)
y_treino=np.array(y_treino)


#Implementa o Algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_treino)
a=kmeans.labels_
b=kmeans.cluster_centers_
#Prevendo novos valores

ncluster=kmeans.predict(x_test)




c=collections.Counter(ncluster[:59-xcont[0]])
print(c)
c1=np.array(c.most_common())
c=collections.Counter(ncluster[59-xcont[0]:130-xcont[0]-xcont[1]])
print(c)
c2=np.array(c.most_common())
c=collections.Counter(ncluster[130-xcont[1]-xcont[0]:178-xcont[2]-xcont[1]-xcont[0]])
print(c)
c3=np.array(c.most_common())
print('Classificação: ',c1[0,0],'\n',(c1[0,1]/(59-xcont[0]))*100)
print('Classificação:',c2[0,0],'\n',(c2[0,1]/(71-xcont[1]))*100)
print('Classificação:',c3[0,0],'\n',(c3[0,1]/(48-xcont[2]))*100)
print(((c1[0,1]+c2[0,1]+c3[0,1])/(178-xcont[2]-xcont[1]-xcont[0]))*100)

matriz=pd.crosstab(y_test,ncluster, rownames=['Real'], colnames=['   Predito'], margins=True)
matriz=pd.DataFrame(matriz)
print(matriz.head(3))
# contador = 0
# erro = 0
# for c in range(len(y_test)):
#     if y_test[c]==d[c]:
#         contador=contador+1
#     else:
#         erro=erro+1
# print((contador/len(y_test))*100,'%','\ntarget.data :\n',y_test,'\n\npredicao :\n',d.transpose(),'\n\n Dados treino: ',y_treino)
# 
# x = targets.reshape(targets.shape[0],-1)