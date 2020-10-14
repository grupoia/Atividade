# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:29:31 2020

@author: marqu
"""

from sklearn.cluster import Birch
from sklearn.datasets import load_wine
import collections
import random
import math
import numpy as np
 
wine=load_wine() 
label=wine.target
data=wine.data

"========Amostra de dados aleatórios provenientes do data set============="
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
    
"Porcentagem dos dados que sobraram para teste de acertos"

x_test=np.array(x_test)
y_test=np.array(y_test)  
x_treino=np.array(x_treino)    
xcont=collections.Counter(y_treino)



# treino e predição dos dados para 3 clusters

brc = Birch(n_clusters=3)
brc.fit(x_treino)
Newcluster_membership=brc.predict(x_test)


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


