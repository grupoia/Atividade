# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:47:14 2020

@author: https://www.codigofluente.com.br/aula-04-instalando-o-pandas/
"""
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn import metrics
from numpy.random import default_rng




#Carrega o iris dataset em iris
iris = load_iris()

X = iris.data
y = iris.target

#variaveis
dado_randomicos=[]
x_treino=[]
y_treino=[]
x_test=[]
y_test=[]
# armazenar o predict


lista_comparacao=[ x for x in range(len(iris.data))]
percentual=0.80
#GERANDO NUMEROS RANDOMICOS NÃO REPETIDOS, PARA COLETAR OS DADOS DE FORMA MAIS IMPARCIAL POSSIVEL
while 1:
    aleatorio=random.randrange(0,len(iris.target))
    if aleatorio not in dado_randomicos:
        dado_randomicos.append(aleatorio)
    if round(len(iris.target)*percentual)==len(dado_randomicos):
        break

#ADICIONANDO DADOS NAS LISTAS
for n in dado_randomicos:
    x_treino.append(iris.data[n])
    y_treino.append(iris.target[n])

    if n in  lista_comparacao:
        lista_comparacao.remove(n)

for n in lista_comparacao:
    x_test.append(iris.data[n])
    y_test.append(iris.target[n])

x_test=np.array(x_test)
y_test=np.array(y_test)
x_treino=np.array(x_treino)
y_treino=np.array(y_treino)


#Implementa o Algoritmo KNN
neigh = KNeighborsClassifier(n_neighbors=3,weights="uniform")
neigh.fit(x_treino, y_treino)
#Prevendo novos valores

d=neigh.predict(x_test)

erro=0
contador=0

#CONTADOR PARA AVALIAR GRAU DE ACERTO
for c in range(len(y_test)):
    if y_test[c]==d[c]:
        contador=contador+1
    else:
        erro=erro+1

#porcentagens
print('porcentagem total:',(contador/len(y_test))*100,'%','\ntarget.data :\n',y_test,'\n\npredicao :\n',d.transpose(),'\n\n Dados treino: ',y_treino)

#dados preditos e acertados 
print (pd.crosstab(y_test,d, rownames=['Real'], colnames=['          Predito'], margins=True))









