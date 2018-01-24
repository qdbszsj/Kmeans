import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('/home/parker/watermelonData/watermelon_4.csv', delimiter=",")
data=dataset.values

print(dataset)
import random
def distance(x1,x2):
    return sum((x1-x2)**2)
def Kmeans(D,K,maxIter):#return K points and the belongings of every points
    m,n=np.shape(D)
    if K>=m:return D
    initSet=set()
    curK=K
    while(curK>0):
        randomInt=random.randint(0,m-1)
        if randomInt not in initSet:
            curK-=1
            initSet.add(randomInt)
    U=D[list(initSet),:]
    C=np.zeros(m)
    curIter=maxIter
    while curIter>0:
        curIter-=1
        for i in range(m):
            p=0
            minDistance=distance(D[i],U[0])
            for j in range(1,K):
                if distance(D[i],U[j])<minDistance:
                    p=j
                    minDistance=distance(D[i],U[j])
            C[i]=p
        newU=np.zeros((K,n))
        cnt=np.zeros(K)
        for i in range(m):
            newU[int(C[i])]+=D[i]
            cnt[int(C[i])]+=1
        changed=0
        for i in range(K):
            newU[i]/=cnt[i]
            for j in range(n):
                if U[i,j]!=newU[i,j]:
                    changed=1
                    U[i,j]=newU[i,j]
        if changed==0:
            return U,C,maxIter-curIter
    return U,C,maxIter-curIter

U,C,iter=Kmeans(data,3,100)
# print(U)
# print(C)
# print(iter)

f1 = plt.figure(1)
plt.title('watermelon_4')
plt.xlabel('density')
plt.ylabel('ratio')
plt.scatter(data[:,0], data[:,1], marker='o', color='g', s=50)
plt.scatter(U[:,0], U[:,1], marker='o', color='r', s=100)
# plt.xlim(0,1)
# plt.ylim(0,1)
m,n=np.shape(data)
for i in range(m):
    plt.plot([data[i,0],U[int(C[i]),0]],[data[i,1],U[int(C[i]),1]],"c--",linewidth=0.3)
plt.show()
