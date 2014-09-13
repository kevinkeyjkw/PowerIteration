from numpy import *
import pdb

f = open('magic04.data.txt','r')
rows = []
x = f.readline()
numAtt = len(x.split(','))-1#We don't count the last one for now
while(x != ''):
	temp = x[:len(x)-3].split(',')
	col = [float(i) for i in temp]
	rows.append(col)
	x = f.readline()
numRows = len(rows)
print(rows[0],'\n',rows[1])

matrix = array(rows)

#1compute mean vector
meanVector = matrix.sum(axis=0)/numRows
print('Mean Vector: ',meanVector)
#2computer sample covariance matrix for multivariables

centeredMat = rows - meanVector
centeredMatTrans = centeredMat.transpose()
covarianceMat = dot(centeredMatTrans,centeredMat)/numRows
print('Sample covariance matrix',covarianceMat)
#numCov = cov(rows,None,0,1,None)
#print('Numpy\'s covariance matrix: ',numCov)
#3compute attribute with max and min variance

temp = centeredMat**2
temp = temp.sum(axis=0)
temp = temp/numRows
minVarAttIndex = temp.argmin(0)
maxVarAttIndex = temp.argmax(0)
#4computer pair of attributes with largest and smallest covariance respectively
#Just find max and min of covariance matrix
#compare each pair
# x1 = 0
# x2 = 1
# n = len(centeredMat)
# maxVarPair,minVarPair=0,999
# 
# while(x1 < numAtt - 1):
# 	c1=(centeredMat.transpose())[x1]
# 	c2=(centeredMat.transpose())[x2]
# 	temp =  ((c1 * c2).sum())/n - (c1.sum()*c2.sum())/(n**2)
# 	if temp < minVarPair: minVarPair = temp
# 	if temp > maxVarPair: maxVarPair = temp
# 	x2 += 1
# 	if(x2 == numAtt):
# 		x1 += 1
# 		x2 = x1 + 1
# print('Max variance: ',maxVarPair,'. Min variance: ',minVarPair)

maxCo = covarianceMat.max()
minCo = covarianceMat.min()
coVarArray = array(covarianceMat)
maxCoIndex = where(coVarArray == maxCo)
minCoIndex = where(coVarArray == minCo)
#4 compute dominant eigenvector and eigenvalue
# n = len(covarianceMat)
# p1 = ones(n)
# p2 = dot(covarianceMat,p1)
# i = argmax(p2)
# j = argmax(p1)
# ev = p2[i]/p1[j]
# p2 = p2/p2[i]
# while(linalg.norm(p2-p1) > 0.0001):
# 	p1 = p2
# 	p2 = dot(covarianceMat,p1)
# 	i = argmax(p2)
# 	j = argmax(p1)
# 	ev = p2[i]/p1[j]
# 	p2 = p2/p2[i]#This is scaling vector, not normalizing
# print("p2 before normalized:",p2)
# p2 = p2/linalg.norm(p2)
# --
pdb.set_trace()
a,b = linalg.eig(covarianceMat)
vec = b[absolute(a).argmax()]

print("NP's Eigenvalue: ",a[absolute(a).argmax()]," Eigenvector: ",vec)
n = len(covarianceMat)
p1 = ones(n)
p2=[]
ev = 0
while(True):
	p2 = dot(covarianceMat,p1)
	#p3 = p2
	ev = linalg.norm(p2)
	p2 = p2/ev#This is scaling vector, not normalizing
	if(linalg.norm(p2-p1) <= 0.0001): 
		break
	else:
		p1 = p2
#ev = dot(dot(covarianceMat,p2),p2)/dot(p2,p2)
# p2 = p2/linalg.norm(p2)
print("Eigenvalue: ",ev," Eigenvector: ",p2*(-1))

#print("Computed eigenvalue is: ",dot(dot(covarianceMat,vec),vec)/dot(vec,vec))


