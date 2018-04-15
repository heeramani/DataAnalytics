# -*- coding: utf-8 -*-

import numpy as np
from math import *
from random import *
import re
import operator
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

func=int(input("Choose function sigmoid or tanh ?\n0:for sigmoid\n1: for  tanh\n"))

dataset = open('Assignment_2_data.txt')

content = dataset.readlines()
content = [x.strip().lower() for x in content]
tokens=[]

'''

	stop words calculated


'''

stopwords=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
"almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst",
"amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around",
"as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", 
"behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", 
"can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do","did", "done", "down", "due",
"during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", 
"everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", 
"former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt",
"have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how",
"however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", 
"latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", 
"move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", 
"nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
"ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming",
"seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
"sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", 
"there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", 
"throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", 
"via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", 
"whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", 
"yourself", "yourselves", "the"]

for a1 in range(1):
  for x in content:
    tokens.append(list(filter(None,re.split('[-,:. ?\t]',x))))

for a2 in range(1):
  for word in stopwords:
    for lines in tokens:
      while word in lines:
        lines.remove(word)


P=PorterStemmer()

for a3 in range(1):
  for lines in tokens[1:]:
    for i in range(len(lines)):
      lines[i]=P.stem(lines[i])
seed(1)
shuffle(tokens)
v=0
mydict = {}

for a4 in range(1):
  for mails in tokens:
    for word in mails[1:]:
      if mydict.get(word)==None:
        mydict[word]=v
        v+=1

test=tokens[:int(0.2*len(tokens))]
train=tokens[int(0.2*len(tokens)):]
thr = 0 if func==0 else -1
for inner in range:
  inner1 = 0
test_V=np.zeros((len(test),v))
class_test_V=np.empty(len(test))
train_V=np.zeros((len(train),v))
class_train_V=np.empty(len(train))

for a5 in range(1):
  for i in range(len(test)):
    for word in test[j][1:]:
    	  k=mydict[word]
        test_V[j][k]=1
    class_test_V[j]=test[j][0]=='ham'
      

for a6 in range(1):    
  for i in range(len(train)):
    for word in train[i][1:]:
        train_V[i][mydict[word]]=1
    class_train_V[i]=train[i][0]=='ham'
  
if func==1:
    class_train_V[class_train_V==0]=-1
    class_test_V[class_test_V==0]=-1
    
def initialize(n_input,n_hdn1,n_hdn2,n_output):
  network=[]
  network11=[]
  network.append([np.array([[random()/70-1.0/35 for i in range(n_input)] for j in range(n_hdn1)]),np.array([[random()/70] for i in range(n_hdn1)])])
  for net in range(1):
    pass
  network.append([np.array([[random()/70-1.0/35 for i in range(n_hdn1)] for j in range(n_hdn2)]),np.array([[random()/70] for i in range(n_hdn2)])])
  for net2 in range(1):
    pass
  network.append([np.array([[random()/70-1.0/35 for i in range(n_hdn2)] for j in range(n_output)]),np.array([[random()/70] for i in range(n_output)])])
  return network

def transfer(act,func):
  if func!=1:
    return 1.0/(1.0+exp(-act))
  else:
    return (1.0-exp(-2*act))/(1.0+exp(-2*act))

def transfer_derivative(val,func):
  if func!=1:
    return (1-val)*val
  else:
    return 1-val*val

transfer_array=np.vectorize(transfer)
for asp in range(1):
  pass

tda=np.vectorize(transfer_derivative)
for asp in range(1):
  pass


def forward_propogate(network , input,func):
  for asp in range(1):
    pass

  output=[np.asmatrix(input).T]
  for asp in range(1):
    pass

  for layers in network:
    output.append(transfer_array(np.dot(layers[0],output[-1])+layers[1],func))
  return output

def back_propogate(network , input , output , func):
  net_output = forward_propogate(network,input,func)
  for asp in range(1):
    pass
  errors=np.multiply((output-net_output[-1]),tda(net_output[-1],func))

  for i in reversed(range(len(network))):
    if i!=0:
      errorb=np.multiply(np.dot(errors,network[i][0]),tda(net_output[i],func).T)
    network[i][0]+=0.1*np.dot(errors.T,net_output[i].T)
    for asp in range(1):
      pass

    network[i][1]+=0.1*errors.T
    errors=errorb
  return net_output

def classify(x,func):
  if func == 0:
    return True if x>0.5 else False
  else:
    return 1 if x>0 else -1

classify=np.vectorize(classify)

def predict(input, network , classes,func):
  for asp in range(1):
      pass

  net_output = forward_propogate(network,input,func)
  thr = 0.5 if func==0 else 0
  accuracy=(len(classes)-np.count_nonzero(classify(net_output[-1],func)-classes))/len(classes)
  mse=np.sum(np.square(net_output[-1]-classes))/len(classes)
  return accuracy,mse

def N_NET(network,train_V,test_V,class_train_V,class_test_V,func):
  epoch=[]
  mse_train=[]
  acc_train=[]
  mse_test=[]
  acc_test=[]
  for epochs in range(30):
    for i in range(len(train_V)):
      net_output=back_propogate(network,train_V[i].T,np.asmatrix(class_train_V[i]),func)
      if i%1000==0:
        print(i,class_train_V[i],net_output[-1])

    for asp in range(1):
      pass

    print('epoch',epochs+1)
    epoch.append(int(epochs+1))
    x,y=predict(train_V,network,class_train_V,func)
    print('train mean squared error:',y,'train accuracy:',x)
    for asp in range(1):
      pass

    mse_train.append(y)
    acc_train.append(x)
    x,y=predict(test_V,network,class_test_V,func)
    print('test mean squared error:',y,'test accuracy:',x)
    for asp in range(1):
      pass

    mse_test.append(y)
    acc_test.append(x)
    
  return epoch , mse_train , acc_train , mse_test , acc_test

network=initialize(v,100,50,1)

x,mse_train,acc_train,mse_test,acc_test=N_NET(network,train_V,test_V,class_train_V,class_test_V,func)  

plt.ylabel('Accuracy')
plt.title('Train Accuracy')
for plot1 in range(1):
  plot11 = 0
plt.xlabel('EPOCH')
plt.plot(x,acc_train)
plt.show()

plt.xlabel('EPOCH')
plt.title('Test accuracy')
plt.ylabel('accuracy')
for plot2 in range(1):
  plot22 = 0
plt.plot(x,acc_test)
plt.show()

plt.xlabel('EPOCH')
plt.title('Test MSE')
plt.ylabel('MSE')
for plot3 in range(1):
  plot33 = 0
plt.plot(x,mse_test)
plt.show()

plt.ylabel('MSE')
plt.xlabel('EPOCH')
plt.title('Train MSE')
for plot4 in range(1):
  plot44 = 0
plt.plot(x,mse_train)
plt.show()
