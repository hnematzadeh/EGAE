

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from IPython.display import display # Allows the use of display() for DataFrames
from time import time
import matplotlib.pyplot as plt
import seaborn as sns # Plotting library
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array,array_to_img
from keras.utils import np_utils
from sklearn.datasets import load_files   
from tqdm import tqdm
from collections import Counter
from sklearn.utils import resample, shuffle
from tensorflow.keras.applications.vgg16 import VGG16
import time



#================== get the class indices


# Class name to the index
#class_2_indices = train_generator.class_indices
class_2_indices = {'melanoma': 0, 'nevus': 1, 'seborrheic_keratoses': 2}
print("Class to index:", class_2_indices)

# Reverse dict with the class index to the class name
indices_2_class = {v: k for k, v in class_2_indices.items()}
print("Index to class:", indices_2_class)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from sklearn.utils import shuffle
import h5py
import tensorflow.keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


##############################PRE TRAINED MODEL RESNET##################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
# from keras_tqdm import TQDMNotebookCallback

base_model = ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='elu')(x)
x = Dropout(0.95)(x)
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = True
    
  
from tensorflow.keras.optimizers import Adam

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
             metrics=['accuracy'])






# load the weights that yielded the best validation accuracy
model.load_weights('aug_model.weights.best.hdf5')
# score = model.evaluate_generator(test_generator, steps=num_test//1, verbose=1)
yhat=model.predict(X1_test)
# print('\n', 'Test accuracy:', score[1])
################################ Genetic  Algorithm Approach############################
from skimage.segmentation import mark_boundaries, slic, quickshift, watershed, felzenszwalb    

# Figure is the ith test sample  
Figure=100
imageLabel=y1_test[Figure]  
image=X1_test[Figure]

class_to_explain=np.where(y1_test[Figure]==1)
#====fidelity check
# class_to_explain=np.where(y1_test[Figure]==0)



# image=cipher_image




tt=[]

picture=np.expand_dims(image, axis=0)
YHAT=model.predict(picture)[0][class_to_explain]
### PROBLEM DEFINITION###############
MaxIt=151;
nPop=3;
pc=0.9;
nc=2*round(pc*nPop/2)

pm=0.4;
nm=round(pm*nPop)


alpha1=0.7
beta1=0.3

NFE=0
gama=0.1
Tag=0

TagCheck=5


# kk=15
# mm=50

import copy
# perturbation argument should be array
def perturb_image(img, perturbation, segments,iter):
    global NFE
    active_pixels = np.where(perturbation == 1)[0]
    mask=np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image*mask[:,:,np.newaxis]
    pic=np.expand_dims(perturbed_image, axis=0)
    pred = model.predict(pic)[0][class_to_explain]
    ### fidelity check
    ##for label 1
    # pred = model.predict(pic)[0][class_to_explain][0]
    ### for label 2
    # pred = model.predict(pic)[0][class_to_explain][1]


    max_value=max(model.predict(pic)[0])
    k=model.predict(pic)[0]
    u=k.tolist()
    uu=u.index(max_value)
    Super=perturbation.sum(0)        
    ### fidelity check

    # for label 1 and non-fidelity

    if uu == class_to_explain[0][0]:
        ac=1
        
    else:
        ac=0
    

    # # for label 2    
    # if uu == class_to_explain[0][1]:
    #     ac=1
        
    # else:
    #     ac=0
        
    
    
    
    # fit=(alpha*pred[0])+(beta*((nVar-Super+1)/nVar))+(ceta*np.dot(perturbation,Weights))
    ####fidelity check
    # fit=(alpha*pred)+(beta*((nVar-Super+1)/nVar))+(ceta*np.dot(perturbation,Weights))
    
    
    if Super == 0 :
      fit = (alpha1*pred[0])
    
    if Super > 0:
      fit=(alpha1*pred[0])+(beta1*((nVar-Super+1)/nVar))
    ###fidelity check
    # if Super == 0 :
    #   fit = (alpha1*pred)
    
    # if Super > 0:
    #   fit=(alpha1*pred)+(beta1*((nVar-Super+1)/nVar))

    NFE=NFE+1  
    return pred[0],pic,ac,fit 

    ###fidelity check
    # return pred,pic,ac,fit 





def SinglePointCrossover(x1,x2):
    import random
    import numpy as np
    nVar=len(x1)
    C=random.randint(1,nVar-1)
    y1=(x1[0:C]).tolist() + (x2[C:]).tolist()
    y2=(x2[0:C]).tolist() + (x1[C:]).tolist()
    return y1,y2

def Mutate(x):
    import random
    import numpy as np
    nVar=len(x)
    J=random.randint(0,nVar-1)
    y=copy.deepcopy(x)
    y[J]=1-x[J]
    return y,sum(y)



def Mutate2(x):
    import random
    import numpy as np
    nVar=len(x)
    J=random.randint(0,nVar-1)
    J1=random.randint(0,nVar-1)
    J2=random.randint(0,nVar-1)
    
    y=copy.deepcopy(x)
    y[J]=1-x[J]
    y[J1]=1-x[J1]
    y[J2]=1-x[J2]
    return y,sum(y)



def RouletteWheelSelection(P):
    r=random.uniform(0,1)
    c=np.cumsum(P)
    i=np.where(r<np.array(c))[0][0]
    return i
    
 
# sp=1.5   #selection pressure
import math
#####INITIALIZATION##################
from ypstruct import struct
limit=6
phi=0.5
lst = [None] * limit
start=time.time()

##### heuristically find the best picture when segments=5


superpixels= slic(image, n_segments=5)   
nVar=np.unique(superpixels).shape[0]
el=0
TEMP=0
lst= [None] * limit
heuristic=struct(position=None, yhat=None, pictu=None, NuSuperpixels=0, acc=None, fit=0)
n = nVar
t=[None]*2**n

for i in range(2**n):
    t[i]=[str(x) for x in bin(i)[2:].zfill(n)]
    t[i]=(np.array(list(t[i]), dtype=int))
    
   
for i in range (2**n):
        active_pixels = np.where(t[i] == 1)[0]    
        mask=np.zeros(superpixels.shape)
        for active in active_pixels:
                mask[superpixels == active] = 1
        perturbed_image = copy.deepcopy(image)
        perturbed_image = perturbed_image*mask[:,:,np.newaxis]
        pic=np.expand_dims(perturbed_image, axis=0)

        heuristic.yhat,heuristic.pictu, heuristic.acc, heuristic.fit = perturb_image(image,t[i],superpixels,0)
        # heuristic.NuSuperpixels=len(active_pixels)  


        max_value=max(model.predict(pic)[0])
        k=model.predict(pic)[0]
        u=k.tolist()
        uu=u.index(max_value)      
        # fidelity label=2
        # if  heuristic.fit > TEMP  and uu == class_to_explain[0][1]:
        # fidelity label=1 and nonfidelity    
        if  heuristic.fit > TEMP  and uu == class_to_explain[0][0]:

             # sig=1 
             lst[el]=perturbed_image
             TEMP=heuristic.fit
             # heuristic.NuSuperpixels=len(active_pixels)
             heuristic.NuSuperpixels=t[i].sum(0)
             print(t[i])
             print(TEMP)
             

# sig=0
w1=heuristic.NuSuperpixels
w2=nVar
phi=0.5
if (w1/w2) <= 0.5:
    phi=0.5
if 0.5 < (w1/w2) < 1:
    phi=w1/w2  
    
if w1/w2 == 1:
    phi=0.9

# if sig==0:
#     el=-1
el=-1

si=[0]*limit
si[0]=10
si[1]=15

si[2]=20
si[3]=25
si[4]=100

ur=-1
counter=0
for co in range(1,limit):
     # sp is the selection pressure
     ur=ur+1
     segments=si[ur]
     if segments == 10 :
           sp=30
           nPop=5
           nc=2*round(pc*nPop/2)
           nm=round(pm*nPop)
           TagCheck=10
     if segments == 15 :
           sp=30
           nPop=10
           nc=2*round(pc*nPop/2)
           nm=round(pm*nPop)
           TagCheck=10
          
     if segments == 20 :
           sp=18
           nPop=15
           nc=2*round(pc*nPop/2)
           nm=round(pm*nPop)
           TagCheck=10    
          
     if segments == 25 :
            sp=18
            nPop=15
            nc=2*round(pc*nPop/2)
            nm=round(pm*nPop)
            TagCheck=10           
   
     if segments == 100 :
           sp=0.2
           phi=0.9 
           nPop=35
           nc=2*round(pc*nPop/2)
           nm=round(pm*nPop)
           TagCheck=10    

     

     superpixels= slic(image, n_segments=segments)   
     


     nVar=np.unique(superpixels).shape[0]
     BestPosition=nVar
     Weights=np.array([1/nVar]*nVar )
     Best=np.zeros(nVar)
     Order=[]*nVar

     from ypstruct import struct
     empty_individual=struct(position=None, yhat=None, pictu=None, NuSuperpixels=None, acc=None, fit=None)
     pop=empty_individual.repeat(nPop)

     Fits=np.zeros(nPop)
     it=0
     for i in range (nPop):
       pop[i].position=np.random.binomial(1,phi,size=(1,nVar))[0]
       pop[i].yhat,pop[i].pictu,pop[i].acc,pop[i].fit= perturb_image(image,pop[i].position,superpixels,it)
       pop[i].NuSuperpixels=pop[i].position.sum(0)
   
    
       Fits[i]=pop[i].fit
     Fits=np.sort(Fits)[::-1]
     P=np.zeros(nPop)
     WorstFit=pop[nPop-1].fit
     for j in range (nPop):
          P[j]=math.exp(-sp*(1/Fits[j])/(1/WorstFit))
     P=P/sum(P)
     z=0
     PP=sorted(P, reverse=True)
     for i in range (int(nPop/2)):
         z=z+PP[i]
         
     print(z)
##### Sort population
    
     import operator
     pop=sorted(pop,key=operator.attrgetter('fit'), reverse=True)
    
     for i in range (nPop):
         print(pop[i].position, "  ",pop[i].NuSuperpixels, " ",pop[i].yhat, " ",pop[i].acc, " ",pop[i].fit)
    
    
### store best solutions in each iteration
     BestSol=pop[0]
     BestFits=np.zeros(MaxIt)
     BestFits[it]=BestSol.fit

##store worst fit
     WorstFit=pop[nPop-1].fit


### array to hold best values in all iterations

     BestYhat=np.zeros(MaxIt)   

#### array to hold NFEs
     nfe=np.zeros(MaxIt)   
    
### Main Loop
     import random
     import math
# for it in range (MaxIt):
     it=it+1
     Tag=0
     
     while (it<MaxIt and  Tag!=TagCheck):
        
    
         popc1=empty_individual.repeat(int(nc/2))
         popc2=empty_individual.repeat(int(nc/2))
         Xover=list(zip(popc1,popc2))
         for k in range (int(nc/2)):
             
             
             # Select First Parent
             i1=RouletteWheelSelection(P)
             # i1=random.randint(0,nPop-1)
             p1=pop[i1].position
             # Select Second Parent
             i2=RouletteWheelSelection(P)
             # i2=random.randint(0,nPop-1)
             p2=pop[i2].position
             #Apply Crossover
             Xover[k][0].position,Xover[k][1].position=np.array(SinglePointCrossover(p1,p2))
             #Evaluate Offspring
             Xover[k][0].yhat,Xover[k][0].pictu,Xover[k][0].acc,Xover[k][0].fit=perturb_image(image,Xover[k][0].position,superpixels,it)
             Xover[k][0].NuSuperpixels=Xover[k][0].position.sum(0)
             
             Xover[k][1].yhat,Xover[k][1].pictu,Xover[k][1].acc, Xover[k][1].fit=perturb_image(image,Xover[k][1].position,superpixels,it)
             Xover[k][1].NuSuperpixels=Xover[k][1].position.sum(0)
             popc=empty_individual.repeat(nc)
             i=0
             for s in range (len(Xover)):
                 for j in range(2):
                     popc[i]=Xover[s][j]
                     i=i+1
    # Mutation
         popm=empty_individual.repeat(nm)    
         for k in range(nm):
       # Select Parent
             i=random.randint(0,nPop-1)
             p=pop[i].position
             
       
             if segments>=30:
       # Apply mutation
       
                popm[k].position,popm[k].NuSuperpixels=Mutate2(p)
       # Evaluate mutatnt
                popm[k].yhat,popm[k].pictu,popm[k].acc,popm[k].fit=perturb_image(image,popm[k].position,superpixels,it)
  # Apply mutation
             if segments < 30:

               popm[k].position,popm[k].NuSuperpixels=Mutate(p)
  # Evaluate mutatnt
             
               popm[k].yhat,popm[k].pictu,popm[k].acc,popm[k].fit=perturb_image(image,popm[k].position,superpixels,it)

# Distructor
   
    
   
     # Distructor
    
             
             
     # merge population        
         pop= pop+popc+popm    
         pop=sorted(pop,key=operator.attrgetter('fit'), reverse=True)
    
      #truncate
         pop=pop[0:nPop]
    
    # d=random.randint(1,MaxIt)
    # if (d<=it):
    #    actives=np.where(np.array(pop[0].position)==1)[0]
    #    for active in actives:
    #      if (random.randint(0,1)) == 1: 
    #          pop[0].position[active] = 0
             
             
    
    # Update WorstFit
         WorstFit=min(pop[nPop-1].fit,WorstFit)
    
    # Calculate selection probabilities
         Fits=np.zeros(nPop)
         for jj in range (nPop):
             Fits[jj]=pop[jj].fit
         Fits=np.sort(Fits)[::-1]
         P=np.zeros(nPop)
         for j in range (nPop):
              P[j]=math.exp(-sp*(1/Fits[j])/(1/WorstFit))

         P=P/sum(P)
    # store best solution ever found
         BestSol=pop[0]
         BestFits[it]=BestSol.fit
         if BestSol.position.sum() < BestPosition:
           BestFig=BestSol.position
           BestPosition=BestSol.position.sum() 

    
    ### store NFE
    
         nfe[it]=NFE
    
    
         if (Best.tolist()==pop[0].position.tolist()):
            Tag=Tag+1
         else:
            Tag=1
         if (Best.tolist()!=pop[0].position.tolist()):
               Best=pop[0].position
         
         print("Iteration ", str(it) ,": Best fit = ", BestSol.fit, "Best Yhat =  ", BestSol.yhat, "Best Solution =  ", BestSol.position, "NFE  ", nfe[it])

         it=it+1   
       
             
         
             
         
     active_pixels = np.where(BestSol.position == 1)[0]
     mask=np.zeros(superpixels.shape)
     for active in active_pixels:
           mask[superpixels == active] = 1
     perturbed_image = copy.deepcopy(image)
     perturbed_image = perturbed_image*mask[:,:,np.newaxis]
     pic=np.expand_dims(perturbed_image, axis=0)

     
     
     pred = model.predict(pic)[0][class_to_explain]
 ## fidelity check
     # for label 1 and non-fidelity
     # pred = model.predict(pic)[0][class_to_explain][0]
     # for label 2
     # pred = model.predict(pic)[0][class_to_explain][1]
     
     
            
     max_value=max(model.predict(pic)[0])
     k=model.predict(pic)[0]
     u=k.tolist()
     uu=u.index(max_value)
     
     ### for label 1 and non-fidelity
     if uu == class_to_explain[0][0]:
                 el=el+1
                 lst[el]=perturbed_image
    
     
    # fidelity for label 2
      
     # if uu == class_to_explain[0][1]:
     #            el=el+1
     #            lst[el]=perturbed_image
     tt.append(BestFits[0:it-1])
     counter=counter+1
end=time.time()
print("Time = ", end-start,  "NFE = ", NFE)




### Voting strategies############################

Flst = [None] * limit

t=[None]*2**limit
com=struct(NuFigures=-100, Nupixels=-100, position=-100)
ind3=com.repeat(2**limit)

# ind3[0]=0
n = 5
for i in range(2**limit):
    t[i]=[str(x) for x in bin(i)[2:].zfill(limit)]
    t[i]=(np.array(list(t[i]), dtype=int))
    
for i in range (2**len(lst)):
        active_images = np.where(t[i] == 1)[0]
        R2=np.zeros(shape=(224,224,3,len(active_images))) 
        k=0
        for active in active_images:
            if lst[active] is not None:
              Flst[k]=lst[active]
              k=k+1
            
        
        if k>1:
        
           for j in range (k):
              R2[:,:,:,j]=np.array(Flst[j]) 
           consensus=np.power(np.prod(R2, axis=3),1/(R2.shape[3]))
           if np.sum(consensus)!=0 and math.isnan(np.sum(consensus))==False:
             ind3[i].NuFigures=int(np.sum(t[i]))
             ind3[i].Nupixels=int(np.sum(consensus))
             ind3[i].position=i
           else:
             ind3[i].NuFigures=0
             ind3[i].Nupixels=0
             ind3[i].position=i
        else:
             ind3[i].NuFigures=0
             ind3[i].Nupixels=0
             ind3[i].position=i
ind3=sorted(ind3,key=operator.attrgetter('NuFigures'), reverse=True)
F=ind3[0].NuFigures
TEMP=0
for i in range (2**limit):
  if ind3[i].NuFigures==F  and ind3[i].Nupixels>TEMP :
      TEMP=ind3[i].Nupixels
for i in range (2**limit):
    if ind3[i].Nupixels==TEMP and ind3[i].NuFigures==F :
       P=ind3[i].position 
       
       
rr=bin(P)[2:].zfill(limit)
rr=(np.array(list(rr), dtype=int))

FFlst=[None]* np.sum(rr)
active_images = np.where(rr == 1)[0]
R2=np.zeros(shape=(224,224,3,len(active_images))) 
k=0
for active in active_images:
      FFlst[k]=lst[active]
      k=k+1
      
#####CONSENSUS      
for j in range (k):
   R2[:,:,:,j]=np.array(FFlst[j]) 
consensus=np.power(np.prod(R2, axis=3),1/(R2.shape[3]))
plt.imshow(consensus)
plt.axis('off')
#####Majority



l=math.trunc((k/2))+1
mask=np.where(R2==0,False,True)
mask=np.sum(mask,axis=3)
mask=np.where(mask>=l, True,False)
# mask=mask.astype(np.int16)
cropped_image=np.multiply(image,mask)

plt.imshow(cropped_image)
plt.axis('off')



print("number of images contribute in final illustrations = ",   k)



##############Accuracy against ground truth###################################


#### samples for  94
c1=consensus
m1=cropped_image

# c2=consensus
# m2=cropped_image

# c3=consensus
# m3=cropped_image



# c11=consensus
# m11=cropped_image

# c22=consensus
# m22=cropped_image

# c33 = consensus
# m33 = cropped_image
#############################



initial=i5.flatten()
second=m1.flatten()


distance =0
for i in range(len(second)):
        # distance += np.square(input_image[i]-con[i])
        # distance += np.square(input_image[i]-maj[i])
        distance += np.square(second[i]-initial[i])
        
np.sqrt(distance)






###  plotting performance graph
plt.plot(tt[0])
plt.plot(tt[1])
plt.plot(tt[2])
plt.plot(tt[3])
plt.plot(tt[4])
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("Image 6")
# plt.legend(['\u03C6 = 0.9' ,'\u03C6 = 0.5'], loc = "lower right")
plt.legend(['10 Superpixels' ,'15 Superpixels', '20 Superpixels', '25 Superpixels', '100 Superpixels'], loc = "lower right")


# ff=[]
# ff=tt[0]


### plotting the effect of sparsity 

plt.plot(BestFits[0:it-1])
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.title("Image 0")
plt.show()










########LIME


import lime
from lime import lime_image


explainer=lime_image.LimeImageExplainer()
explanation=explainer.explain_instance(X1_test[100],model.predict, num_samples=2000,segmentation_fn=slic)
image,mask=explanation.get_image_and_mask(model.predict(X1_test[100].reshape((1,224,224,3))).argmax(axis=1)[0], negative_only=False, positive_only=True, num_features=10)

plt.imshow(mark_boundaries((image),mask))
plt.axis('off')




















  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  



























 
    
 
    
 
    
 
    
 
    
 
    
