#!/usr/bin/env python
# coding: utf-8

# # HW4: implementing item-based CF with cosine
# First, run recommenderDemo.ipynb and be familar with the code and data.
# Second, implement item-based CF with cosine

# In[114]:


import gzip
from collections import defaultdict
import scipy
import scipy.optimize
import numpy as np
import random
import pandas as pd
import sys


# 1. load the data, and convert integer-valued fields as we go. Note that here we use the same "Musical Instruments" dataset. Download the date from here: https://web.cs.wpi.edu/~kmlee/cs547/amazon_reviews_us_Musical_Instruments_v1_00_small.tsv.gz
# The dataset contains 20K user-item reviews.

# In[115]:


# From https://web.cs.wpi.edu/~kmlee/cs547/amazon_reviews_us_Musical_Instruments_v1_00_small.tsv.gz
#----------------------------------------------
# Your code starts here
#   Please add comments or text cells in between to explain the general idea of each block of the code.
#   Please feel free to add more cells below this cell if necessary

path = "./amazon_reviews_us_Musical_Instruments_v1_00_small.tsv.gz"
file = gzip.open(path, 'rt', encoding="utf8")


# In[116]:


#----------------------------------------------
# Your code starts here
#   Please add comments or text cells in between to explain the general idea of each block of the code.
#   Please feel free to add more cells below this cell if necessary
header = file.readline()
header = header.strip().split('\t')


# In[117]:


print(header)


# 2. now store the loaded data into a matrix -- you may use numpy array/matrix to store the untility matrix or use sparse matrix (advanced approach)

# In[118]:


my_dataset =[]

for single_line in file :
    fields = single_line.strip().split('\t')
    d = dict(zip(header, fields))
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    d['customer_id']=str(d['customer_id'])
    d['product_id']=str(d['product_id'])
    my_dataset.append(d)

len(my_dataset)


# In[119]:


#----------------------------------------------
# Your code starts here
#   Please add comments or text cells in between to explain the general idea of each block of the code.
#   Please feel free to add more cells below this cell if necessary

my_dataframe = pd.DataFrame(my_dataset)# Storing in dataframe
my_dataframe.head()


# In[120]:


print(np.unique(my_dataframe['star_rating']))


# In[121]:


matrix = my_dataframe.pivot_table(index='product_id', columns='customer_id', values='star_rating').fillna(0)


# In[122]:


matrix.loc['B003LRN53I','14640079']


# In[123]:


matrix.loc['B0006VMBHI','6111003']



# In[124]:


matrix.shape


# In[125]:


sys.getsizeof(matrix)/1000000


# In[126]:


usersPerSingleItem = defaultdict(set)

reviewsPerSingleUser = defaultdict(list)
reviewsPerSingleItem = defaultdict(list)

for iterator in my_dataset:
    user,item = iterator['customer_id'], iterator['product_id']
    reviewsPerSingleUser[user].append(iterator)
    reviewsPerSingleItem[item].append(iterator)


# 3. Implement cosine function and rating prediction function by using the cosine function. If a hasn't rated any similar items before, then return ratingMean (i.e., global rating mean). Refer to predictRating() in hw4jaccard.ipynb

# In[127]:


def Cosine(a,b):
    a=np.array(a)
    b=np.array(b)
    
    for itr in range(len(a)):
        if a[itr]!=0:
            a[itr]=a[itr]-np.mean(a)
        if b[itr]!=0:
              b[itr]=b[itr]-np.mean(b)
              
    return np.round(np.dot(a,b)/(np.sqrt((a*a).sum())*np.sqrt((b*b).sum())),4)

ratingMean = sum([d['star_rating'] for d in my_dataset]) / len(my_dataset)

def predictRatingCosine(user,item):

    ratings = []
    similarities = []
    for d in reviewsPerSingleUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        ratings.append(matrix.loc[i2,user])
        similarities.append(Cosine(matrix.loc[i2].values,matrix.loc[item].values))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean


# In[128]:


labels = [d['star_rating'] for d in my_dataset]


# 4. Measure and report MSE (don't need to change the below code)

# In[130]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

cfPredictions = [predictRatingCosine(d['customer_id'], d['product_id']) for d in my_dataset]
print(MSE(cfPredictions, labels))


# In[112]:


MSE(alwaysPredictMean, labels)


# In[131]:


MSE(cfPredictions, labels)


# (optional/bonus task: you will get additional 25 points) 
# download https://web.cs.wpi.edu/~kmlee/cs547/amazon_reviews_us_Musical_Instruments_v1_00_large.tsv.gz
# this dataset contains over 900K user-item reviews. repeat the above process (i.e., meauring MSE with cosine). report the MSE and compare it with MSE of alwaysPredictMean. This optional task would require better data structure and implementation.

# In[110]:


#----------------------------------------------
# Your code starts here
#   Please add comments or text cells in between to explain the general idea of each block of the code.
#   Please feel free to add more cells below this cell if necessary

path = "./amazon_reviews_us_Musical_Instruments_v1_00_large.tsv.gz"
file2 = gzip.open(path, 'rt', encoding="utf8")

header = file2.readline()
header = header.strip().split('\t')


# In[111]:


large_dataset = []

for single_line in file2:
    fields = single_line.strip().split('\t')
    d = dict(zip(header, fields))
    d['star_rating'] = int(d['star_rating'])
    d['helpful_votes'] = int(d['helpful_votes'])
    d['total_votes'] = int(d['total_votes'])
    d['customer_id']=str(d['customer_id'])
    d['product_id']=str(d['product_id'])
    large_dataset.append(d)
    
df = pd.DataFrame(large_dataset)
df.head()


# In[ ]:


matrix= sparse.coo_matrix((data, (row, col)), shape=(4, 4))


# In[ ]:


reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for d in dataset:
    user,item = d['customer_id'], d['product_id']
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)
    
def Cosine(a,b):
    a=np.array(a)
    b=np.array(b)
    for itr in range(len(a)):
        if a[itr]!=0:
            a[itr]=a[itr]-np.mean(a)
        if b[itr]!=0:
              b[itr]=b[itr]-np.mean(b)
              
    return np.round(np.dot(a,b)/(np.sqrt((a*a).sum())*np.sqrt((b*b).sum())),4)


ratingMean = sum([d['star_rating'] for d in large_dataset]) / len(large_dataset)

def predictRatingCosine(user,item):

    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['product_id']
        if i2 == item: continue
        ratings.append(matrix.loc[i2,user])
        similarities.append(Cosine(matrix.loc[i2].values,matrix.loc[item].values))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return sum(weightedRatings) / sum(similarities)
    else:
        return ratingMean

labels = [d['star_rating'] for d in large_dataset]

def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)

cfPredictions = [predictRatingCosine(d['customer_id'], d['product_id']) for d in large_dataset]
print(MSE(cfPredictions, labels))


# 

# *-----------------
# # Done
# 
# All set! 
# 
# ** What do you need to submit?**
# 
# * **hw4.ipynb Notebook File**: Save this Jupyter notebook with all output, and find the notebook file in your folder (for example, "filename.ipynb"). This is the file you need to submit. 
# 
# ** How to submit: **
#         Please submit through canvas.wpi.edu
# 
