
# coding: utf-8

# In[1]:


#Gradient Descent for finding relationships between MPG and related stats in old cars 

#Dependencies
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Get Data: (n,1) ndarray
mpg = np.genfromtxt('mpg.txt',usecols = (0))
acc = np.genfromtxt('mpg.txt',usecols = (5))

# print(mpg.shape)
# print(acc.shape)


# # In[2]:


# #Error Math
# def error(w0, w1, x, y):
	# er = 0
	# for i in range(len(mpg)):
		# er += abs(y[i] - (w0 + w1 * x[i]))
	# return er/len(mpg)




# # In[ ]:


# # np.random.seed(0)
# # sw0 = 0.0
# # while abs(sw0) < 200:
    # # sw0 = 300 * (2 * np.random.random((1,)) - 1)
# # w0 = np.asscalar(sw0)

# # sw1 = 0.0
# # while abs(sw1) < 200:
    # # sw1 = 300 * (2 * np.random.random((1,)) - 1)
# # w1 = np.asscalar(sw1)

# y = mpg

# starting_error = error(w0,w1, acc, y)



# # In[ ]:


# #Gradient Descent
# lr = .01
# m = len(mpg)



# for i in range(100):
    # #if (i == 1) or (i % 50 == 0):
    # #print('Error:')
    # #print(error(w0,w1, acc, y))
    # #print('\nWeights:')
    # #print(w0, w1)
    # #print(w0.shape)
    # for j in range(len(acc)):
        # if (j in (1,2,3,4,5,6,7,8,9,10)):
            # print('Error:')
            # print(error(w0,w1, acc, y))
            # print('Weights:')
            # print(w0, w1)
        # nc_w0 = w0
        # nc_w1 = w1
        # w0 += (m**-1) * (nc_w0 - lr * (-1 * ( y[j] - (nc_w1 * acc[j] + nc_w0))))
        # w1 += (m**-1) * (nc_w1 - lr * (-acc[j] * ( y[j] - (nc_w1 * acc[j] + nc_w0))))


# # In[ ]:


# print('Final Values')
# print('Starting Error:')
# print(starting_error)
# print('Error after Gradient Descent:')
# print(error(w0,w1, acc, y))

# print('\nStarting Weights:')
# print(sw0, sw1)
# print('Weights after Gradient Descent:')
# print(w0, w1)


# # In[ ]:


# #Output

# xplt = acc
# yplt = w0 + xplt *w1

# plt.scatter(xplt, y)
# plt.plot(xplt, yplt)
# plt.show()


# # In[61]:


# w0 = np.asscalar(sw0)
# w1 = np.asscalar(sw1)
# print(w0)
# w0 += (m**-1) * (w0 - .001 * (-1 * ( y[0] - (w1 * acc[0] + w0))))
# print(w0)
# print((m**-1) * (w0 - .001 * (-1 * ( y[0] - (w1 * acc[0] + w0)))))
# print(w1, y[0])


# In[ ]:

from scikit-learn import linear_model

regr = linear_model.LinearRegression()
regr.fit(mpg, acc)
print(regr.coef_)

plt.scatter(mpg, acc)
plt.plot(mpg, regr.predict(mpg))


