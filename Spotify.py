# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 02:31:22 2019

@author: 15083
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

spotify_data = pd.read_csv("Spotifydata.csv")
##to  create histogram using numpy
hist1,edges1=np.histogram(spotify_data.popularity)
plt.bar(edges1[:-1],hist1,width=edges1[:1]-edges1[:-1])
plt.figure()

print(spotify_data.corr())
##scatter plot 
plt.scatter(spotify_data.popularity,spotify_data.danceability)

### Regression analysis
y=spotify_data.popularity
x=spotify_data.danceability
## In order to find the constant value in linear regression equation Y= Coeff*x+constant
## y is popularity
## x is danceability
x=sm.add_constant(x)
# To find out relation between popularity and danceability in order to predict popularity
##OLS ordinary least square method
lr_model= sm.OLS(y,x).fit()
print(lr_model.summary())

x_prime = np.linspace(x.danceability.min(),x.danceability.max(),100)
x_prime = sm.add_constant(x_prime)
y_hat = lr_model.predict(x_prime)
plt.scatter(x.danceability,y)
plt.xlabel("danceability")
plt.ylabel("popularity")
plt.plot(x_prime[:,1],y_hat)




