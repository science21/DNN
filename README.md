# DNN
Microbial contaminants in groundwater used for drinking present significant risks to human health. Routine monitoring of the microbial quality of groundwater is difficult and mostly impractical, especially in resource-limited areas. Therefore, models used to accurately predict microbial contaminants in groundwater are necessary. 

In this study, we examined multiple types of variables related to microbial quality of tubewells in Matlab, Bangladesh, including tubewell characteristics, sanitation infrastructure, land use, weather, and population density. 


We then developed deep neural network (DNN) models to predict the presence and the concentration of E. coli, an indicator of fecal contaminants in shallow groundwater using Tensorflow, a new deep learning framework. The model successfully predicted the presence/absence of E. coli in water with an accuracy of 75.6%, which is much higher than the accuracy (61.2%) obtained using a logistic regression model. 

The DNN model also predicted the concentration of E. coli in water with good performance, as indicated by the significant correlation between the predicted and measured values (r=0.300, p<0.001), which is slightly better than the negative binomial regression model (r=0.278, p<0.001).  


Our findings suggest that DNN models perform better than traditional statistical models to predict microbial contaminants in shallow groundwater, which has implications for developing a monitoring system of groundwater microbial quality.


KEYWORDS: Fecal contamination, global health, prediction, Bangladesh, deep learning, Tensorflow

