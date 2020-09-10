# Churn Project
Exploration of two different types of predictive models (glmnet and rf) using the CARET package


The churn dataset contains data on a variety of telecom customers and the modeling challenge is to predict which 
customers will cancel their service (or churn).

I will explore two different types of predictive models, GLMNET and RANDOM FOREST, using the package CARET - that automates supervised learning models.

The first order of business is to create a reusable trainControl object that I can use to reliably compare them.

* The data: customer churn at telecom company
* Fit different models and choose the best
* Models must use the same training/test splits
* Create a shared trainControl object
