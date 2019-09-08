'08/09/2019 - Sunday

The churn dataset contains data on a variety of telecom 
customers and the modeling challenge is to predict which 
customers will cancel their service (or churn).

I will explore two different types of predictive models: glmnet and rf. 

I will use the package CARET that Automates supervised learning (predictive modeling)

The first order of business is to create a reusable trainControl object 
that I can use to reliably compare them.'

#The data: customer churn at telecom company
#Fit different models and choose the best
#Models must use the same training/test splits
#Create a shared trainControl object

setwd("~/3. DATACAMP/Projects/Churn")

load("~/3. DATACAMP/Projects/Churn/Churn.RData")


library(caret)
library(C50)

set.seed(42)

'Make custom train/test indices'
#Use createFolds() to create 5 Cross-Validation folds on churn_y, the target variable
# Create custom indices: myFolds
myFolds <- createFolds(churn_y, k = 5)

# Compare class distribution
i <- myFolds$Fold1
table(churn_y[i]) / length(i)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)
'By saving the indexes in the train control, 
we can fit many models using the same CV folds.
'

'glmnet as a baseline model
The glmnet model is simple, fast, and easy to interpret
 You can interpret the coefficients the same way as the coefficients
 from an lm or glm model.
'
# Fit glmnet model: model_glmnet
model_glmnet <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "glmnet",
  trControl = myControl
)

'Random forest '
# Fit random forest: model_rf
model_rf <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "ranger",
  trControl = myControl
)

'This random forest uses the custom CV folds, 
o we can easily compare it to the baseline model.'


'Matching train/test indices
What is the primary reason that train/test indices 
need to match when comparing two models?
Because otherwise you would not be doing a fair comparison of your models and your 
results could be due to chance.
Train/test indexes allow you to evaluate your models 
out of sample so you know that they work!
'



'Create a resamples object
it is time to compare the out-of-sample predictions 
and choose which one is the best model for my dataset.
'
#Create a resamples object
# Create model_list
model_list <- list(item1 = model_glmnet, item2 = model_rf)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)


'BOX and WHISKER PLOT
the box-and-whisker plot allows you to compare the 
distribution of predictive accuracy (in this case AUC) for the two models.

In general, you want the model with the higher median AUC, 
as well as a smaller range between min and max AUC.

Box and whisker plots show: 
the median of each distribution as a line and 
the interquartile range of each distribution as a box around the median line. 
'
# Create bwplot
bwplot(resamples, metric = "ROC")


#bwplot(resamples)

'Create a scatterplot
Another useful plot for comparing models is the scatterplot, 
also known as the xy-plot. 
This plot shows you how similar the two models performances 
are on different folds.

It is particularly useful for identifying if one model is
consistently better than the other across all folds, 
or if there are situations when the inferior model produces 
better predictions on a particular subset of the data.

These scatterplots let you see if one model is always better than the other.
'
# Create xyplot
xyplot(resamples, metric = "ROC")
