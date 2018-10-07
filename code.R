# install dataset and caret packages
# install.packages("caret")
# install.packages("ISLR")
# install.packages("e1071") 
# install.packages("fastAdaboost")
# 
# packageDescription("caret")

# Load the packages
library(caret)
library(ISLR)

# Import dataset
data(OJ)

# Structure of the dataframe
str(OJ)
summary(OJ)

# See top 6 rows and 10 columns
head(OJ[, 1:10])
head(OJ)

# Create the training and test datasets using caret's createDataPartition function. 
set.seed(2)

# tag rows for the training data
trainRow <- createDataPartition(OJ$Purchase, p=0.8, list=FALSE)

# Create the training  dataset
trainset <- OJ[trainRow,]

# Create the test dataset
testset <- OJ[-trainRow,]
summary(testset)

# Store Y for later use.
y = trainset$Purchase

any(is.na(OJ))

# One-Hot Encoding. Creating dummy variables by converting a categorical variable to as many binary variables as here are categories.
# The Purchase variable is excluded
dummies_model <- dummyVars(Purchase ~ ., data=trainset)

# Create the dummy variables using predict. 
trainData_mat <- predict(dummies_model, newdata = trainset)

# Convert to dataframe
trainset <- data.frame(trainData_mat)

# See the structure of the new dataset
str(trainset)

# Normalize our data set using another caret function
preProcess_range_model <- preProcess(trainset, method='range')
trainset <- predict(preProcess_range_model, newdata = trainset)

# Append the Y variable back to training set
trainset$Purchase <- y
str(trainset)
# see first 10 variables, confirm its btw 0 and 1. predictors are now within range of 0 and 1
apply(trainset[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

# visualize variables using box plots and caret featurePlot function
featurePlot(x = trainset[, 1:18], 
            y = trainset$Purchase, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
# visualize variables using density plots and caret featurePlot function
featurePlot(x = trainset[, 1:18], 
            y = trainset$Purchase, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

# recursive feature elimination using carets rfe function
set.seed(100)

subsets <- c(1:5, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

#install.packages("randomForest")

lmProfile <- rfe(x=trainset[, 1:18], y=trainset$Purchase,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

# Train the model using weka and predict on the training data itself.
modelLookup('J48')
model_weka <- train(Purchase ~ ., data=trainset, method='J48')
model_weka
# plot model and see important variables
plot(model_weka, main="Model Accuracy with weka")
varimp_mars <- varImp(model_weka)
plot(varimp_mars, main="Variable Importance with J48")

# Create one-hot encodings (dummy variables) for test set
testset1 <- predict(dummies_model, testset)

# Transform the features to range between 0 and 1
testset1 <- predict(preProcess_range_model, testset1)

# View
head(testset1[, 1:10])
#Run prediction on test set
model_predict <- predict(model_weka, testset1)
# Compute the confusion matrix
confusionMatrix(reference = testset$Purchase, data = model_predict, mode='everything', positive='MM')

#adaboost method
model_adaboost <- train(Purchase ~ ., data=trainset, method='adaboost')
model_adaboost
ada_predict <- predict(model_adaboost, testset)
confusionMatrix(reference = testset$Purchase, data = ada_predict, mode='everything', positive='MM')
# important variables
varimp_mars <- varImp(model_adaboost, useModel = FALSE)
plot(varimp_mars, main="Variable Importance with MARS")


model_gbm <- train(Purchase ~ ., data=trainset, method='lvq')
varimp_mars <- varImp(model_gbm)
plot(varimp_mars, main="Variable Importance with MARS")

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, WEKA=model_weka))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

