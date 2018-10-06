# install dataset and caret packages
install.packages("caret")
install.packages("ISLR")
install.packages("e1071") 
install.packages("fastAdaboost")

packageDescription("caret")

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

# Create the dummy variables using predict. The Purchase variable will not be present in trainData_mat.
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

# visualize important variables using box plots and caret featurePlot function
featurePlot(x = trainset[, 1:18], 
            y = trainset$Purchase, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
# visualize important variables using density plots and caret featurePlot function
featurePlot(x = trainset[, 1:18], 
            y = trainset$Purchase, 
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
# recursive feature elimination using carets rfe function
set.seed(100)
options(warn=-1)

subsets <- c(1:5, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

install.packages("randomForest")

lmProfile <- rfe(x=trainset[, 1:18], y=trainset$Purchase,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

splom(trainset)
splom(trainset[,3:7])
correlationmatrix <- cor(trainset)
correlationmatrix <- cor(OJ[,2:18])
plot(correlationmatrix)
correlationmatrix
# checks the full dataset for missing data

# Train the model using randomForest and predict on the training data itself.
modelLookup('J48')
model_weka <- train(Purchase ~ ., data=trainset, method='J48')
model_weka
# plot model and see important variables
plot(model_weka, main="Model Accuracy with weka")
varimp_mars <- varImp(model_weka)
plot(varimp_mars, main="Variable Importance with J48")


#varimp_mars <- varImp(model_weka, useModel = FALSE)

fitted <- predict(model_weka)
fitted

# Preprocess test data set Step 1: Impute missing values 
#testData2 <- predict(preProcess_missingdata_model, testData)  

# Step 2: Create one-hot encodings (dummy variables)
testset1 <- predict(dummies_model, testset)

# Step 3: Transform the features to range between 0 and 1
testset1 <- predict(preProcess_range_model, testset1)

# View
head(testset1[, 1:10])

model_predict <- predict(model_weka, testset1)
model_predict
head(model_predict)
summary(model_predict)
head(testset)
str(testset)

# Compute the confusion matrix
confusionMatrix(reference = testset$Purchase, data = model_predict, mode='everything', positive='MM')

model_adaboost <- train(Purchase ~ ., data=trainset, method='adaboost')
model_adaboost
ada_predict <- predict(model_adaboost, testset)
confusionMatrix(reference = testset$Purchase, data = ada_predict, mode='everything', positive='MM')
# important variables
varimp_mars <- varImp(model_adaboost, useModel = FALSE)
plot(varimp_mars, main="Variable Importance with MARS")

# Define the training control
/*
  fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 5,                      # number of folds
    savePredictions = 'final',       # saves predictions for optimal tuning parameter
    classProbs = T,                  # should class probabilities be returned
    summaryFunction=twoClassSummary)  # results summary function
*/
  
  # Step 1: Tune hyper parameters by setting tuneLength
  /* set.seed(100)
model_mars2 = train(Purchase ~ ., data=trainData, method='earth', tuneLength = 5, metric='ROC', trControl = fitControl)
model_mars2

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars2, testData4)
confusionMatrix(reference = testData$Purchase, data = predicted2, mode='everything', positive='MM')
*/
  #svm
  model_svm <- train(Purchase ~ ., data=trainset, method='svmRadial', tuneLength = 5, metric = 'ROC', trControl = fitControl)
model_svm <- train(Purchase ~ ., data=trainset, method='svmRadial', importance = TRUE)
model_svm
svm_predict <- predict(model_svm, testset1)
confusionMatrix(reference = testset$Purchase, data = svm_predict, mode='everything', positive='MM')
#important variables
varimp_mars <- varImp(model_svm )
plot(varimp_mars, main="Variable Importance with SVM")

install.packages("pROC")
library(pROC)
selectedIndices <- model_svm$pred$mtry ==2
selectedIndices
plot.roc(model_svm$pred$obs[selectedIndices], model_svm$pred$M[selectedIndices])
#or
library(ggplot2)
ggplot(model_svm$pred[selectedIndices, ], 
       aes(m =M, d= factor(obs, levels = c ("R", "M"))))+
  geom_roc(n.cuts=0)+ coord_equal()+ style_roc() 

#geom_roc(hjust =-0.4, vjust =1.5)+ coord_equal() 

model_gbm <- train(Purchase ~ ., data=trainset, method='lvq')
varimp_mars <- varImp(model_gbm)
plot(varimp_mars, main="Variable Importance with MARS")

# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, WEKA=model_weka, SVM=model_svm))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

#, trControl=trainControl running multiple methods at once
install.packages("caretEnsemble")
library(caretEnsemble)
algorithmList <- c('adaboost', 'J48', 'svmRadial')
models <- caretList(Purchase ~ ., data=trainset, methodList=algorithmList)
results <- resamples(models)
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

#combining multiple models. not working
stack.glm <- caretStack(models, method="glm")
print(stack.glm)
stack_predict <- predict(stack.glm, testset)
confusionMatrix(reference = testset$Purchase, data = stack_predict, mode='everything', positive='MM')


#, metric="Accuracy", trControl=stackControl combine methods to one
stack.glm <- caretStack(models, method="glm")
print(stack.glm)
stack_predict <- predict(stack.glm, testset)
confusionMatrix(reference = testset$Purchase, data = stack_predict, mode='everything', positive='MM')
