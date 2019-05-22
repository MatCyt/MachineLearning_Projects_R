### Simple submission to try out caret workflow and model comparison

### libraries
library(caret)
library(dplyr)
library(pROC)
library(readr)

### data
train = read_csv('./Titanic/data/train.csv')
test  = read_csv('./Titanic/data/test.csv')

# merge
test$Survived = NA
full = rbind(train, test)



### simple data preprocessing ----

# change into factor
factor_vars = c('Pclass','Sex','Embarked')
full[factor_vars] = lapply(full[factor_vars], function(x) as.factor(x))

# Factorize Survived

full = full %>%
  mutate(Survived = ifelse(Survived == 1, "yes", "no"))

full$Survived = as.factor(as.character(full$Survived))



### NA imputation ----

# simple imputation of NAs in age with a mean
full$Age[is.na(full$Age)] = mean(full$Age, na.rm = T)

# simple imputation of NAs in Fare - median
full[is.na(full$Fare),] # 1 row

full$Fare[is.na(full$Fare)] = median(full$Fare, na.rm = T)

# simple imputation of NAs in Embarked
sum(is.na(full$Embarked))
full[is.na(full$Embarked), ]

table(full$Embarked, full$Pclass)

full$Embarked[is.na(full$Embarked)] = "S"




# Model comparison preparation ----

### run several models for comparison - glm, decision tree, random forest, xgboost
val = full %>%
  filter(complete.cases(full$Survived)) # new training set with imputed NAs

train_index = createDataPartition(val$Survived, p = 0.8, list=FALSE) # train-test split for internal model comparison 

val_train = val[train_index,]
val_test = val[-train_index,]

myControl = trainControl(method="cv", number=10, classProbs = T, summaryFunction = twoClassSummary)

# Run models fo comparison ----
# Models 
names(getModelInfo())

# GLM with tuneLength
glm_model = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                  val_train,
                  metric = "Accuracy", 
                  method = "glmnet",
                  tuneLength = 5,
                  trControl = myControl)

print(glm_model)
plot(glm_model)

# Decision trees
dt_model = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                  val_train,
                  metric = "Accuracy", 
                  method = "glmnet",
                  tuneLength = 5,
                  trControl = myControl)

print(dt_model)
plot(dt_model)


# Random Forest
rf_model = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                  val_train,
                  metric = "Accuracy", 
                  method = "ranger",
                  tuneLength = 5,
                  trControl = myControl)

print(rf_model)
plot(rf_model)

# XGBoost
xgb_model = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                 val_train,
                 metric = "Accuracy", 
                 method = "xgbTree",
                 tuneLength = 2,
                 trControl = myControl)

print(xgb_model)
plot(xgb_model)


# Compare models
model_list = list(glmmet = glm_model,
                  dt = dt_model,
                  rf = rf_model,
                  xgb = xgb_model)
resamp = resamples(model_list)

# results comparison
summary(resamp)

# visual comparison
library(lattice)
lattice::bwplot(resamp, metric = "ROC")


# compare on dummy test - for my own trial
p_glm = predict(glm_model, val_test)
confusionMatrix(p_glm, as.factor(val_test$Survived))

p_dt = predict(dt_model, val_test)
confusionMatrix(p_dt, as.factor(val_test$Survived))

p_rf = predict(rf_model, val_test)
confusionMatrix(p_rf, as.factor(val_test$Survived))

p_xgb = predict(xgb_model, val_test)
confusionMatrix(p_xgb, as.factor(val_test$Survived))



### ACTUAL RUN ON WHOLE TRAIN SET FOR RF AND XGB

# re - split
train2 = full[!is.na(full$Survived), ]
test2 = full[is.na(full$Survived), ]
  
myControl = trainControl(method="cv", number=10, classProbs = T)

# Random Forest
rf_model2 = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                 train2,
                 metric = "Accuracy", 
                 method = "ranger",
                 tuneLength = 5,
                 trControl = myControl)

print(rf_model2)
plot(rf_model2)

# XGBoost
xgb_model2 = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                  train2,
                  metric = "Accuracy", 
                  method = "xgbTree",
                  tuneLength = 2,
                  trControl = myControl)

print(xgb_model2)
plot(xgb_model2)


### small tuning - for later

# tune grid


### predict and save results

# predict rf
predict_rf = predict(rf_model2, test2, OOB=TRUE)
submit = data.frame(PassengerId = test$PassengerId, Survived = predict_rf) %>%
  mutate(Survived = ifelse(as.character(Survived) == 'yes', 1, 0))
write.csv(submit, file = "my_test_rf.csv", row.names = FALSE)

# predict xgb
predict_xgb = predict(xgb_model2, test2)
submit = data.frame(PassengerId = test$PassengerId, Survived = predict_xgb) %>%
  mutate(Survived = ifelse(as.character(Survived) == 'yes', 1, 0))
write.csv(submit, file = "my_test_xgb.csv", row.names = FALSE)
