# LAST BEST RESULTS

### Titanic | Kaggle Learning Competition
# https://www.kaggle.com/c/titanic/overview


# LOAD DATA AND LIBRARIES -------------------------------------------------

# Libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(readr,dplyr, caret, DescTools, corrplot, DataExplorer, ggthemes, VIM)

# Datasets
train = read_csv('./data/train.csv')
test  = read_csv('./data/test.csv')

# merge
test$Survived = NA
full = rbind(train, test)

# Factorize variables
full = full %>%
  mutate(Survived = as.factor(Survived),
         Pclass = as.factor(Pclass),
         Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))

full_initial = full # leave the original dataset out


# DATA ENGINEERING --------------------------------------------------------

# Missing values
sapply(full, function(y) length(which(is.na(y))))
plot_missing(full)  # age and fare to be corrected later

# Simple imputation - delete the fare and embarked, mean for age, delete cabin
full$Cabin = NULL

full$Age[is.na(full$Age)] = mean(full$Age, na.rm = T)

full$Fare[is.na(full$Fare)] = median(full$Fare, na.rm = T)

full = full[complete.cases(full$Embarked), ]

# TODO Age imputation - complex one

# TODO Fare imputation

# TODO Embarked imputation

# TODO CREATE NEW VARIABLES: Title and more

# TODO cabin into level of the ship - leave undefined "U" - see if this brings your result up




# CLASSIFICATION MODELS ---------------------------------------------------

# Add class names to Survived for Caret's sake
full = full %>%
  mutate(Survived = ifelse(as.numeric(Survived) == 1, "yes", "no"),
         Survived = as.factor(Survived))









# TODO eliminate this shit - it is temporary columns being deleted for NOW
full = full %>%
  select(PassengerId, Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

# List of variables to be used

str(full)
Pclass + Sex + Age + SibSp + Parch + Ticket + Fare + Embarked + Title




# Back to Train-Test Split
train2 = full[!is.na(full$Survived), ]
test2 = full[is.na(full$Survived), ]

# Create Validation Set for model training
train_index = createDataPartition(train2$Survived, p = 0.8, list=FALSE) # for internal model comparison and tuning

# Apply cross validation with 10 folds
myControl = trainControl(method="cv", number = 10, classProbs = T, summaryFunction = twoClassSummary)

metric = "ROC"

# linear regression
glm.model = train(Survived~., 
                  data=train2, 
                  method="glm", 
                  metric=metric,
                  tuneLength = 5,
                  trControl=myControl)

print(glm.model)
plot(glm.model)

# decision tree 
dt.model = train(Survived~., 
                 data=train2, 
                 method="rpart", 
                 metric=metric,
                 tuneLength = 5,
                 trControl=myControl)

print(dt.model)
plot(dt.model)

# SVM
svm.model = train(Survived~., 
                  data=train2, 
                  method="svmRadial", 
                  metric=metric,
                  tuneLength = 5,
                  trControl=myControl)

print(svm.model)
plot(svm.model)

# Random Forest
rf.model = train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, 
                 data=train2,
                 method="rf", 
                 metric=metric,
                 tuneLength = 5,
                 trControl=myControl)

print(rf.model)
plot(rf.model)


# XGBoost
xgb.model = train(Survived~.,
                  data=train2,
                  metric = metric, 
                  method = "xgbTree",
                  tuneLength = 5,
                  trControl = myControl)

print(xgb.model)
plot(xgb.model)

# Compare models
results = resamples(list(glm=glm.model, dt=dt.model, rf=rf.model, xgb=xgb.model))

summary(results)
dotplot(results)

varImp(rf.model)

# visual comparison
library(lattice)
lattice::bwplot(results, metric = "ROC")


# MODEL TUNING AND ENSEMBLE -----------------------------------------------


# TODO model ensemble and tuining


# PREDICT AND SAVE RESULTS ------------------------------------------------

# TODO if you want to have your own prediction before you have to create validation test


# predict
model = rf.model
predictions = predict(model, test2)

# write submission

model_name = "rf.model"
approach_name ="_imp1"

file_name = paste("./output/", model_name, approach_name, ".csv", sep = "")

submit = data.frame(PassengerId = test$PassengerId, Survived = predictions) %>%
  mutate(Survived = ifelse(as.character(Survived) == 'yes', 1, 0))

write.csv(submit, file = file_name, row.names = FALSE)