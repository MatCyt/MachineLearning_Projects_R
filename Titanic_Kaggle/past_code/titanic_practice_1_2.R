# TITANIC practice approach 2
# https://alexiej.github.io/kaggle-titanic/#r-wstep

library(tidyverse) # metapackage with lots of helpful functions

# csv read
train = read_csv('./Titanic/data/train.csv')
test  = read_csv('./Titanic/data/test.csv')

# set full
test$Survived = NA
full = rbind(train, test)

# Dane
full$Sex2 = as.numeric(as.factor(full$Sex))
full$Embarked2 = as.numeric(as.factor(full$Embarked))
full$Cabin2 =as.numeric(as.factor(substring(full$Cabin, 0, 1) ))

## Title
full$Title = str_match(full$Name, ",\\s([^ .]+)\\.?\\s+")[,2]
full$Title2 = full$Title
full$Title2[ full$Title %in% c('Mlle','Ms','Lady')] = 'Miss'
full$Title2[ full$Title %in% c('Mme')] = 'Mrs'
full$Title2[ full$Title %in% c('Sir')] = 'Mr'
full$Title2[ ! full$Title %in% c('Miss','Master','Mr','Mrs')] = 'Other' 
full$TitleN = as.numeric(as.factor(full$Title2))

## TicketCount
full = full %>% group_by(Ticket) %>% mutate(TicketCount = n()) %>% ungroup()

# Uzupelnianie NaN
## Embarked
full[62,'Embarked'] = 'S'
full[830,'Embarked'] = 'S'
full$Embarked2 = as.numeric(as.factor(full$Embarked))

##  Fare
full[1044,'Fare'] =  (7.25 + 6.2375)/2 # we set average for this values

## Age
library(zoo)
full$Age = as.vector((na.aggregate(full[,"Age"],na.rm=FALSE))$Age)

## Cabin
full =  full %>% mutate(Cabin2 = ifelse( is.na(Cabin2),3, Cabin2) )

## Summarise N/A
# full %>%  select(everything()) %>% summarise_all(funs(sum(is.na(.))))

# Normalizacja
full$Fare2 = as.vector(scale(full$Fare))

# Age categorize
full$AgeCategory = as.numeric(as.factor(cut(full$Age,breaks = c(0,9,18,30,40,50,100))))

cols = c('Pclass','SibSp','Parch','Sex2','Embarked2','Cabin2','TitleN','TicketCount','AgeCategory','Fare2')
cols2 = c('Pclass','SibSp','Parch','Sex2','Embarked2','Cabin2','TitleN','TicketCount','AgeCategory','Fare2','Survived')

#ML Section
library(mlr)

#full_input = normalizeFeatures(full_input, target = "Survived")
train_input = full[,cols2]  %>% filter(!is.na(Survived) )
train_input$Survived = as.factor(train_input$Survived)

# Create an xgboost learner that is classification based and outputs
#install.packages("kernlab")
task = makeClassifTask(data = train_input, target = "Survived")

xgb_learner = makeLearner("classif.xgboost")
mod = train(xgb_learner, task)
pred = predict(mod, task = task)
print(performance(pred, measures = list("acc" = acc)))

#predict
test_data = full %>% filter(is.na(Survived) ) 
test_passengersID = test_data[,c('PassengerId')]
test_input = test_data[,cols]

pred = as.data.frame(predict(mod,newdata = test_input))

# write to csv
colnames(pred) = c("Survived")
write.csv(cbind(test_passengersID,pred),'output.csv', quote = FALSE, row.names = FALSE)


## IMPROVEMENT

# better age imputation
# optimize parameters for XGBoost


# Better age imputation - correlation with other features
abs(cor(full[,c('Age','Sex2', 'Embarked2','Fare', 'Parch' , 'Pclass', 'SibSp', 'Cabin2','TitleN')],
        use="complete.obs")[,"Age"]) %>% .[order(., decreasing = TRUE)]

title.age = aggregate(full$Age,by = list(full$TitleN,full$Pclass), FUN = function(x) median(x, na.rm = T))
title.age[15,]$x = 41
full[is.na(full$Age), "Age"] = apply(full[is.na(full$Age), ] , 1, function(x) title.age[title.age[, 1]==x["TitleN"] & title.age[, 2]==x["Pclass"], 3])
# Age categorize
full$AgeCategory = as.numeric(as.factor(cut(full$Age,breaks = c(0,9,18,30,40,50,100))))

# Re-Run predict - slight improvement


# Optimize XGBoost parameters - TLDR - overfitting and lower Kaggle score

# getParamSet("classif.xgboost")
xgb_learner = makeLearner(
  "classif.xgboost",
  predict.type = "response",
  par.vals = list(
    objective = "binary:logistic",
    eval_metric = "error",
    nrounds = 200
  )
)


par.set = makeParamSet(
  # The number of trees in the model (each one built sequentially)
  makeIntegerParam("nrounds", lower = 100, upper = 500),
  # number of splits in each tree
  makeIntegerParam("max_depth", lower = 1, upper = 10),
  # "shrinkage" - prevents overfitting
  makeNumericParam("eta", lower = .1, upper = .5),
  # L2 regularization - prevents overfitting
  makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x),
  
  # Dla Overfitting
  makeNumericParam("gamma", -15, 15, trafo = function(x) 2^x),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)

control = makeTuneControlRandom(maxit = 20)

# Create a description of the resampling plan
resampling <- makeResampleDesc("CV", iters = 4) #Cross Validation

tuned_params <- tuneParams(
  learner = xgb_learner,
  task = task,
  resampling = resampling,
  par.set = par.set,
  control = control
)


# Create a new model using tuned hyperparameters
xgb_tuned_learner <- setHyperPars(
  learner = xgb_learner,
  par.vals = tuned_params$x
)

# Verify performance on cross validation folds of tuned model
resample(xgb_tuned_learner,task,resampling,measures = list(acc,mmce))

mod = train(xgb_tuned_learner, task)
pred = predict(mod, task = task)
print(performance(pred, measures = list("acc" = acc)))


#predict
test_data <- full %>% filter(is.na(Survived) ) 
test_passengersID <- test_data[,c('PassengerId')]
test_input <- test_data[,cols]

pred <- as.data.frame(predict(mod,newdata = test_input))
# write to csv
colnames(pred) = c("Survived")
write.csv(cbind(test_passengersID,pred),'outputR.csv', quote = FALSE, row.names = FALSE)
