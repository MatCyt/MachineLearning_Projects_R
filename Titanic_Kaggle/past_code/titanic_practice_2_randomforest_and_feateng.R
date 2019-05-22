# Titanic Practice 2.2 random forest
# https://trevorstephens.com/kaggle-titanic-tutorial/r-part-5-random-forests/

library(readr)

# csv read
train = read_csv('./Titanic/data/train.csv')
test  = read_csv('./Titanic/data/test.csv')

# Bind train and test
test$Survived = NA
combi = rbind(train, test)


# Install and load required packages for decision trees and forests
library(rpart)
library(randomForest)
library(party)

# Join together the test and train sets for easier feature engineering
test$Survived = NA
combi = rbind(train, test)


# Convert to a string
combi$Name = as.character(combi$Name)


# Engineered variable: Title
combi$Title = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title = sub(' ', '', combi$Title)
# Combine small title groups
combi$Title[combi$Title %in% c('Mme', 'Mlle')] = 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'
# Convert to a factor
combi$Title = factor(combi$Title)


# Engineered variable: Family size
combi$FamilySize = combi$SibSp + combi$Parch + 1


# Engineered variable: Family
combi$Surname = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID = paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] = 'Small'
# Delete erroneous family IDs
famIDs = data.frame(table(combi$FamilyID))
famIDs = famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] = 'Small'
# Convert to a factor
combi$FamilyID = factor(combi$FamilyID)


#### Fill in Age NAs
summary(combi$Age)
Agefit = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] = predict(Agefit, combi[is.na(combi$Age),])

# Check what else might be missing
summary(combi)


# Fill in Embarked blanks
summary(combi$Embarked)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked = factor(combi$Embarked)


# Fill in Fare NAs
summary(combi$Fare)
which(is.na(combi$Fare))
combi$Fare[1044] = median(combi$Fare, na.rm=TRUE)


# Okay. Our dataframe is now cleared of NAs. Now on to restriction number two: Random Forests in R 
# can only digest factors with up to 32 levels. Our FamilyID variable had almost double that. 
# We could take two paths forward here, either change these levels to their underlying integers (using the unclass() function) 
# and having the tree treat them as continuous variables, 
#or manually reduce the number of levels to keep it under the threshold.

# Let’s take the second approach. To do this we’ll copy the FamilyID column to a new variable, FamilyID2, 
#and then convert it from a factor back into a character string with as.character(). 
#We can then increase our cut-off to be a “Small” family from 2 to 3 people. 
#Then we just convert it back to a factor and we’re done:


# New factor for Random Forests, only allowed <32 levels, so reduce number
combi$FamilyID2 = combi$FamilyID
# Convert back to string
combi$FamilyID2 = as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] = 'Small'
# And convert back to factor
combi$FamilyID2 = factor(combi$FamilyID2)


combi$Sex = as.factor(combi$Sex)

# Split back into test and train sets
train = combi[1:891,]
test = combi[892:1309,]

# Build Random Forest Ensemble
set.seed(415)
fit = randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2,
                    data=train, importance=TRUE, ntree=2000)

# Look at variable importance
varImpPlot(fit)

prop.table(table(combi$Title, combi$Survived), 1)

# Now let's make a prediction and write a submission file
Prediction = predict(fit, test)
submit = data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)


# Build condition inference tree Random Forest
set.seed(415)
fit = cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) 


# Now let's make a prediction and write a submission file
Prediction = predict(fit, test, OOB=TRUE, type = "response")
submit = data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "ciforest.csv", row.names = FALSE)
