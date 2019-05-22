# Titanic Practice 2
# https://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/

# csv read
train = read_csv('./Titanic/data/train.csv')
test  = read_csv('./Titanic/data/test.csv')


### Gender Class Naive Model

# Proportion of each sex that survived
prop.table(table(train$Sex, train$Survived))
prop.table(table(train$Sex, train$Survived), 1) # sum to 1 across gender

# Assign survive class to all females in test set
# Create new column in test set with our prediction that everyone dies
test$Survived = 0
# Update the prediction to say that all females will survive
test$Survived[test$Sex == 'female'] = 1

train$Child = 0
train$Child[train$Age < 18] = 1
aggregate(Survived ~ Child + Sex, data=train, FUN=sum) # number of survivors across age and sex
aggregate(Survived ~ Child + Sex, data=train, FUN=length) # total number
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)}) # proportion

# not conclusive for age (for 18 years old)


# Class variable and Fare

# Lets bin the fare 
train$Fare2 = '30+'
train$Fare2[train$Fare < 30 & train$Fare >= 20] = '20-30'
train$Fare2[train$Fare < 20 & train$Fare >= 10] = '10-20'
train$Fare2[train$Fare < 10] = '<10'
aggregate(Survived ~ Fare2 + Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})

# most of the males didn't make it but so did the women who paid more than 20 and were in class 3

# New prediction - males and females with class 3 and ticket worth more than 20 - 0 and other females are 1

# Create new column in test set with our prediction that everyone dies
test$Survived = 0
# Update the prediction to say that all females will survive
test$Survived[test$Sex == 'female'] = 1
# Update once more to say that females who pay more for a third class fare also perish
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] = 0

# Create submission dataframe and output to file
submit = data.frame(PassengerId = test$PassengerId, Survived = test$Survived)
write.csv(submit, file = "genderclassmodel.csv", row.names = FALSE)


str(train)
