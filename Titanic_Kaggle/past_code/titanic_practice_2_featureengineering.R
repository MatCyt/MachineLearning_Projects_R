# Titanic Practice 2.2 Feature engineering
# https://trevorstephens.com/kaggle-titanic-tutorial/getting-started-with-r/

# csv read
train = read_csv('./Titanic/data/train.csv')
test  = read_csv('./Titanic/data/test.csv')

# Bind train and test
test$Survived = NA
combi = rbind(train, test)

# Working with NAME 
combi$Name[1]

# Find the indexes for the tile piece of the name
strsplit(combi$Name[1], split='[,.]')[[1]][2]

# New variable - Title
combi$Title = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title = sub(' ', '', combi$Title)

table(combi$Title)

# Combine titles

# Combine small title groups
combi$Title[combi$Title %in% c('Mme', 'Mlle')] = 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir')] = 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] = 'Lady'
# Convert to a factor
combi$Title = factor(combi$Title)

# Engineered variable: Family size
combi$FamilySize = combi$SibSp + combi$Parch + 1


# Engineered variable: Family. Extract name and create family for people with same famili size and surname
combi$Surname = sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID = paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] = 'Small'
# Inspect new feature
table(combi$FamilyID)
# Delete erroneous family IDs
famIDs = data.frame(table(combi$FamilyID))
famIDs = famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] = 'Small'
# Convert to a factor
combi$FamilyID = factor(combi$FamilyID)



# Split back into test and train sets
train = combi[1:891,]
test = combi[892:1309,]

# Build a new tree with our new features
fit = rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
             data=train, method="class")
fancyRpartPlot(fit)

# Now let's make a prediction and write a submission file
Prediction = predict(fit, test, type = "class")
submit = data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "engineeredfeaturestree.csv", row.names = FALSE)