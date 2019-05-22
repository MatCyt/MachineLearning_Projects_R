### Titanic | Kaggle Learning Competition
# https://www.kaggle.com/c/titanic/overview


# LOAD DATA AND LIBRARIES -------------------------------------------------

# Libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(readr,dplyr, caret, DescTools, corrplot, DataExplorer, ggthemes)

# Datasets
train = read_csv('./data/train.csv')
test  = read_csv('./data/test.csv')

# merge
test$Survived = NA
full = rbind(train, test)

attach(full)


# DATA EXPLORATION --------------------------------------------------------

str(full)
summary(full)

View(full)

# Factorize variables
full = full %>%
  mutate(Survived = as.factor(Survived),
         Pclass = as.factor(Pclass),
         Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))

# Frequencies for categorical variables
plot_bar(full)

head(full$Name) # names and titles
head(full$Cabin) # possibly a deck level and cabin number combined

# Frequencies for numerical variables
plot_histogram(full[ ,-1])
plot_density(full)

# Missing Values
sapply(full, function(y) length(which(is.na(y))))
plot_missing(full)  # age and fare to be corrected later


### Survival exploration

# Sex impact on survival
# females were significantly more likely to survive the ship sinking
survived_sex = prop.table(table(Survived, Sex), 2)

ggplot() + 
  geom_bar(aes(y = value, x = Sex, fill = factor(Survived)), data = melt(survived_sex), stat = "identity") + 
  labs(y = "percentage") + 
  scale_fill_discrete(name = "Survived",
                      labels = c("Yes", "No"))


# Age impact on survival
# higher chance of survival for younger people 
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram(bins = 40) + 
  xlab("Age") +
  scale_fill_discrete(name = "Survived") +
  theme_few()
# Two graphs from above show the rule "Woman and Children first" - in real life.


# Family size
# it's definitely bad to be a single male
ggplot(full[1:891,], aes(x = SibSp, fill = factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  labs(x = 'Siblings/Spouses') +
  theme_few() +
  facet_grid(.~Sex)

ggplot(full[1:891,], aes(x = Parch, fill = factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  labs(x = 'Parents/Children') +
  theme_few() + 
  facet_grid(.~Sex)

# Embarked and survival
# there is a slight difference in the port of origin and survival level
# maybe caused by different classes of passengers entering at different port?
ggplot(na.omit(full), aes(x = Embarked, fill = factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  labs(x = 'Port of Origin') +
  theme_few()

# Money impact on survival - Class and fare price
ggplot(na.omit(full), aes(x = Pclass, fill = factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  labs(x = 'Class') +
  theme_few()

prop.table(table(Survived, Pclass), 2) # much higher chance for the 1st class passengers to survive the crash

ggplot(full[1:891,], aes(Fare, fill = factor(Survived))) + 
  geom_histogram(bins = 40) + 
  xlab("Fare") +
  scale_fill_discrete(name = "Survived") +
  theme_few()

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







# Back to Train-Test Split
train2 = full[!is.na(full$Survived), ]
test2 = full[is.na(full$Survived), ]

# Create Validation Set for model training
train_index = createDataPartition(train2$Survived, p = 0.8, list=FALSE) # for internal model comparison and tuning

# Apply cross validation with 10 folds
myControl = trainControl(method="cv", number = 5, classProbs = T, summaryFunction = twoClassSummary)

metric = "ROC"

# linear regression
glm.model = train(Survived~., 
                  data=train2, 
                  method="glm", 
                  metric=metric,
                  tuneLength = 2,
                  trControl=myControl)

print(glm.model)
plot(glm.model)

# decision tree 
dt.model = train(Survived~., 
                 data=train2, 
                 method="rpart", 
                 metric=metric,
                 tuneLength = 2,
                 trControl=myControl)

print(dt.model)
plot(dt.model)

# SVM
svm.model = train(Survived~., 
                  data=train2, 
                  method="svmRadial", 
                  metric=metric,
                  tuneLength = 4,
                  trControl=myControl)

print(svm.model)
plot(svm.model)

# Random Forest
rf.model = train(Survived~., 
                 data=train2,
                 method="rf", 
                 metric=metric,
                 tuneLength = 2,
                 trControl=myControl)

print(rf.model)
plot(rf.model)

# XGBoost
xgb.model = train(Survived~.,
                  data=train2,
                  metric = metric, 
                  method = "xgbTree",
                  tuneLength = 2,
                  trControl = myControl)

print(xgb.model)
plot(xgb.model)

# Compare models
results = resamples(list(glm=glm.model, dt=dt.model, rf=rf.model, xgb=xgb.model))

summary(results)
dotplot(results)

# visual comparison
library(lattice)
lattice::bwplot(resamp, metric = "ROC")


# MODEL TUNING AND ENSEMBLE -----------------------------------------------


# TODO model ensemble and tuining


# PREDICT AND SAVE RESULTS ------------------------------------------------

# TODO if you want to have your own prediction before you have to create validation test


# predict
model = rf.model
predictions = predict(model, test2)

# write submission

model_name = "rf.model"
approach_name ="_basics"

file_name = paste("./output/", model_name, approach_name, ".csv", sep = "")

submit = data.frame(PassengerId = test$PassengerId, Survived = predictions) %>%
  mutate(Survived = ifelse(as.character(Survived) == 'yes', 1, 0))

write.csv(submit, file = file_name, row.names = FALSE)
