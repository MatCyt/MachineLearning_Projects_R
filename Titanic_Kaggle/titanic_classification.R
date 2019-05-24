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

full_initial = full # leave the original dataset out

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

### DE | Create new variables ----

# Title
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

table(full$Sex, full$Title)

officers = c("Capt", "Col", "Major", "Dr")
royalty = c("Dona", "Don", "Jonkheer", "Lady", "Rev", "Sir", "the Countess")

full$Title[full$Title %in% royalty] = "royalty"
full$Title[full$Title %in% officers] = "officer"
full$Title[full$Title == "Mlle" | full$Title == "Ms"] = "Miss"
full$Title[full$Title == "Mme"] = "Mrs"
full$Title[full$Title == "the Countess"]

barplot(prop.table(table(full$Survived, full$Title),2))


## Missing values - Cabin
# Leave the first letter as indication of the deck number and replace others for "U" as in unknown
full = full %>%
  mutate(Cabin = ifelse(is.na(Cabin), "U", Cabin),
         Cabin = factor(substr(Cabin, 1, 1)))

test %>% filter(Cabin == "G")



# TODO Mother




### DE | Missing values ----
sapply(full, function(y) length(which(is.na(y))))
plot_missing(full)  # age and fare to be corrected later



## Missing values - Age

# TODO Age imputation - complex one

full$Age[is.na(full$Age)] = mean(full$Age, na.rm = T)




### Missing values - Embarked


## mode - Southampton
Mode(full$Embarked)


## analysis
# both are upper class, middle-upper age females with fare at 80
full[is.na(Embarked), ] 

# highes age average for Cherbourg
full %>%
  group_by(Embarked) %>%
  summarise(age = mean(Age, na.rm = T)) 

# Fare by Pclass and Embarked
embarked = full %>%
  filter(complete.cases(Embarked))

# Cherbourg Fare match their fare level for their class
ggplot(embarked, aes(x = Embarked, y = Fare, fill = Pclass)) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80)) +
  theme_few()
# most first class passengers entered the ship at Cherbourg
prop.table(table(Pclass, Embarked),2)

# From the pure analysis it would seem that the two most likely embarked at Cherbourg


# knn - VIM
embarked_knn = VIM::kNN(full, variable = c("Embarked"), k = 5)
embarked_knn[c(62,830), ]

# contrary to analysis knn suggest that both of them embarked in Southampton - let's stick with that
full = embarked_knn %>%
  select(-Embarked_imp)




# TODO Fare imputation
full[is.na(Fare), ]

fare_knn = VIM::kNN(full, variable = c("Fare"), k = 5)
fare_mean_adjusted = mean(full[Pclass ==3, "Fare"], na.rm = T)

# TODO 
full = fare_knn %>%
  select(-Fare_imp)


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
