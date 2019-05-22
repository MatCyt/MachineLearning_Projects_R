# TITANIC practice approach
# https://alexiej.github.io/kaggle-titanic/#r-wstep


### Libraries ----
library(dplyr)
library(caret)
library(readr)
library(corrplot)
library(stringr)
library(tidyr)


### Data ----
# load the datasets from local paths
train = read_csv("./Titanic/data/train.csv")
test = read_csv("./Titanic/data/test.csv")

# combine test and set to work and evaluate full dataset 
test$Survived = NA #  input NAs in test class column
full = rbind(train, test)


### Explore and re-work Data ----

## basic exploration

dim(full)
str(full)
summary(full)

View(full)


## Categorize Cabin, Sex and Embarked
full$Cabin2 = as.numeric(as.factor(substring(full$Cabin, 0, 1) ))
full$Sex2 = as.numeric(as.factor(full$Sex))
full$Embarked2 = as.numeric(as.factor(full$Embarked))


## Visualize Correlations
cols = c('Age','Sex2','Embarked2','Fare','Parch','Pclass','SibSp','Survived','Cabin2')
cor(full[,cols], use="complete.obs")

corrplot(cor(full[,cols], use="complete.obs"), method = "number")


## Get the title from name and surname
full$Title = str_match(full$Name, ",\\s([^ .]+)\\.?\\s+")[,2]

# count by title
full %>% 
  group_by(Title) %>% 
  summarise(cnt = n()) %>%
  arrange(desc(cnt))

# group the titles - too many of them
full$Title2 = full$Title
full$Title2[ full$Title %in% c('Mlle','Ms','Lady')] = 'Miss'
full$Title2[ full$Title %in% c('Mme')] = 'Mrs'
full$Title2[ full$Title %in% c('Sir')] = 'Mr'
full$Title2[ ! full$Title %in% c('Miss','Master','Mr','Mrs')] = 'Other' 
full$TitleN = as.numeric(as.factor(full$Title2))

full %>% 
  group_by(Title2) %>% 
  summarise(cnt = n()) %>%
  arrange(desc(cnt))

# Ticket count
full = full %>% 
  group_by(Ticket) %>% 
  mutate(TicketCount = n()) %>% 
  ungroup()


### Visualize the dataset ----

# histograms and boxplots
ggplot(data=full, aes(full$Age)) + geom_histogram()

ggplot(data=full, aes(Survived, Age, group = Survived)) + geom_boxplot() # age - in favor of younger

ggplot(data=full, aes(Pclass, Age, group = Pclass)) + geom_boxplot() # 3rd class if the youngest

# correlation map - other way
cols = c('Age','Sex2','Embarked2','Fare','Parch','Pclass','SibSp','Survived','Cabin2')
corr = cor( full[,cols], use="complete.obs")

heatmap(corr, na.rm = T)

# ggplot way
library(reshape2)
corr_df = melt(corr, na.rm = TRUE)

ggplot(corr_df, aes(x=Var1,y=Var2, fill=value)) +
  geom_tile() +
  geom_text(aes(Var2, Var1, label = round(value,2)), color = "white", size = 4) 

## Performance Analytics plot
library("PerformanceAnalytics")

cols = c('Age','Sex2','Embarked2','Fare','Parch','Pclass','SibSp','Survived','Cabin2','TitleN')

chart.Correlation(full[,cols], histogram=TRUE, pch=19, font.size = 15)


## Plot and visualize by variable and survival
library(ggplot2)
require(gridExtra)

full$AgeCut = cut(full$Age,breaks = seq(0, 100, by = 10))
full$FareCut = cut(full$Fare,10)


cols = c('AgeCut','Sex','Embarked','FareCut','Parch','Pclass','SibSp', 'Cabin2','Title2')
vector = list()


get_plot = function (data, col) {
  data_group = data[,c(col,"Survived")] %>% 
    select(x = col, "Survived") %>%
    drop_na(Survived) %>% 
    group_by(x) %>% 
    summarise(Total = n(),Survived = sum(Survived,na.rm = T), ratio = sum(Survived,na.rm = T)/n())
  
  plot = ggplot(data_group ,aes(x, Survived, label='A')) +
    geom_bar(aes(y=Total), stat="identity", fill="red") +
    geom_bar(aes(y=Survived), stat="identity", fill="lightgreen")  +
    geom_text(aes(label = (round(Survived/Total,4) * 100)), color='blue', vjust = -5.25) + 
    xlab(col)
  
  return(plot)
}

for (col in cols) {
  vector[[col]] = get_plot(full,col)
}

grid.arrange( grobs = vector, ncol = 2)

### Imputing NAs ----

# count NAs by column

full %>%
  summarise_all(funs(sum(is.na(.)))) 

colSums(is.na(full))

# EMBARKED - few values
full %>% 
  filter( is.na(Embarked) )

# fill embarked with values from similar rows.
# which features is it related to?
cols = c('Age','Sex2', 'Embarked2','Fare', 'Parch' , 'Pclass', 'SibSp', 'Cabin2','TitleN')
abs(cor(full[,cols],use="complete.obs")[,"Embarked2"]) %>% .[order(., decreasing = TRUE)]

# we need to calculate fare for Cabin2 and Pclass and see which Embarked is the closest one
full %>% 
  filter(Pclass == 1 & Cabin2 == 2  ) %>% 
  group_by(Embarked) %>% 
  summarise( sum = sum(Fare), count = n(), mean = mean(Fare), median = median(Fare))

# Show on boxplot
ggplot(data=full %>%filter(Pclass == 1 & Cabin2 == 2 & Embarked %in% c('C','S') ), aes(Embarked, Fare, group = Embarked)) + 
  geom_boxplot() + geom_hline(yintercept=80,color='red')

# Median values are similar for C and S, but since the mean is closer to S (less variability in mean) we will go with S
full[62,'Embarked'] = 'S'
full[830,'Embarked'] = 'S'

full$Embarked2 = as.numeric(as.factor(full$Embarked))

# FARE - imputation
# similar approach as in Embarked - look for similar values, correlated variables and impute the median based on the closes value

full %>% 
  filter( is.na(Fare) )

cols = c('Age','Sex2', 'Embarked2','Fare', 'Parch' , 'Pclass', 'SibSp', 'Cabin2','TitleN')
abs(cor(full[,cols],use="complete.obs")[,"Fare"]) %>% .[order(., decreasing = TRUE)]

full %>% 
  filter(Pclass == 3 & Embarked == 'S' & Parch == 0 & Sex =='male'  & Age > 40) %>% 
  group_by(Age) %>% 
  summarise( sum = sum(Fare), count = n(), mean = mean(Fare), median = median(Fare))

full[1044,'Fare'] =  (7.25 + 6.2375)/2 # we set average for this values


# AGE - mean value imputation
library(zoo)
full$Age <- as.vector((na.aggregate(full[,"Age"],na.rm=FALSE))$Age)

# Cabin - most frequent element - we will use Cabin2
sort(table(full$Cabin2)) # it is 3

full =  full %>% mutate(Cabin2 = ifelse( is.na(Cabin2),3, Cabin2) ) # impute the NAs


# no remaining NAs
full %>%
  select(everything()) %>%  # replace to your needs
  summarise_all(funs(sum(is.na(.))))


#### MODEL ----

# Normalize all columns
# Select best model
# Predict scores and send them to kaggle

# Normalize the Fare (since Age is going to be categorized)
full$Fare2 = as.vector(scale(full$Fare))

# Categorize Age
full$AgeCategory = as.numeric(as.factor(cut(full$Age,breaks = c(0,9,18,30,40,50,100))))


# Select columns taking part in the modelling (train and test)
cols = c('Pclass','SibSp','Parch','Sex2','Embarked2','Cabin2','TitleN','TicketCount','AgeCategory','Fare2')
cols2 = c('Pclass','SibSp','Parch','Sex2','Embarked2','Cabin2','TitleN','TicketCount','AgeCategory','Fare2','Survived')
full[,cols]


# Run MLR

#ML Section
library(mlr)

#full_input <- normalizeFeatures(full_input, target = "Survived")
train_input = full[,cols2]  %>% filter(!is.na(Survived) )
train_input$Survived = as.factor(train_input$Survived)

# Classification Task
task = makeClassifTask(data = train_input, target = "Survived")

# Make Learner
xgb_learner <- makeLearner("classif.xgboost")

# Train
mod = train(xgb_learner, task)

# Predict on training data
pred = predict(mod, task = task)
print(performance(pred, measures = list("acc" = acc)))


# Save for Kaggle upload
test_data = full %>% filter(is.na(Survived) ) 
test_passengersID = test_data[,c('PassengerId')]
test_input = test_data[,cols]


# Predict on test
pred = as.data.frame(predict(mod,newdata = test_input))


# write to csv
colnames(pred) = c("Survived")
write.csv(cbind(test_passengersID,pred),'output.csv', quote = FALSE, row.names = FALSE)
