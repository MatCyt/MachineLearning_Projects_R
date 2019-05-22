# Titanic Practice - Kaggle - Exploring the Titanic Dataset
# https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic

# Load packages ----
library('ggplot2') # visualization
library('ggthemes') # visualization
library('scales') # visualization
library('dplyr') # data manipulation
library('mice') # imputation
library('randomForest') # classification algorithm


# data ----
train = read_csv('./Titanic/data/train.csv')
test  = read_csv('./Titanic/data/test.csv')

full  = bind_rows(train, test) # bind training & test data

# check data
str(full)


# Feature Engineering ----

# Name - Title
# Grab title from passenger names
full$Title = gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)

# Titles with very low cell counts to be combined to "rare" level
rare_title = c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
full$Title[full$Title == 'Mlle']        = 'Miss' 
full$Title[full$Title == 'Ms']          = 'Miss'
full$Title[full$Title == 'Mme']         = 'Mrs' 
full$Title[full$Title %in% rare_title]  = 'Rare Title'

# Show title counts by sex again
table(full$Sex, full$Title)

# Finally, grab surname from passenger name
full$Surname = sapply(full$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])


# Families 

# Create a family size variable including the passenger themselves
full$Fsize = full$SibSp + full$Parch + 1

# Create a family variable 
full$Family = paste(full$Surname, full$Fsize, sep='_')

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()


# We can see that there’s a survival penalty to singletons and those with family sizes above 4. 
# We can collapse this variable into three levels which will be helpful since 
# there are comparatively fewer large families.

# Discretize family size

full$FsizeD = NULL
full$FsizeD[full$Fsize == 1] = 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] = 'small'
full$FsizeD[full$Fsize > 4] = 'large'

# Show family size by survival using a mosaic plot
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)


# Passenger Cabin
full$Deck = factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))


# Missing Values Imputation ----

# Embarkment 
# Passengers 62 and 830 are missing Embarkment
full[c(62, 830), 'Embarked']

# Get rid of our missing passenger IDs
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

# The median fare for a first class passenger departing from Charbourg (‘C’) 
# coincides nicely with the $80 paid by our embarkment-deficient passengers. 
# I think we can safely replace the NA values with ‘C’.

# Since their fare was $80 for 1st class, they most likely embarked from 'C'
full$Embarked[c(62, 830)] = 'C'

# Remaining passenger
# Show row 1044
full[1044, ]

# This is a third class passenger who departed from Southampton (‘S’). 
# Let’s visualize Fares among all others sharing their class and embarkment (n = 494).
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()

# Replace missing fare value with median fare for class/embarkment
full$Fare[1044] = median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)


# Age imputation with predictive model


# Show number of missing Age values
sum(is.na(full$Age))

# Using mice package

# Factorize variables
# Make variables factors into factors
factor_vars = c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')
full[factor_vars] = lapply(full[factor_vars], function(x) as.factor(x))

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod = mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 

# Save the complete output 
mice_output = complete(mice_mod)

# Let’s compare the results we get with the original distribution of passenger ages to ensure that nothing has gone completely awry.
# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Replace Age variable from the mice model.
full$Age = mice_output$Age

# Show new number of missing Age values
sum(is.na(full$Age))


# Adding features - child and mother ----
# Create the column child, and indicate whether child or adult
full$Child[full$Age < 18] = 'Child'
full$Child[full$Age >= 18] = 'Adult'

# Show counts
table(full$Child, full$Survived)

# Adding Mother variable
full$Mother = 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] = 'Mother'

# Show counts
table(full$Mother, full$Survived)

# Finish by factorizing our two new factor variables
full$Child  = factor(full$Child)
full$Mother = factor(full$Mother)


# Modeling - Random Forest ----

# Split the data back into a train set and a test set
train = full[1:891,]
test = full[892:1309,]

# Set a random seed
set.seed(754)

# Build the model (note: not all possible variables are used)
rf_model = randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           FsizeD + Child + Mother,
                         data = train)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


# Get importance
importance    = importance(rf_model)
varImportance = data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance = varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()


# Predict using the test set
prediction = predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution = data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
