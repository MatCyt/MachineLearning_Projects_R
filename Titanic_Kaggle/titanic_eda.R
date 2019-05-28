# TITANIC EDA

# DATA EXPLORATION --------------------------------------------------------

str(full)
summary(full)

View(full)

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