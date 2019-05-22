# TODO explore Survival by group

full %>%
  group_by(Pclass,Sex) %>%
  summarise(Survived = sum(Survived, na.rm = T))

# TODO work with cabin variable

# TODO visualize using caret 

# TODO % share of class in each feature 

# TODO compare running the model with imputing NAs for the two small variables versus keeping them

# TODO better age value imputation


# TODO what happens with character variables if not converted - compare strings and not


# TODO linear regression for Age prediction OR decision tree


# TODO Survival % by age - not simply child < 18