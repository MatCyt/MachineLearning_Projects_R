### CREDIT DEFAULT PREDICTION

# Dataset and data description: https://www.kaggle.com/wendykan/lending-club-loan-data
# Aim of the following script is to preprocess data and build ML based model for detecting default and non-default loans.



# 1. ENVIRONMENT ----
# libraries 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(readr, dplyr, caret, DescTools, corrplot, stringr)

# dataset 
full_load = read_csv("./data/loan.csv", guess_max = 1000000) 

problems(full_load) # no loading dataset problems (thanks to increasing the guess_max for proper column identification)



### 2. DATA EXPLORATION ----
# Full description of the variables in LCDataDictionary

## 2.1 Summary and structure ----
full = full_load # keeping the original loaded file name as a separate variable to avoid re-loading the file
attach(full)

dim(full)
str(full)
summary(full)

View(head(full, 50))

## 2.1.1 Extract Sample of the full dataset ----
# Since the size of the original dataset is significant we will cut the sample of the dataset to save the time
set.seed(123)
smp_size = floor(0.25 * nrow(full)) # adjust the sample size - the lower the shorter models will run
smaller_sample = sample(1:nrow(full), size = smp_size)

full = full[smaller_sample, ]


## 2.2 Simple feature selection ----
# to decrease the file size and ease later work. 
# We have 145 features which is a number we need to cut down - to select only the meaningful variables and cut the noise

# Drop meaningless columns - only one unique value or empty
distinct_col = lapply(full, function(x) length(unique(x)))
distinct_col[distinct_col == 1]

full = full %>%
  select(-id, - member_id, -url, - policy_code)

# Drop some of the categorical values
# Mostly those that are impossible or very time consuming to get value out of at the moment
caterogical_full = select_if(full, is.character)
categorical_distinct = sapply(caterogical_full, function(x) length(unique(x))) 

categorical_distinct # distinct categorical values per column
# in some cases we have thousands of different possible values (levels)
# they will most probably add the noise, signigicantly increase the computing time and won't provide a good distinction.

names(categorical_distinct[categorical_distinct > 30])

full = full %>%
  select(-emp_title, -desc, -title, -zip_code, -earliest_cr_line, -next_pymnt_d, -last_pymnt_d,
         -last_credit_pull_d, -sec_app_earliest_cr_line, debt_settlement_flag_date, -settlement_date)
    
# Eliminate Features from future
# We have here variable that is describing "future state" what happened after someone defaulted

full = full %>%
  select(-recoveries, -collection_recovery_fee)

# Eliminate features with high percentage of missing values
na_percentage = as.data.frame(colMeans(is.na(full)))
colnames(na_percentage) = "na_share"

na_percentage = na_percentage %>%
  mutate(variable = rownames(na_percentage)) %>%
  select(variable, na_share) %>%
  arrange(na_share)

passed_columns = na_percentage[na_percentage$na_share <= 0.1 , "variable"]

full = full[,passed_columns]


## 2.3 Target variable - exploration and processing ----

# Target variable levels
unique(full$loan_status)

# % distribution of classes inside the target variable
as.data.frame(table(full$loan_status))
as.data.frame(prop.table(table(full$loan_status)) * 100)

# New target variable based on the definition of defaulf provided | BINARY
full = full %>%
  mutate(default_binary = NA,
         default_binary = replace(default_binary, loan_status == "Charged Off", "Default"),
         default_binary = replace(default_binary, loan_status == "Default", "Default"),
         default_binary = replace(default_binary, loan_status == "Does not meet the credit policy. Status:Charged Off", "Default"),
         default_binary = replace(default_binary, loan_status == "Late (31-120 days)", "Default"),
         default_binary = replace(default_binary, loan_status == "Current", "NonDefault"),
         default_binary = replace(default_binary, loan_status == "Does not meet the credit policy. Status:Fully Paid", "NonDefault"),
         default_binary = replace(default_binary, loan_status == "Fully Paid", "NonDefault"),
         default_binary = replace(default_binary, loan_status == "Late (16-30 days)", "NonDefault"),
         default_binary = replace(default_binary, loan_status == "In Grace Period", "NonDefault"),
         default_binary = as.factor(default_binary)
  )

full = full %>%
  select(-loan_status)

# New proportions of target variables
as.data.frame(prop.table(table(full$default_binary)) * 100)

ggplot(as.data.frame(prop.table(table(full$default_binary)) * 100), aes(fill=Var1, y=Freq, x = 0)) + 
  geom_bar( stat="identity") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  scale_y_continuous(name = "% share", breaks = seq(10,100, by = 10))


## 2.4 Data Visualization - Variables of interest

# Main variables of interest at early stage : credit score grade and subgrade, home ownership, 
# annual income, interest rate, reason to apply, lenght of employment, region/adress

# Target variable exploration
# status comparison: default vs nondefault
Desc(full$default_binary, plotit = T)


# status by annual income
Desc(full$annual_inc, plotit = T)

featurePlot(x = full$annual_inc[full$annual_inc < 500000],
            y = as.factor(full$default_binary[full$annual_inc < 500000]),
            plot = "box")


# status by interest rate
Desc(full$int_rate, plotit = T)

featurePlot(x = full$int_rate,
            y = as.factor(full$default_binary),
            plot = "box")


# status by grade - credit score
s_by_g = as.data.frame(prop.table(table(full$default_binary, full$grade)) * 100) 

ggplot(s_by_g , aes(x = Var2 , y = Freq , fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  theme_minimal()


# status by home ownership
s_by_ho = as.data.frame(prop.table(table(full$default_binary, full$home_ownership)) * 100) 

ggplot(s_by_ho , aes(x = Var2 , y = Freq , fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  theme_minimal()


# loan status by reasons to apply 
s_by_p = as.data.frame(prop.table(table(full$default_binary, full$purpose)) * 100) 

ggplot(s_by_p, aes(x = Var2 , y = Freq , fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  theme(axis.text.x = element_text(angle = 90))


# loan status by verification
s_by_v = as.data.frame(prop.table(table(full$default_binary, full$verification_status)) * 100) 

ggplot(s_by_v, aes(x = Var2 , y = Freq , fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  theme_minimal()


# loan amount by application type
app_amn = full %>%
  group_by(purpose) %>%
  summarize(total = sum(loan_amnt)) %>%
  arrange(total)

ggplot(data=app_amn, aes(x=purpose, y=total)) +
  geom_bar(stat="identity", fill="steelblue") +
  coord_flip()


# Status by issued loan date
full = full %>%
  mutate(issue_year = str_sub(issue_d, -4)) %>%
  select(-issue_d)

prop.table(table(full$issue_year, full$default_binary), 1) * 100 # increasing nonDefault over time


### 3. FEATURE PREPROCESSING AND SELECTION ----

# 3.1 Deleting intercorrelated numerical variables ----
full_numerical = select_if(full, is.numeric)
full_numerical$default_binary = ifelse(full$default_binary == "Default", 1, 0)

cor_full = cor(full_numerical, use = "complete.obs")

# find variables with correlation value above 0.7
high_cor = findCorrelation(cor_full, cutoff=0.8) # putt any value as a "cutoff" 
high_cor = sort(high_cor)

# Detect highly intercorrelated variables - manually leave one of the pair/group and leave others
View(cor_full[ ,c(high_cor)])

full = full %>%
  select(-funded_amnt_inv, -total_pymnt_inv, -out_prncp_inv, -total_rec_prncp, -num_op_rev_tl, -bc_util, -percent_bc_gt_75,
         -total_bal_ex_mort, -bc_open_to_buy, -total_rev_hi_lim, -num_actv_bc_tl, -open_acc_6m, -open_acc, -tot_hi_cred_lim,
         -total_bc_limit, -num_rev_tl_bal_gt_0, -num_op_rev_tl, -acc_now_delinq, -avg_cur_bal, -num_bc_tl)

# Detect and drop numerical variables with marginal correlation with target variables
full_numerical2 = select_if(full, is.numeric)
full_numerical2$default_binary = ifelse(full$default_binary == "Default", 1, 0)

target_cor = as.data.frame(cor(full_numerical2, use = "complete.obs"))

drop_num_var = target_cor %>%
  filter(default_binary > -0.05 & default_binary < 0.05) %>%
  select(rownames)

drop_num_var = as.vector(drop_num_var$rownames)

full = full %>% select(-one_of(drop_num_var))


## 3.2 Missing values ----

# Variables with high level of missing values were already deleted.
# Categorical variables as shown below do not have any missing values

# Regarding the missing numerical values at this stage, due to the file size they are going to be deleted
# Ideally they should be imputed using on model based approach by for example mice package but
# this is not possible with my available resources. Right now the aim is to decrease the file size.


# No Missing Values in Categorical Columns
categorical_full = select_if(full, is.character)
as.data.frame(colSums(is.na(categorical_full)))

# Numerical variables
full = na.omit(full)
colSums(is.na(full))

# Median for numerical missing variables - maybe in future
# full = data.frame(lapply(full,function(x) {
# if(is.numeric(x)) ifelse(is.na(x),median(x,na.rm=T),x) else x}))

## Categorical Variables

# Drop state - lots of levels and no obvious differences besides single states
state = as.data.frame(prop.table(table(full$default_binary, full$addr_state)) * 100) 
ggplot(state, aes(x = Var2 , y = Freq , fill = Var1)) + 
  geom_bar(stat = "identity", position = "fill") +
  theme_minimal()

full = full %>%
  select(-addr_state)

# 3.3 Feature transformation - factors ----
full = as.data.frame(unclass(full))


### 4. EVALUATE ALGORITHMS ----

# 4.1 Train-Test split 70-30 and cross validation ----
set.seed(42)
split_index = createDataPartition(full$default_binary, p=0.7, list = F)

train = full[split_index,]
test = full[-split_index,]

# specify cross validation (for model comparison and tuning) and metric
myControl = trainControl(method = "cv", 
                         number = 5, # 5 due file size and computational cost
                         classProbs = T, # binary class problem
                         summaryFunction = twoClassSummary)
metric = "ROC"



## 4.2 Compare performance of several algorithms ----
# set 2 basic different default values for main parameter - limited by computational resources

# linear regression
glm.model = train(default_binary~., 
                  data=train, 
                  method="glm", 
                  metric=metric,
                  tuneLength = 2,
                  trControl=myControl)

print(glm.model)
plot(glm.model)

# decision tree 
dt.model = train(default_binary~., 
                 data=train, 
                 method="rpart", 
                 metric=metric,
                 tuneLength = 2,
                 trControl=myControl)

print(dt.model)
plot(dt.model)

# SVM
svm.model = train(default_binary~., 
                  data=train, 
                  method="svmRadial", 
                  metric=metric,
                  tuneLength = 2,
                  trControl=myControl)

print(svm.model)
plot(svm.model)

# Random Forest
rf.model = train(default_binary~., 
                 data=train,
                 method="rf", 
                 metric=metric,
                 tuneLength = 2,
                 trControl=myControl)

print(rf.model)
plot(rf.model)

# XGBoost
xgb_model = train(default_binary~.,
                  data=train,
                  metric = metric, 
                  method = "xgbTree",
                  tuneLength = 2,
                  trControl = myControl)

xgb.model = xgb_model
print(xgb_model)
plot(xgb_model)

# Compare models with resample

results = resamples(list(glm=glm.model, dt=dt.model, svm=svm.model, rf=rf.model, xgb=xgb.model))

summary(results)
dotplot(results)


# In my selected sample Random Forest achieved highest result based on the ROC result.

### 5. Possible tuning ----

# Grid search - 10 values
tunegrid = expand.grid(.mtry=c(1:10))

# Random Search - 10 values + setting the tune lenght to 10
finalcontrol = trainControl(method='repeatedcv', 
                            number=10, 
                            repeats=3,
                            search = 'random')

### 6. PREDICT ----

# predict
predictions = predict(rf.model, test)
confusionMatrix(predictions, test$default_binary)

# Results for the 10% sample of the full dataset
# 0.98 of accuracy and 0.99 precision and 0.9 recall
# It seems suspiciously high.

# save the model
saveRDS(rf.model, "rf.model.rds")

