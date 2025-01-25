library(ggplot2)
library(mice)
library(dplyr)
library(VIM)
library(fastDummies)
library(ggcorrplot)
library(tidyr)
library(naniar)
library(gridExtra)
library(missForest)
library(Boruta)
library(caret)
library(car)
library(rpart)
library(rpart.plot)
library(caret)
library(class)
library(randomForest)
library(fastshap)
library(pROC)
library(stats)
library(plotly)


#reading file
loan.df<-read.csv("Loan.csv")
#observing first rows
head(loan.df,100)

#======================================
# Phase 1: Data Preprocessing
#======================================
#checking structure
str(loan.df)

# Rename the column BAD to default
colnames(loan.df)[colnames(loan.df) == "BAD"] <- "DEFAULT"
colnames(loan.df)

#rechecking structure to observe change
str(loan.df)

#summary statistics
summary(loan.df)

#-------------------------------------
# 1.1 Missing Values
#-------------------------------------

#======= 1.1.1 Exploration of Missing Values=======#

#number of NA values accross variables
colSums(is.na(loan.df))
#number of blank values accross columns
colSums(loan.df == "")

#converting blanks into NA
loan.df[loan.df == ""] <- NA

#converting data types 
loan.df$DEFAULT <- as.factor(loan.df$DEFAULT)
loan.df$REASON <- as.factor(loan.df$REASON)
loan.df$JOB <- as.factor(loan.df$JOB)

#reobserving structure
str(loan.df)

#percentage of missing values for each column
na_percentages <- colSums(is.na(loan.df)) / nrow(loan.df) * 100

# Convert to a data frame for easier plotting
na_df <- data.frame(Variable = names(na_percentages), 
                    Percent_NA = na_percentages)

colSums(is.na(loan.df))

#Bar plot of NA's
ggplot(na_df, aes(x = Variable, y = Percent_NA)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(round(Percent_NA, 1), "%")), vjust = -0.3) +
  labs(title = "Percentage of Missing Values by Variable",
       x = "Variable",
       y = "Percentage of Missing Values (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Count how many missing values each record has
loan.df$missing_count <- rowSums(is.na(loan.df))

# Create a frequency table of how many records have a specific number of missing values
missing_summary <- as.data.frame(table(loan.df$missing_count))

# Rename columns for clarity
colnames(missing_summary) <- c("num_missing_variables", "count_of_records")

# Convert the number of missing variables to numeric for plotting
missing_summary$num_missing_variables <- as.numeric(as.character(missing_summary$num_missing_variables))

# Calculate the proportion of records with each number of missing variables
missing_summary$proportion <- missing_summary$count_of_records / nrow(loan.df) * 100

# Plot the results 
ggplot(missing_summary, aes(x = num_missing_variables, y = proportion)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(round(proportion, 1), "%")), 
            vjust = -0.5, color = "black", size = 3.5) +  # Add percentage on top of each bar
  labs(title = "Proportion of Records by Number of Missing Variables",
       x = "Number of Missing Variables",
       y = "Proportion of Records (%)") +
  scale_x_continuous(breaks = seq(min(missing_summary$num_missing_variables), 
                                  max(missing_summary$num_missing_variables), 
                                  by = 1)) +  # Ensure x-axis shows only integers
  theme_minimal()

#=======1.1.2 Removing Rows with Excessive Missing Data =========#

#------Observing a dataset with only complete cases --------#

# Create a subset of the dataset with complete cases (rows without missing values)
loan_complete <- loan.df[complete.cases(loan.df), ]
loan_complete$missing_count<-NULL

# View the subset with complete cases
head(loan_complete)  # Show the first few rows of the complete cases

#summary
summary(loan_complete)

#---------- Filtering original dataset to only rows with less than 4 NAs---------#

# Subset the dataset to keep rows with less than 4 missing variables
loan_subset <- loan.df[loan.df$missing_count < 4, ]
str(loan_subset)

# Remove the missing_count column from the new subset
loan_subset$missing_count <- NULL

# Confirming structure of subsetted dataset
str(loan_subset)

#Summary Statistics
summary(loan_subset)

#percentage of missing values for each column
na_percentages <- colSums(is.na(loan_subset)) / nrow(loan.df) * 100

# Convert to a data frame for easier plotting
na_df <- data.frame(Variable = names(na_percentages), 
                    Percent_NA = na_percentages)

#Bar plot of NA's
ggplot(na_df, aes(x = Variable, y = Percent_NA)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(round(Percent_NA, 1), "%")), vjust = -0.3) +
  labs(title = "Percentage of Missing Values by Variable",
       x = "Variable",
       y = "Percentage of Missing Values (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#=======1.1.3 Diagnosing Missing Data Mechanisms =========#

#-----Correlation of Missing Values-----#

#Binary matrix where 1 indicates missing and 0 indicates observed
missing_matrix <- as.data.frame(sapply(loan_subset, function(x) as.numeric(is.na(x))))

# Remove columns with no missing values, as these cause zero variance
missing_matrix <- missing_matrix[, colSums(missing_matrix) > 0]

# Calculate the correlation matrix of missingness
missing_corr <- cor(missing_matrix)

# Create a correlation heatmap
ggcorrplot(missing_corr, 
           hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           title = "Correlation Heatmap of Missing Values")


#----percentage of missing values by default status----#

# Calculate percentage of missing values for each group (DEFAULT = 1 and DEFAULT = 0)
missing_pct_default_1 <- colMeans(is.na(loan_subset[loan_subset$DEFAULT == 1, ])) * 100
missing_pct_default_0 <- colMeans(is.na(loan_subset[loan_subset$DEFAULT == 0, ])) * 100

# Combine the results into a data frame
missing_pct_df <- data.frame(
  Variable = names(loan_subset),
  Missing_Pct_Default_1 = missing_pct_default_1,
  Missing_Pct_Default_0 = missing_pct_default_0
)

# View the result
missing_pct_df

# Manually reshape the data to long format without using pivot_longer
missing_pct_df_long <- data.frame(
  Variable = rep(missing_pct_df$Variable, 2),
  Default_Status = rep(c("Default = 1", "Default = 0"), each = nrow(missing_pct_df)),
  Missing_Pct = c(missing_pct_df$Missing_Pct_Default_1, missing_pct_df$Missing_Pct_Default_0)
)

# Create the plot and include percentage labels on top of each bar
ggplot(missing_pct_df_long, aes(x = Variable, y = Missing_Pct, fill = Default_Status)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.1f%%", Missing_Pct)), 
            position = position_dodge(width = 0.9), vjust = -0.5) +  # Add percentage labels
  scale_fill_manual(values = c("Default = 1" = "red", "Default = 0" = "blue")) +  # Set colors manually
  theme_minimal() +
  labs(title = "Missing Data Percentage by Default Status",
       x = "Variable",
       y = "Missing Percentage (%)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels

#-------MCAR TEST--------#

mcar_result<-mcar_test(loan_subset)
print(mcar_result)

#--------MAR TEST PERFORMING LOGISTIC REGRESSION--------#

#creating duplicate dataset 
loan.df2<-loan_subset

#1.DEBTINC

# Create the is_missing column for DEBTINC
loan.df2$is_missing <- ifelse(is.na(loan.df2$DEBTINC), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and DEBTINC
variables <- setdiff(names(loan.df2), c("is_missing", "DEBTINC"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model (R will automatically exclude rows with missing values)
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p1<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'DEBTINC'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))

#-----------------

#2.DEROG 

# Create the is_missing column for DEROG
loan.df2$is_missing <- ifelse(is.na(loan.df2$DEROG), 1, 0)

# Check the updated dataset
head(loan.df2,30)

# List of variables excluding is_missing and DEROG
variables <- setdiff(names(loan.df2), c("is_missing", "DEROG"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p2<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'DEROG'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))

#--------------

#3.MORTDUE 

# Create the is_missing column for  MORTDUE
loan.df2$is_missing <- ifelse(is.na(loan.df2$MORTDUE), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and MORTDUE
variables <- setdiff(names(loan.df2), c("is_missing", "MORTDUE"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p3<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'MORTDUE'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#-------------------

#4. YOJ 

# Create the is_missing column for YOJ
loan.df2$is_missing <- ifelse(is.na(loan.df2$YOJ), 1, 0)

# Check the updated dataset
head(loan.df2)


# List of variables excluding is_missing and YOJ
variables <- setdiff(names(loan.df2), c("is_missing", "YOJ"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p4<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'YOJ'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#------------

#5.DELINQ

# Create the is_missing column for DELINQ
loan.df2$is_missing <- ifelse(is.na(loan.df2$DELINQ), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and DELINQ
variables <- setdiff(names(loan.df2), c("is_missing", "DELINQ"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p5<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'DELINQ'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#-----------------

# 6. NINQ 

# Create the is_missing column for NINQ
loan.df2$is_missing <- ifelse(is.na(loan.df2$NINQ), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and NINQ
variables <- setdiff(names(loan.df2), c("is_missing", "NINQ"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p6<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'NINQ'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#------------------

#7.REASON 

# Create the is_missing column for REASON
loan.df2$is_missing <- ifelse(is.na(loan.df2$REASON), 1, 0)

# Check the updated dataset
head(loan.df2)


# List of variables excluding is_missing and REASON
variables <- setdiff(names(loan.df2), c("is_missing", "REASON"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p7<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'REASON'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#------------------

#8. JOB 

# Create the is_missing column for JOB
loan.df2$is_missing <- ifelse(is.na(loan.df2$JOB), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and JOB
variables <- setdiff(names(loan.df2), c("is_missing", "JOB"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p8<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'JOB'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#-------------------

#9. VALUE 

# Create the is_missing column for VALUE
loan.df2$is_missing <- ifelse(is.na(loan.df2$VALUE), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and VALUE
variables <- setdiff(names(loan.df2), c("is_missing", "VALUE"))

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p9<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'VALUE'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))
#---------------------

#10. CLAGE 

# Create the is_missing column for CLAGE
loan.df2$is_missing <- ifelse(is.na(loan.df2$CLAGE), 1, 0)

# Check the updated dataset
head(loan.df2)

# List of variables excluding is_missing and CLAGE
variables <- setdiff(names(loan.df2), c("is_missing", "CLAGE","DELINQ")) #excluded predictor DELINQ due to perfect separation warning
variables

# Initialize an empty data frame to store p-values and variable names
p_values_df <- data.frame(variable = character(), p_value = numeric(), stringsAsFactors = FALSE)

# Loop through each predictor
for (variable in variables) {
  # Create the logistic regression formula
  formula <- as.formula(paste("is_missing ~", variable))
  
  # Fit the logistic regression model 
  model <- glm(formula, family = binomial, data = loan.df2)
  
  # Extract the p-value for the predictor variable
  p_value <- summary(model)$coefficients[2, 4]
  
  # Add the variable name and p-value to the data frame
  p_values_df <- rbind(p_values_df, data.frame(variable = variable, p_value = p_value))
}

# Add a column to indicate statistical significance
p_values_df$significance <- ifelse(p_values_df$p_value < 0.05, "Significant", "Not Significant")

# View the result
p_values_df

p10<-ggplot(p_values_df, aes(x = reorder(variable, -log10(p_value)), y = -log10(p_value), fill = significance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_manual(values = c( "Not Significant" = "grey","Significant" = "blue")) +
  labs(title = "Significance of Variables in Predicting Missing Values for 'CLAGE'", x = "Variable", y = "-log10(p-value)") +
  theme_minimal() +
  guides(fill = guide_legend(title = NULL))

#PLOTTING RESULTS 

#Putting all plots together
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, ncol = 2)

#========== 1.1.3 Evaluating Imputation Methods===========#

#The goal is to insert artificially generated missingness to the complete dataset
#this way we can test imputation methods performance

# Create a copy of loan_complete
loan_incomplete <- loan_complete
head(loan_incomplete)
str(loan_incomplete)

set.seed(123)
# excluding columns from missingness since they dont need imputation (CLNO, DEFAULT, LOAN)
excluded_columns <- c("CLNO", "DEFAULT", "LOAN")

#  names of the columns that are allowed to have missing values
columns_to_modify <- setdiff(names(loan_incomplete), excluded_columns)

# Introduce 20% missingness for each allowed column
for (col in columns_to_modify) {
  missing_indices <- sample(1:nrow(loan_incomplete), size = floor(0.2 * nrow(loan_incomplete)))
  loan_incomplete[missing_indices, col] <- NA
}

# Display summary of missing data to verify 20% missingness
summary(loan_incomplete)
str(loan_incomplete)

#-------- RF MICE---------#

set.seed(123)
mice_imputed <- mice(loan_incomplete, m = 5, method = 'rf', maxit = 25)


# Function to calculate MSE and RMSE for a specific variable
calculate_mse_rmse <- function(variable_name, loan_complete, loan_incomplete, mice_imputed) {
  mse_values <- numeric()
  rmse_values <- numeric()
  
  # Step 1: Identify rows where the variable was missing
  missing_rows <- is.na(loan_incomplete[[variable_name]])
  
  # Loop through each imputed dataset and calculate MSE and RMSE
  for (i in 1:5) {
    # Step 2: Extract the completed data for the imputation
    imputed_data <- complete(mice_imputed, i)
    
    # Step 3: Subset the imputed values for the variable
    imputed_values <- imputed_data[missing_rows, variable_name]
    
    # Step 4: Extract the corresponding original values
    original_values <- loan_complete[missing_rows, variable_name]
    
    # Step 5: Calculate MSE for this imputation
    mse <- mean((original_values - imputed_values)^2, na.rm = TRUE)
    
    # Calculate RMSE as the square root of MSE
    rmse <- sqrt(mse)
    
    # Store the MSE and RMSE values
    mse_values[i] <- mse
    rmse_values[i] <- rmse
  }
  
  # Step 6: Calculate and return the average MSE and RMSE across all 5 imputations
  avg_mse <- mean(mse_values)
  avg_rmse <- mean(rmse_values)
  
  return(list(avg_mse = avg_mse, avg_rmse = avg_rmse))
}

# Variables to calculate MSE and RMSE for
variables <- c("DEBTINC", "YOJ", "MORTDUE", "VALUE", "DEROG", "DELINQ", "CLAGE", "NINQ")

# Loop through each variable and print the results
for (var in variables) {
  results <- calculate_mse_rmse(var, loan_complete, loan_incomplete, mice_imputed)
  print(paste("Average MSE for", var, "across all imputations:", results$avg_mse))
  print(paste("Average RMSE for", var, "across all imputations:", results$avg_rmse))
}

#-------PMM MICE--------#

set.seed(123)
mice_imputed <- mice(loan_incomplete, m = 5, method = 'pmm', maxit = 25)

# Function to calculate MSE and RMSE for a specific variable
calculate_mse_rmse <- function(variable_name, loan_complete, loan_incomplete, mice_imputed) {
  mse_values <- numeric()
  rmse_values <- numeric()
  
  # Step 1: Identify rows where the variable was missing
  missing_rows <- is.na(loan_incomplete[[variable_name]])
  
  # Loop through each imputed dataset and calculate MSE and RMSE
  for (i in 1:5) {
    # Step 2: Extract the completed data for the i-th imputation
    imputed_data <- complete(mice_imputed, i)
    
    # Step 3: Subset the imputed values for the variable
    imputed_values <- imputed_data[missing_rows, variable_name]
    
    # Step 4: Extract the corresponding original values
    original_values <- loan_complete[missing_rows, variable_name]
    
    # Step 5: Calculate MSE for this imputation
    mse <- mean((original_values - imputed_values)^2, na.rm = TRUE)
    
    # Calculate RMSE as the square root of MSE
    rmse <- sqrt(mse)
    
    # Store the MSE and RMSE values
    mse_values[i] <- mse
    rmse_values[i] <- rmse
  }
  
  # Step 6: Calculate and return the average MSE and RMSE across all 5 imputations
  avg_mse <- mean(mse_values)
  avg_rmse <- mean(rmse_values)
  
  return(list(avg_mse = avg_mse, avg_rmse = avg_rmse))
}

# Variables to calculate MSE and RMSE for
variables <- c("DEBTINC", "YOJ", "MORTDUE", "VALUE", "DEROG", "DELINQ", "CLAGE", "NINQ")

# Loop through each variable and print the results
for (var in variables) {
  results <- calculate_mse_rmse(var, loan_complete, loan_incomplete, mice_imputed)
  print(paste("Average MSE for", var, "across all imputations:", results$avg_mse))
  print(paste("Average RMSE for", var, "across all imputations:", results$avg_rmse))
}

#---------MISS FOREST---------#

set.seed(123)

#Applying Missforest
missforest_imputed <- missForest(loan_incomplete)

# Extracting the imputed data
imputed_data_forest <- missforest_imputed$ximp
str(imputed_data_forest)
# MSE and RMSE for each variable 
calculate_mse_rmse_forest <- function(variable_name, loan_complete, loan_incomplete, imputed_data_forest) {
  mse_values <- numeric()
  rmse_values <- numeric()
  
  # Step 1: Identify rows where the variable was missing
  missing_rows <- is.na(loan_incomplete[[variable_name]])
  
  # Step 2: Subset the imputed values for the variable
  imputed_values <- imputed_data_forest[missing_rows, variable_name]
  
  # Step 3: Extract the corresponding original values
  original_values <- loan_complete[missing_rows, variable_name]
  
  # Step 4: Calculate MSE and RMSE
  mse <- mean((original_values - imputed_values)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  
  return(list(mse = mse, rmse = rmse))
}

# Variables to evaluate
variables <- c("DEBTINC", "YOJ", "MORTDUE", "VALUE", "DEROG", "DELINQ", "CLAGE", "NINQ")

# Loop through each variable and print the results
for (var in variables) {
  results <- calculate_mse_rmse_forest(var, loan_complete, loan_incomplete, imputed_data_forest)
  print(paste("MSE for", var, "using MissForest:", results$mse))
  print(paste("RMSE for", var, "using MissForest:", results$rmse))
}

#--------KNN IMPUTATION--------#

# Scaling the numeric variables
numeric_vars <- c("DEBTINC", "YOJ", "MORTDUE", "VALUE", "DEROG", "DELINQ", "CLAGE", "NINQ")
scaled_data_to_impute <- loan_incomplete
scaled_data_to_impute[numeric_vars] <- scale(loan_incomplete[numeric_vars])

# KNN Imputation on the scaled data
set.seed(123)  
knn_imputed_data_scaled <- kNN(scaled_data_to_impute, k = 5)

# Reverse the scaling for the imputed data to match the original scale
knn_imputed_data <- knn_imputed_data_scaled
knn_imputed_data[numeric_vars] <- lapply(numeric_vars, function(var) {
  # Reverse scaling based on the mean and sd of the original data
  mean_orig <- mean(loan_incomplete[[var]], na.rm = TRUE)
  sd_orig <- sd(loan_incomplete[[var]], na.rm = TRUE)
  knn_imputed_data_scaled[[var]] * sd_orig + mean_orig
})

# Calculate MSE and RMSE for each variable 
calculate_mse_rmse_knn <- function(variable_name, loan_complete, loan_incomplete, knn_imputed_data) {
  mse_values <- numeric()
  rmse_values <- numeric()
  
  # Step 1: Identify rows where the variable was missing
  missing_rows <- is.na(loan_incomplete[[variable_name]])
  
  # Step 2: Subset the imputed values for the variable
  imputed_values <- knn_imputed_data[missing_rows, variable_name]
  
  # Step 3: Extract the corresponding original values
  original_values <- loan_complete[missing_rows, variable_name]
  
  # Step 4: Calculate MSE and RMSE
  mse <- mean((original_values - imputed_values)^2, na.rm = TRUE)
  rmse <- sqrt(mse)
  
  return(list(mse = mse, rmse = rmse))
}

# Variables to evaluate
variables <- c("DEBTINC", "YOJ", "MORTDUE", "VALUE", "DEROG", "DELINQ", "CLAGE", "NINQ")

# Loop through each variable and print the results
for (var in variables) {
  results <- calculate_mse_rmse_knn(var, loan_complete, loan_incomplete, knn_imputed_data)
  print(paste("MSE for", var, "using KNN Imputation:", results$mse))
  print(paste("RMSE for", var, "using KNN Imputation:", results$rmse))
}

#------- FINAL IMPUTATION -----------#
str(loan_subset)

set.seed(123)
# Apply MissForest to the dataset
final_imputed_data <- missForest(loan_subset)
final_imputed_data

# The imputed data will be stored in imputed_data$ximp
imputed_loan <- final_imputed_data$ximp
str(final_imputed_data)

#checking first 20 rows 
head(imputed_loan,20)

# Round the imputed values of NINQ,DELINQ, and DEROG to the nearest whole number since they are originally integers
imputed_loan$NINQ <- round(imputed_loan$NINQ)
imputed_loan$DELINQ <- round(imputed_loan$DELINQ)
imputed_loan$DEROG <- round(imputed_loan$DEROG)

# Convert the back to integer type
imputed_loan$NINQ <- as.integer(imputed_loan$NINQ)
imputed_loan$DELINQ <- as.integer(imputed_loan$DELINQ)
imputed_loan$DEROG <- as.integer(imputed_loan$DEROG)

# View the modified imputed data
head(imputed_loan,100)
str(imputed_loan)

#obtaining summary stats of imputed dataset
summary(imputed_loan)

#==========================================
#Phase 2: Feature Engineering
#==========================================

#------------------------------------------
#2.1 EQUITY
#------------------------------------------

# Creating column equity, which is the difference between what client owes for mortgage 
#and market value of the house
imputed_loan$EQUITY <- imputed_loan$VALUE - imputed_loan$MORTDUE

# View the first few rows to confirm
head(imputed_loan)

#------------------------------------------
#2.2 UNSECURED LOAN AMOUNT
#------------------------------------------

# Calculate uncovered loan for both defaulters and non-defaulters
imputed_loan$UNSECURED_LOAN <- ifelse(imputed_loan$EQUITY < imputed_loan$LOAN, 
                                      pmin(imputed_loan$LOAN - imputed_loan$EQUITY, imputed_loan$LOAN), 
                                      0)

#------------------------------------------
#2.3 DEFINITIVE LOSS AMOUNT
#------------------------------------------

# Create a column for definitive loss for those who have defaulted
imputed_loan$DEFINITIVE_LOSS <- ifelse(imputed_loan$DEFAULT == 1 & imputed_loan$EQUITY < imputed_loan$LOAN, 
                                   imputed_loan$LOAN - pmax(imputed_loan$EQUITY, 0), 
                                   0)

#------------------------------------------
#2.4 LTE RATIO
#------------------------------------------

summary(imputed_loan)
#Creating Loan to Equity Ratio column in dataset

# Checking how many rows have equity equal to 0 first, since division by 0 may be an issue
sum(imputed_loan$EQUITY == 0)

#Since there are no 0's, we proceed by:
imputed_loan$LTE_RATIO <- imputed_loan$LOAN / imputed_loan$EQUITY

#------------------------------------------
#2.5 YOJ TO LOAN
#------------------------------------------

# YOJ-to-Loan Ratio
imputed_loan$yoj_to_loan_ratio <- imputed_loan$YOJ / imputed_loan$LOAN

#------------------------------------------
#2.6 LTV RATIO
#------------------------------------------

# LTV Ratio
imputed_loan$LTV_RATIO <- imputed_loan$LOAN / imputed_loan$VALUE

#------------------------------------------
#2.7 OWNERSHIP
#------------------------------------------

# Ownership Ratio
imputed_loan$OWNERSHIP <- imputed_loan$EQUITY / imputed_loan$VALUE

#------------------------------------------
#2.8 CREDIT SCORE
#------------------------------------------

# Binning and assigning points for relevant variables

# For Payment History (DELINQ)
imputed_loan$points_payment_history <- ifelse(imputed_loan$DELINQ == 0, 50, 
                                              ifelse(imputed_loan$DELINQ == 1, 30, 10))

# For Debt-to-Income Ratio (DEBTINC)
imputed_loan$points_amounts_owed <- ifelse(imputed_loan$DEBTINC < 20, 50, 
                                           ifelse(imputed_loan$DEBTINC >= 20 & imputed_loan$DEBTINC <= 35, 30, 10))


# For Length of Credit History (CLAGE in months)
imputed_loan$points_length_of_credit <- ifelse(imputed_loan$CLAGE > 120, 50, 
                                               ifelse(imputed_loan$CLAGE >= 60 & imputed_loan$CLAGE <= 120, 30, 10))

# Create thresholds for Ownership Ratio with 20% as the lower threshold
imputed_loan$points_ownership <- ifelse(imputed_loan$OWNERSHIP >= 0.50, 50,
                                        ifelse(imputed_loan$OWNERSHIP >= 0.20 & imputed_loan$OWNERSHIP < 0.50, 30, 10))


# For New Credit Inquiries (NINQ)
imputed_loan$points_new_credit <- ifelse(imputed_loan$NINQ == 0, 50, 
                                         ifelse(imputed_loan$NINQ == 1, 30, 10))

# Now calculate the total score
imputed_loan$score <- (imputed_loan$points_payment_history * 0.35) + 
  (imputed_loan$points_amounts_owed * 0.30) + 
  (imputed_loan$points_length_of_credit * 0.15) + 
  (imputed_loan$points_ownership  * 0.10) + 
  (imputed_loan$points_new_credit * 0.10)

# Scale the score to the range of 300-850
imputed_loan$CREDIT_SCORE <- 300 + ((imputed_loan$score / max(imputed_loan$score)) * 550)

head(imputed_loan)

#------------------------------------------
#2.9 CREDIT SCORE CATEGORY
#------------------------------------------

# Define the score categories similar to FICO ranges
imputed_loan$SCORE_CATEGORY <- cut(imputed_loan$CREDIT_SCORE, 
                                   breaks = c(300, 579, 669, 739, 799, 850), 
                                   labels = c("Poor", "Fair", "Good", "Very Good", "Excellent"), 
                                   include.lowest = TRUE)

# List of columns to remove
columns_to_remove <- c("points_payment_history", "points_amounts_owed_debtinc", "points_amounts_owed_mortdue", 
                       "points_amounts_owed", "points_length_of_credit", "points_ownership", "points_new_credit", 
                       "score")

# Remove the specified columns
imputed_loan <- imputed_loan[, setdiff(names(imputed_loan), columns_to_remove)]

#checking dataset update
head(imputed_loan)

#==========================================
#Phase 3: Exploratory Data Analysis
#==========================================

#------------------------------------------
#3.1 Summary Statistics and Distributions
#------------------------------------------

#============3.1.1 Histograms of Original Variables=========#

# Creating histograms for selected numeric variables
p1 <- ggplot(imputed_loan, aes(x = LOAN)) +
  geom_histogram(binwidth = 1000, fill = "steelblue", color = "black") +
  labs(title = "Loan Amount", x = "Loan Amount", y = "Frequency") +
  theme_minimal()

p2 <- ggplot(imputed_loan, aes(x = MORTDUE)) +
  geom_histogram(binwidth = 5000, fill = "steelblue", color = "black") +
  labs(title = "Mortgage Due", x = "Mortgage Due", y = "Frequency") +
  theme_minimal()

p3 <- ggplot(imputed_loan, aes(x = VALUE)) +
  geom_histogram(binwidth = 20000, fill = "steelblue", color = "black") +
  labs(title = "Property Value", x = "Property Value", y = "Frequency") +
  theme_minimal()

p4 <- ggplot(imputed_loan, aes(x = DEBTINC)) +
  geom_histogram(binwidth = 2, fill = "steelblue", color = "black") +
  labs(title = "Debt-to-Income Ratio", x = "Debt-to-Income Ratio", y = "Frequency") +
  theme_minimal()

p5 <- ggplot(imputed_loan, aes(x = CLAGE)) +
  geom_histogram(binwidth = 20, fill = "steelblue", color = "black") +
  labs(title = "Credit Line Age", x = "Credit Line Age (Months)", y = "Frequency") +
  theme_minimal()

p6 <- ggplot(imputed_loan, aes(x = YOJ)) +
  geom_histogram(binwidth = 2, fill = "steelblue", color = "black") +
  labs(title = "Years on Job", x = "Years on Job", y = "Frequency") +
  theme_minimal()

p7 <- ggplot(imputed_loan, aes(x = DEROG)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  labs(title = "Derogatory Reports", x = "DEROG", y = "Frequency") +
  theme_minimal()

p8 <- ggplot(imputed_loan, aes(x = DELINQ)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  labs(title = "Delinquencies", x = "DELINQ", y = "Frequency") +
  theme_minimal()

p9 <- ggplot(imputed_loan, aes(x = NINQ)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  labs(title = "Number of Inquiries", x = "NINQ", y = "Frequency") +
  theme_minimal()

p10 <- ggplot(imputed_loan, aes(x = CLNO)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black") +
  labs(title = "Number of Credit Lines", x = "CLNO", y = "Frequency") +
  theme_minimal()

# Arranging all plots in a grid (3 columns)
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, p9,p10, ncol = 2)

#============3.1.2 Boxplots of Original Variables=========#

# Creating boxplots for numeric variables 
b1 <- ggplot(imputed_loan, aes(y = LOAN)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Loan Amount", y = "Loan Amount") +
  theme_minimal()

b2 <- ggplot(imputed_loan, aes(y = MORTDUE)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Mortgage Due", y = "Mortgage Due") +
  theme_minimal()

b3 <- ggplot(imputed_loan, aes(y = VALUE)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Property Value", y = "Property Value") +
  theme_minimal()

b4 <- ggplot(imputed_loan, aes(y = DEBTINC)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Debt-to-Income Ratio", y = "Debt-to-Income Ratio") +
  theme_minimal()

b5 <- ggplot(imputed_loan, aes(y = CLAGE)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Credit Line Age", y = "Credit Line Age (Months)") +
  theme_minimal()

b6 <- ggplot(imputed_loan, aes(y = YOJ)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Years on Job", y = "Years on Job") +
  theme_minimal()

b7 <- ggplot(imputed_loan, aes(y = DEROG)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Derogatory Reports", y = "DEROG") +
  theme_minimal()

b8 <- ggplot(imputed_loan, aes(y = DELINQ)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Delinquencies", y = "DELINQ") +
  theme_minimal()

b9 <- ggplot(imputed_loan, aes(y = NINQ)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Number of Inquiries", y = "NINQ") +
  theme_minimal()

b10 <- ggplot(imputed_loan, aes(y = CLNO)) +
  geom_boxplot(fill = "green", color = "black") +
  labs(title = "Number of Credit Lines", y = "CLNO") +
  theme_minimal()

# Arranging all boxplots in a grid 
grid.arrange(b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, ncol = 2)

#==========3.1.3 Summary Statistics of Derived Variables==============#

# Summary statistics for DERIVED VARIABLES
summary(imputed_loan[, c("EQUITY", "UNSECURED_LOAN", "DEFINITIVE_LOSS", "LTE_RATIO", "yoj_to_loan_ratio","LTV_RATIO", "OWNERSHIP", "CREDIT_SCORE","SCORE_CATEGORY")])

#---------------------------------------------------
#3.2 Visual Exploration of Relationships
#--------------------------------------------------

#========3.2.1 PERCENTAGES OF DEFAULTS VS NON-DEFAULTS =========#

# Total number of loans and number of defaults
total_loans <- nrow(imputed_loan)
num_defaults <- sum(imputed_loan$DEFAULT == 1)

# Percentage of defaults and non-defaults
percent_defaults <- (num_defaults / total_loans) * 100
percent_non_defaults <- 100 - percent_defaults

# Data frame for plotting
df <- data.frame(
  Status = c("Defaults", "Non-Defaults"),
  Percentage = c(percent_defaults, percent_non_defaults)
)

# Bar Plot for Percentage of Defaults vs Non-Defaults
ggplot(df, aes(x = Status, y = Percentage, fill = Status)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("red", "green")) +
  ggtitle("Percentage of Defaults vs Non-Defaults") +
  ylab("Percentage (%)") +
  xlab("") +
  geom_text(aes(label = paste0(round(Percentage, 2), "%")), vjust = -0.3) +
  theme_minimal()

#========3.2.2 TOTAL LOAN AMOUNT VS TOTAL DEFAULTED AMOUNT =========#

# Calculate total loan amount and total defaulted loan amount
total_loan_amount <- sum(imputed_loan$LOAN, na.rm = TRUE)
total_defaulted_loan_amount <- sum(imputed_loan$LOAN[imputed_loan$DEFAULT == 1], na.rm = TRUE)

# Data frame for plotting
loan_vs_default_data <- data.frame(
  Category = c("Total Loan Amount", "Total Defaulted Amount"),
  Amount = c(total_loan_amount, total_defaulted_loan_amount)
)

# Bar Plot for Total Loan Amount vs Total Defaulted Amount
loan_vs_default_plot <- ggplot(loan_vs_default_data, aes(x = Category, y = Amount, fill = Category)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Total Loan Amount" = "blue", "Total Defaulted Amount" = "red")) +
  ggtitle("Total Loan Amount vs Total Defaulted Amount") +
  ylab("Amount") +
  geom_text(aes(label = scales::comma(Amount)), vjust = -0.3) +
  theme_minimal()

# Display the plot
print(loan_vs_default_plot)

#========3.2.3 DEFAULTED LOAN AMOUNT: DEFINITIVE VS RECOVERABLE LOSSES  =========#

# Filter for defaulted loans
defaulted_loans <- imputed_loan[imputed_loan$DEFAULT == 1, ]

# Calculate definitive and recoverable losses
definitive_losses <- sum(defaulted_loans$DEFINITIVE_LOSS)
recoverable_losses <- sum(defaulted_loans$LOAN) - definitive_losses
total_defaulted_loan <- sum(defaulted_loans$LOAN)

# Data frame for plotting loss breakdown
loss_breakdown <- data.frame(
  Loss_Type = c("Definitive Losses", "Recoverable Losses"),
  Amount = c(definitive_losses, recoverable_losses),
  Percentage = c(definitive_losses / total_defaulted_loan * 100, recoverable_losses / total_defaulted_loan * 100)
)

ggplot(loss_breakdown, aes(x = Loss_Type, y = Amount, fill = Loss_Type)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Definitive Losses" = "red", "Recoverable Losses" = "green")) +
  ggtitle("Breakdown of Defaulted Loan Amount: Definitive vs Recoverable Losses") +
  ylab("Amount (in US Dollars)") +
  xlab("") + # Removes the x-axis label
  geom_text(aes(label = paste0(scales::comma(Amount), " (", round(Percentage, 2), "%)")), 
            vjust = -0.3, size = 4) +
  theme_minimal() +
  labs(fill = NULL) + 
  theme(legend.title = element_blank(), 
        axis.title.x = element_blank()) 

#========3.2.4 UNSECURED AND FULLY COVERED LOANS  =========#

# Filter for unsecured and fully covered loans
unsecured_loans <- imputed_loan[imputed_loan$EQUITY <= 0, ]
loans_fully_covered_by_collateral <- imputed_loan[imputed_loan$EQUITY >= imputed_loan$LOAN, ]

# Count the number of unsecured and fully covered loans
num_unsecured_loans <- nrow(unsecured_loans)
num_fully_covered_loans <- nrow(loans_fully_covered_by_collateral)

# Print results
cat("Number of unsecured loans: ", num_unsecured_loans, "\n")
cat("Number of fully covered loans: ", num_fully_covered_loans, "\n")

# Create the Security_Status column
imputed_loan <- imputed_loan %>%
  mutate(Security_Status = case_when(
    EQUITY >= LOAN ~ "Fully Secured",
    EQUITY > 0 & EQUITY < LOAN ~ "Partially Secured",
    EQUITY <= 0 ~ "Non-Secure"
  ))

# Summary table for loan security counts
loan_security_counts <- imputed_loan %>%
  group_by(Security_Status) %>%
  summarise(Count = n())

# Bar Plot for Count of Fully Secured, Partially Secured, and Non-Secure Loans
a1 <- ggplot(loan_security_counts, aes(x = Security_Status, y = Count, fill = Security_Status)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Fully Secured" = "blue", "Partially Secured" = "yellow", "Non-Secure" = "red")) +
  ggtitle("Count of Fully Secured, Partially Secured, and Non-Secure Loans") +
  ylab("Number of Loans") +
  geom_text(aes(label = Count), vjust = -0.3) +
  theme_minimal()

# Calculate default rates for each Security_Status category
default_rate_counts <- imputed_loan %>%
  group_by(Security_Status) %>%
  summarise(
    Total = n(),
    Defaults = sum(DEFAULT == 1),
    Default_Rate = Defaults / Total * 100
  )

# Bar Plot for Default Rates by Loan Security Status
a2 <- ggplot(default_rate_counts, aes(x = Security_Status, y = Default_Rate, fill = Security_Status)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Fully Secured" = "blue", "Partially Secured" = "yellow", "Non-Secure" = "red")) +
  ggtitle("Default Rates by Loan Security Status") +
  ylab("Default Rate (%)") +
  geom_text(aes(label = sprintf("%.2f%%", Default_Rate)), vjust = -0.3) +
  theme_minimal()

# Arrange the plots side by side
grid.arrange(a1, a2, ncol = 2)

#=========== 3.2.5 DEFINITIVE LOSS AND UNSECURED LOANS ============#

# Calculations for total actual loss and uncovered loan amounts
total_actual_loss <- sum(imputed_loan$DEFINITIVE_LOSS)
total_uncovered_amount <- sum(imputed_loan$UNSECURED_LOAN)
actual_loss_count <- sum(imputed_loan$DEFINITIVE_LOSS > 0)
uncovered_loan_count <- sum(imputed_loan$UNSECURED_LOAN > 0)

# Data frame for plotting (including total amounts and counts)
loss_data <- data.frame(
  Loss_Type = c("Definitive Loss", "Unsecured Loan Amount"),
  Amount = c(total_actual_loss, total_uncovered_amount),
  Count = c(actual_loss_count, uncovered_loan_count)
)

# Bar Plot for Total Definitive Loss and Uncovered Loan Amounts
bar_plot <- ggplot(loss_data, aes(x = Loss_Type, y = Amount, fill = Loss_Type)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Definitive Loss" = "red", "Unsecured Loan Amount" = "orange")) +
  ggtitle("Bank's Total Definitive Loss and Unsecured Loan Amounts") +
  ylab("Amount (in US Dollars)") +
  geom_text(aes(label = paste0(scales::comma(Amount), " (", Count, " records)")), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  labs(fill = NULL, x = NULL) + 
  theme(legend.title = element_blank())

# Calculate percentage based on record counts
total_records <- nrow(imputed_loan)
loss_data$Percentage <- (loss_data$Count / total_records) * 100

# Percentage Plot for Total Record Counts
percentage_plot <- ggplot(loss_data, aes(x = Loss_Type, y = Percentage, fill = Loss_Type)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Definitive Loss" = "red", "Unsecured Loan Amount" = "orange")) +
  ggtitle("Percentages Relative to Total Number of Records") +
  ylab("Percentage") +
  geom_text(aes(label = paste0(round(Percentage, 1), "%")), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  labs(fill = NULL, x = NULL) + 
  theme(legend.title = element_blank())

# Combine the two plots
grid.arrange(bar_plot, percentage_plot, ncol = 2)

#=========== 3.2.6 CLNO VS CLAGE ============#

# Scatter plot of CLNO vs CLAGE with custom highlighting
ggplot(imputed_loan, aes(x = CLAGE, y = CLNO)) +
  geom_point(alpha = 0.5, color = "blue") +  # Base scatter plot for all points
  
  # Highlight points where CLAGE = 0 and CLNO > 0 (in red, with a legend)
  geom_point(data = imputed_loan[imputed_loan$CLAGE == 0 & imputed_loan$CLNO > 0, ], 
             aes(x = CLAGE, y = CLNO, color = "CLAGE = 0, CLNO > 0"), size = 2) +
  
  # Highlight points where CLNO = 0 and CLAGE > 0 (in green, with a legend)
  geom_point(data = imputed_loan[imputed_loan$CLNO == 0 & imputed_loan$CLAGE > 0, ], 
             aes(x = CLAGE, y = CLNO, color = "CLNO = 0, CLAGE > 0"), size = 2) +
  
  # Scale color manually to assign specific colors
  scale_color_manual(values = c("CLAGE = 0, CLNO > 0" = "red", "CLNO = 0, CLAGE > 0" = "orange")) +
  
  # Add labels and title
  labs(title = "Scatter Plot of CLNO vs CLAGE", 
       x = "Credit Line Age (CLAGE in months)", 
       y = "Number of Credit Lines (CLNO)",
       color = "Highlighted Points") +  # Legend title
  
  theme_minimal()

# Subset rows where CLAGE > 0 and CLNO == 0
inconsistent_rows <- imputed_loan[imputed_loan$CLAGE > 0 & imputed_loan$CLNO == 0, ]
nrow(inconsistent_rows )

#Obtaining CLAGE range for inconsistent rows 
min_clage <- min(inconsistent_rows$CLAGE)/12
max_clage <- max(inconsistent_rows$CLAGE)/12
min_clage
max_clage

#Obtaining number of defaults
inconsistent_rows_defaults<-nrow(subset(inconsistent_rows,DEFAULT=="1"))
inconsistent_rows_defaults

#Obtaining number of defaults in percentage
inconsistent_rows_defaults_percentage<-inconsistent_rows_defaults/nrow(inconsistent_rows )*100
inconsistent_rows_defaults_percentage

# Delete rows where CLAGE > 0 and CLNO == 0
imputed_loan <- imputed_loan[!(imputed_loan$CLAGE > 0 & imputed_loan$CLNO == 0), ]

#=========== 3.2.7 LOAN AMOUNT VS EQUITY ============#

# First plot: Differentiating defaulters and non-defaulters by color
plot1 <- ggplot(imputed_loan, aes(x = LOAN, y = EQUITY, color = as.factor(DEFAULT))) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Loan Amount vs. Equity by Default Status", x = "Loan Amount", y = "Equity", color = "Default Status") +
  scale_color_manual(values = c("green", "red"), labels = c("Non-Defaulter", "Defaulter"))

# Second plot: Scatter plot with fitted linear regression line
plot2 <- ggplot(imputed_loan, aes(x = LOAN, y = EQUITY)) +
  geom_point(alpha = 0.6, color = "darkblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  theme_minimal() +
  labs(title = "Loan Amount vs. Equity with Fitted Line", x = "Loan Amount", y = "Equity")

# Combining the two plots 
grid.arrange(plot1, plot2, ncol = 1)

# Subset the data for negative equity
negative_equity_data <- imputed_loan[imputed_loan$EQUITY < 0, ]

# Calculate the number of defaulters and non-defaulters in the negative equity group
negative_equity_counts <- table(negative_equity_data$DEFAULT)

# Calculate total number of records in negative equity group
total_negative_equity <- sum(negative_equity_counts)

# Calculate the percentage of defaulters and non-defaulters
percent_defaulters <- (negative_equity_counts["1"] / total_negative_equity) * 100
percent_defaulters

percent_non_defaulters <- (negative_equity_counts["0"] / total_negative_equity) * 100
percent_non_defaulters

#=========== 3.2.8 DEFAULT vs LOAN REASON ============#

# Default rate by loan reason
ggplot(imputed_loan, aes(x = REASON, fill = as.factor(DEFAULT))) +
  geom_bar(position = "fill") +
  theme_minimal() +
  scale_fill_manual(values = c("green", "red"), name = "Default Status", labels = c("No Default", "Default")) +
  labs(title = "Default Rate by Loan Reason", x = "Loan Reason", y = "Proportion")

#=========== 3.2.9 CLAGE VS DEFAULT ============#

ggplot(imputed_loan, aes(x = CLAGE, fill = as.factor(DEFAULT))) +
  geom_density(alpha = 0.6) +
  theme_minimal() +
  scale_fill_manual(values = c("green", "red"), name = "Default Status", labels = c("No Default", "Default")) +
  labs(title = "Distribution of CLAGE by Default Status", x = "CLAGE", y = "Density")

#=========== 3.2.10 DEBTINC vs LOAN AMOUNT ============#

ggplot(imputed_loan, aes(x = DEBTINC, y = LOAN, size = NINQ, color = as.factor(DEFAULT))) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  scale_color_manual(values = c("green", "red"), name = "Default Status", labels = c("No Default", "Default")) +
  labs(title = "Debt-to-Income Ratio vs. Loan Amount", x = "Debt-to-Income Ratio", y = "Loan Amount")

#=========== 3.2.11 DEBTINC vs DEFAULT ============#
imputed_loan <- imputed_loan %>%
  mutate(DEBTINC_bin = cut(DEBTINC, 
                           breaks = seq(0, max(DEBTINC, na.rm = TRUE) + 10, by = 10), 
                           include.lowest = TRUE))

# Create a proportional bar plot for DEBTINC vs DEFAULT
ggplot(imputed_loan, aes(x = DEBTINC_bin, fill = as.factor(DEFAULT))) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("green", "red"), labels = c("No Default", "Default")) +
  labs(title = "Proportion of Defaults by Debt-to-Income (DEBTINC) Categories",
       x = "Debt-to-Income Ratio (%)",
       y = "Proportion",
       fill = "Default Status") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


imputed_loan$DEBTINC_bin<-NULL
#=========== 3.2.12 DEFAULT VS NINQ ============#

# Create a dataset with proportions of defaults and non-defaults for each NINQ
default_ninq_proportions <- imputed_loan %>%
  group_by(NINQ, DEFAULT) %>%
  summarise(Count = n()) %>%
  mutate(Proportion = Count / sum(Count))  # Calculate the proportion for each group

# Plot default and non-default percentages across NINQ
ggplot(default_ninq_proportions, aes(x = as.factor(NINQ), y = Proportion, fill = as.factor(DEFAULT))) +
  geom_bar(stat = "identity", position = "fill") +
  theme_minimal() +
  scale_fill_manual(values = c("green", "red"), name = "Default Status", labels = c("Non-default", "Default")) +
  labs(title = "Default and Non-default Proportions Across NINQ", x = "Number of Inquiries (NINQ)", y = "Proportion") +
  scale_y_continuous(labels = scales::percent)

#=========== 3.2.13 DEROG VS DEFAULT ============#

# Create a bar plot for DEROG and DEFAULT
ggplot(imputed_loan, aes(x = as.factor(DEROG), fill = as.factor(DEFAULT))) +
  geom_bar(position = "dodge") +
  labs(title = "Relationship Between Derogatory Marks and Default Status",
       x = "Number of Derogatory Marks",
       y = "Count",
       fill = "Default Status") +
  scale_fill_manual(values = c("0" = "green", "1" = "red"), labels = c("Non-default", "Default")) +
  theme_minimal()

#=========== 3.2.14 YOJ VS DEFAULT ============#

# Create a new column for YOJ ranges in intervals of 5 years
imputed_loan <- imputed_loan %>%
  mutate(YOJ_range = cut(YOJ, breaks = seq(0, 45, by = 5), right = FALSE, include.lowest = TRUE))

# Calculate the default rate across YOJ ranges
default_rate_by_yoj_range <- imputed_loan %>%
  group_by(YOJ_range) %>%
  summarise(
    total_loans = n(),
    default_count = sum(DEFAULT == 1),
    default_rate = (default_count / total_loans) * 100  # Convert to percentage
  )

# Simple bar plot for default rate across YOJ ranges, with loan counts labeled
ggplot(default_rate_by_yoj_range, aes(x = YOJ_range, y = default_rate)) +
  geom_bar(stat = "identity", fill = "red") +  # Solid red bars
  geom_text(aes(label = total_loans), vjust = -0.5, size = 3.5) +  # Adding loan count labels
  labs(title = "Default Rate Across Years of Job (YOJ) Ranges with Loan Counts",
       x = "Years of Job (YOJ) Ranges",
       y = "Default Rate (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

imputed_loan$YOJ_range<-NULL

#=========== 3.2.15 DEFAULT VS JOB CATEGORIES ============#

# Calculate the default rate across job categories
default_rate_by_job <- imputed_loan %>%
  group_by(JOB) %>%
  summarise(
    total_loans = n(),
    default_count = sum(DEFAULT == 1),
    default_rate = (default_count / total_loans) * 100  # Convert to percentage
  )

# Plot a bar plot for the default rate across job categories
ggplot(default_rate_by_job, aes(x = reorder(JOB, -default_rate), y = default_rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +  # Removed alpha for full opacity
  geom_text(aes(label = total_loans), vjust = -0.5, size = 3.5) +  # Adding loan count labels
  labs(title = "Default Rate Across Job Categories",
       x = "Job Category",
       y = "Default Rate (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for readability

#=========== 3.2.16 LOAN AMOUNTS VS DEFAULT RATE ============#

# Ensure LOAN is numeric and remove any NA values
imputed_loan <- imputed_loan %>% 
  filter(!is.na(LOAN))  # Remove any rows where LOAN is NA

# Bin loan amounts into ranges of 2500
loan_bins <- cut(imputed_loan$LOAN, breaks = seq(0, max(imputed_loan$LOAN, na.rm = TRUE), by = 2500), include.lowest = TRUE)

# Calculate default rate by loan amount range
default_rates_by_loan <- imputed_loan %>%
  mutate(loan_bins = loan_bins) %>%
  group_by(loan_bins) %>%
  summarise(Default_Rate = sum(DEFAULT == 1) / n() * 100,
            Count = n()) %>%
  filter(!is.na(loan_bins))  # Exclude NA bins

# Plot default rate by loan amount range, with a trend line
ggplot(default_rates_by_loan, aes(x = loan_bins, y = Default_Rate, group = 1)) +
  geom_col(fill = "red") +  # Bar plot for default rate by loan amount range
  geom_smooth(method = "loess", se = FALSE, color = "blue") +  # Trend line using LOESS smoothing
  labs(title = "Default Rate by Loan Amount Range", x = "Loan Amount Range", y = "Default Rate (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels

#=========== 3.2.17 SCORE CATEGORY VS DEFAULT ============#

# Calculate the percentage of defaults for each credit score category
score_default_summary <- imputed_loan %>%
  group_by(SCORE_CATEGORY) %>%
  summarise(default_rate = mean(DEFAULT == 1) * 100)

# Plot the percentage of defaults by credit score category
plot1 <- ggplot(score_default_summary, aes(x = SCORE_CATEGORY, y = default_rate)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(round(default_rate, 1), "%")), 
            vjust = -0.5, size = 4) +  # Add labels above bars
  labs(title = "Percentage of Defaults by Credit Score Category",
       x = "Credit Score Category",
       y = "Default Rate (%)") +
  theme_minimal()

# Calculate the percentage of applicants in each Credit Score Category
score_category_percentage <- imputed_loan %>%
  group_by(SCORE_CATEGORY) %>%
  summarise(count = n()) %>%
  mutate(percentage = (count / sum(count)) * 100)

# Create a bar plot showing percentage distribution of Credit Score Categories
plot2 <- ggplot(score_category_percentage, aes(x = SCORE_CATEGORY, y = percentage)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            vjust = -0.5, size = 4) +  # Add labels on top of bars
  labs(title = "Percentage Distribution of Credit Score Categories", 
       x = "Credit Score Category", 
       y = "Percentage of Applicants (%)") +
  theme_minimal()

# Combine the two plots side by side
grid.arrange(plot1, plot2, ncol = 2)

#=========== 3.2.18 Key Credit Metrics by Credit Score Category ============#

category_summary <- imputed_loan %>%
  group_by(SCORE_CATEGORY) %>%
  summarize(
    Avg_Delinquencies = mean(DELINQ, na.rm = TRUE),
    Avg_Debt_Income = mean(DEBTINC, na.rm = TRUE),
    Avg_Credit_Age = mean(CLAGE, na.rm = TRUE),
    Avg_Ownership_Ratio = mean(OWNERSHIP, na.rm = TRUE),
    Avg_Inquiries = mean(NINQ, na.rm = TRUE),
    Count = n()
  )

# View the summary
print(category_summary)

# Create individual plots using the same color (steel blue)
plot_score1 <- ggplot(category_summary, aes(x = SCORE_CATEGORY, y = Avg_Delinquencies)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average Delinquencies by Credit Score Category",
       x = "Credit Score Category",
       y = "Average Delinquencies") +
  theme_minimal()

plot_score2 <- ggplot(category_summary, aes(x = SCORE_CATEGORY, y = Avg_Debt_Income)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average Debt-to-Income Ratio by Credit Score Category",
       x = "Credit Score Category",
       y = "Average Debt-to-Income Ratio") +
  theme_minimal()

plot_score3 <- ggplot(category_summary, aes(x = SCORE_CATEGORY, y = Avg_Credit_Age)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average Credit Age by Credit Score Category",
       x = "Credit Score Category",
       y = "Average Credit Age (Months)") +
  theme_minimal()

plot_score4 <- ggplot(category_summary, aes(x = SCORE_CATEGORY, y = Avg_Ownership_Ratio)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average Ownership Ratio by Credit Score Category",
       x = "Credit Score Category",
       y = "Average Ownership Ratio") +
  theme_minimal()

plot_score5 <- ggplot(category_summary, aes(x = SCORE_CATEGORY, y = Avg_Inquiries)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Average Number of Inquiries by Credit Score Category",
       x = "Credit Score Category",
       y = "Average Number of Inquiries") +
  theme_minimal()

# Arrange all plots together
grid.arrange(plot_score1, plot_score2, plot_score3, plot_score4, plot_score5, ncol = 2)

#=========== 3.2.19 DEFAULT RATE AND AVERAGE LOAN AMOUNT VS CREDIT SCORE ============#

# Define the credit score thresholds 
credit_score_thresholds <- seq(300, 850, by = 10) 
# Initialize an empty data frame to store results
credit_score_results <- data.frame(CREDIT_SCORE = numeric(), Default_Rate = numeric(), Avg_Loan_Amount = numeric())

# Loop through each CREDIT_SCORE threshold and calculate the default rate and average loan amount
for (cs in credit_score_thresholds) {
  filtered_data <- imputed_loan[imputed_loan$CREDIT_SCORE >= cs, ]
  default_rate <- sum(filtered_data$DEFAULT == 1) / nrow(filtered_data) * 100
  avg_loan_amount <- mean(filtered_data$LOAN, na.rm = TRUE)
  credit_score_results <- rbind(credit_score_results, data.frame(CREDIT_SCORE = cs, Default_Rate = default_rate, Avg_Loan_Amount = avg_loan_amount))
}

# Find the CREDIT_SCORE threshold closest to a 5% default rate
credit_score_sweet_spot <- credit_score_results[which.min(abs(credit_score_results$Default_Rate - 5)), ]

# Print the result
print(paste("The sweet spot CREDIT_SCORE for approximately 5% default rate is:", round(credit_score_sweet_spot$CREDIT_SCORE, 2)))
print(paste("The corresponding default rate is:", round(credit_score_sweet_spot$Default_Rate, 2), "%"))

# Plot Default Rate and Average Loan Amount vs. Credit Score
ggplot(credit_score_results, aes(x = CREDIT_SCORE)) +
  geom_line(aes(y = Default_Rate, color = "Default Rate")) +
  geom_line(aes(y = Avg_Loan_Amount / 1000, color = "Avg Loan Amount (in thousands)")) +  # Dividing loan amount for better scale comparison
  geom_vline(xintercept = credit_score_sweet_spot$CREDIT_SCORE, linetype = "dashed", color = "red") +
  labs(title = "Default Rate and Average Loan Amount vs. Credit Score",
       x = "Credit Score",
       y = "Default Rate (%) and Average Loan Amount (in thousands)") +
  scale_y_continuous(sec.axis = sec_axis(~.*1000, name = "Average Loan Amount")) +  # Secondary y-axis for average loan amount
  scale_color_manual(values = c("blue", "green")) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(color = "Metric")

#=========== 3.2.20 RISK VS LOAN AMOUNT FOR EACH SCORE CATEGORY ============#

# Summarize data by score category
score_summary <- imputed_loan %>%
  group_by(SCORE_CATEGORY) %>%
  summarise(Avg_Loan_Amount = mean(LOAN, na.rm = TRUE),
            Default_Rate = sum(DEFAULT == 1) / n() * 100)

# View the summary
score_summary

# Plot Risk (Default Rate) vs Loan Amount for each credit score category
ggplot(score_summary, aes(x = Avg_Loan_Amount, y = Default_Rate, label = SCORE_CATEGORY)) +
  geom_point(size = 4, color = "blue") +
  geom_text(vjust = -0.5) +
  labs(title = "Default Risk vs Loan Amount for Each Credit Score Category",
       x = "Average Loan Amount",
       y = "Default Rate (%)") +
  theme_minimal()

#=========== 3.2.21 CUMULATIVE CHARTS: SCORE VS LOAN VS DEFAULT  ============#

# Sort the data by loan amount within each credit score category
imputed_loan <- imputed_loan %>%
  arrange(SCORE_CATEGORY, LOAN)

# Calculate the cumulative average loan amount and cumulative default rate
cumulative_gain_data <- imputed_loan %>%
  group_by(SCORE_CATEGORY) %>%
  mutate(Cumulative_Defaults = cumsum(DEFAULT == 1),  # Cumulative count of defaults
         Cumulative_Loans = row_number(),  # Count of loans processed at each step
         Cumulative_Avg_Loan = cumsum(LOAN) / Cumulative_Loans,  # Cumulative average loan amount
         Cumulative_Default_Rate = Cumulative_Defaults / Cumulative_Loans * 100)  # Cumulative default rate

# Plot cumulative default rate vs cumulative average loan amount for each score category
ggplot(cumulative_gain_data, aes(x = Cumulative_Avg_Loan, y = Cumulative_Default_Rate, color = SCORE_CATEGORY)) +
  geom_line() +
  facet_wrap(~ SCORE_CATEGORY, scales = "free") +  # Separate plots for each category
  labs(title = "Cumulative Default Rate vs. Cumulative Average Loan Amount by Credit Score Category",
       x = "Cumulative Average Loan Amount",
       y = "Cumulative Default Rate (%)") +
  theme_minimal() +
  theme(legend.position = "none")  

#=========== 3.2.22 CREDIT SCORE VS DEFAULT RATE, LOAN AMOUNT AND PROFIT ============#

# Get the range of credit scores from the dataset
min_credit_score <- min(imputed_loan$CREDIT_SCORE, na.rm = TRUE)
max_credit_score <- max(imputed_loan$CREDIT_SCORE, na.rm = TRUE)

# Define credit score thresholds from the minimum to the maximum credit score in the dataset
credit_score_thresholds <- seq(min_credit_score, max_credit_score, by = 10)

# Create a dataframe to store the results
threshold_results <- data.frame(
  Threshold = numeric(),
  Avg_Loan_Amount = numeric(),
  Default_Rate = numeric(),
  Total_Loan_Amount = numeric(),
  Profit = numeric()  # Adding Profit column
)

# Loop over thresholds and calculate metrics
for (threshold in credit_score_thresholds) {
  filtered_data <- imputed_loan %>% filter(CREDIT_SCORE >= threshold)
  
  if (nrow(filtered_data) > 0) {  # Ensure we have valid data for each threshold
    avg_loan <- mean(filtered_data$LOAN, na.rm = TRUE)
    default_rate <- sum(filtered_data$DEFAULT == 1) / nrow(filtered_data) * 100
    total_loan_amount <- sum(filtered_data$LOAN, na.rm = TRUE)
    
    # Calculate Profit
    losses <- total_loan_amount * (default_rate / 100)
    profit <- total_loan_amount - losses
    
    # Append the results to the dataframe
    threshold_results <- rbind(threshold_results, 
                               data.frame(Threshold = threshold, 
                                          Avg_Loan_Amount = avg_loan, 
                                          Default_Rate = default_rate,
                                          Total_Loan_Amount = total_loan_amount,
                                          Profit = profit))  # Include Profit in the dataframe
  }
}

# Print the results
print(threshold_results)

# Plot 1: Credit Score vs Default Rate
plot1 <- ggplot(threshold_results, aes(x = Threshold, y = Default_Rate)) +
  geom_line(color = "blue", linewidth = 1) +
  labs(title = "Credit Score vs Default Rate",
       x = "Credit Score Threshold",
       y = "Default Rate (%)") +
  theme_minimal()

# Plot 2: Credit Score vs Total Loan Amount
plot2 <- ggplot(threshold_results, aes(x = Threshold, y = Total_Loan_Amount)) +
  geom_line(color = "green", linewidth = 1) +
  labs(title = "Credit Score vs Total Loan Amount",
       x = "Credit Score Threshold",
       y = "Total Loan Amount") +
  theme_minimal()

# Plot 3: Credit Score vs Profit
plot3 <- ggplot(threshold_results, aes(x = Threshold, y = Profit)) +
  geom_line(color = "purple", linewidth = 1) +
  labs(title = "Credit Score vs Profit",
       x = "Credit Score Threshold",
       y = "Profit") +
  theme_minimal()

# Print the plots
print(plot1)
print(plot2)
print(plot3)

# Plot all three together in a 1x3 grid layout
grid.arrange(plot1, plot2, plot3, ncol = 3)

#==========================================
#Phase 4 : Predictor Analysis and Relevancy
#==========================================

# Convert DEFAULT to numeric (0 and 1) for correlation purposes
imputed_loan$DEFAULT <- as.numeric(as.character(imputed_loan$DEFAULT))

# Select numeric and integer columns, including DEFAULT
numeric_data <- imputed_loan %>%
  select(where(is.numeric))

# Compute the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Create a correlation heatmap using ggcorrplot
ggcorrplot(correlation_matrix, 
           hc.order = TRUE,  # Hierarchical clustering order for better visualization
           type = "lower",  # Show only the lower part of the matrix
           lab = TRUE,  # Add correlation coefficients as labels
           title = "Correlation Heatmap of Numeric Variables including DEFAULT",
           outline.col = "white",  # Optional: Add white outline around each cell
           colors = c("blue", "white", "red"))  # Color gradient (low to high correlation)

imputed_loan$DEFAULT<-as.factor(imputed_loan$DEFAULT)
str(imputed_loan)

head(imputed_loan)

#==========================================
#Phase 5: Data Partitioning
#==========================================

# Data Partitioning
set.seed(123)  # Setting seed for reproducibility

# Calculate the number of rows for each subset (70%, 20%, 10%)
total_rows <- nrow(imputed_loan)
train_rows <- round(0.7 * total_rows)
validation_rows <- round(0.2 * total_rows)

# Generate indices for partitioning
train_indices <- sample(seq_len(total_rows), size = train_rows)
remaining_indices <- setdiff(seq_len(total_rows), train_indices)
validation_indices <- sample(remaining_indices, size = validation_rows)
test_indices <- setdiff(remaining_indices, validation_indices)

# Split the dataset into training, validation, and testing sets
train_set <- imputed_loan[train_indices, ]
validation_set <- imputed_loan[validation_indices, ]
test_set <- imputed_loan[test_indices, ]

# Print the number of rows in each subset for verification
cat("Training Set Rows:", nrow(train_set), "\n")
cat("Validation Set Rows:", nrow(validation_set), "\n")
cat("Testing Set Rows:", nrow(test_set), "\n")

#==========================================
# Phase 6: Feature Selection
#==========================================

#------------------------------------------------------------------
# 6.1 Exclusion of Outcome Dependent Variable and Redundant Variables 
#------------------------------------------------------------------

# Exclude variables that are not relevant for modeling
# Variables removed: UNSECURED_LOAN, DEFINITIVE_LOSS, CREDIT_SCORE, SCORE_CATEGORY, MORTDUE, VALUE, Security_Status
train_set <- subset(train_set, select = -c(DEFINITIVE_LOSS, CREDIT_SCORE, SCORE_CATEGORY, MORTDUE, VALUE, Security_Status))
validation_set <- subset(validation_set, select = -c(DEFINITIVE_LOSS, CREDIT_SCORE, SCORE_CATEGORY, MORTDUE, VALUE, Security_Status))
test_set <- subset(test_set, select = -c(DEFINITIVE_LOSS, CREDIT_SCORE, SCORE_CATEGORY, MORTDUE, VALUE, Security_Status))

# Preview the modified training set
head(train_set)

#------------------------------------------
# 6.2 VIF
#------------------------------------------

# Step 10.2: Variance Inflation Factor (VIF) Analysis
# Fit a logistic regression model on the training set
logistic_model <- glm(DEFAULT ~ ., data = train_set, family = "binomial")

# Calculate and display VIF values
vif_values <- vif(logistic_model)
cat("Variance Inflation Factor (VIF) Values:\n")
print(vif_values)

#------------------------------------------
# 6.3 BORUTA FEATURE SELECTION
#------------------------------------------

#Boruta Feature Selection
set.seed(123)  # Setting seed for reproducibility

# Perform Boruta feature selection
boruta_result <- Boruta(DEFAULT ~ ., data = train_set, doTrace = 2)

# Apply TentativeRoughFix to finalize the selection
final_result <- TentativeRoughFix(boruta_result)

# Display the final decision on feature importance
cat("Boruta Final Decision:\n")
print(final_result)

# Extract important attributes from Boruta results
important_attributes <- attStats(final_result)
cat("Important Attributes:\n")
print(important_attributes)

# Visualizing Boruta Feature Importance
# Adjusting plot margins for better readability
par(mar = c(12, 5, 5, 2))

# Boruta feature importance plot
plot(boruta_result, main = "Boruta Feature Importance", las = 2)  

# Displaying confirmed important variables in a clear format
important_vars <- important_attributes %>%
  filter(decision == "Confirmed")

# Bar Plot of Important Features
ggplot(important_vars, aes(x = reorder(row.names(important_vars), meanImp), y = meanImp)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Boruta Feature Importance",
       x = "Features",
       y = "Mean Importance")

#==========================================
# Phase 7: Model Fitting
#==========================================

#------------------------------------------
# 7.1 Stepwise Logistic Regression
#------------------------------------------
head(train_set)
set.seed(123)
# Fit a full model including all predictors
full_model <- glm(DEFAULT ~ ., data = train_set, family = binomial)

# Perform stepwise selection
stepwise_model <- step(full_model, direction = "both")

# Display the summary of the selected model
summary(stepwise_model)

#------------------------------------------
# 7.2 Decision Tree
#------------------------------------------
set.seed(123)
# Train the decision tree model
decision_tree_model <- rpart(DEFAULT ~ ., data = train_set, method = "class")

# Plot the decision tree for transparency
rpart.plot(decision_tree_model, split.font = 1, shadow.col = "gray", varlen = -18, cex = 0.7)

# Extract the frame (node details) from the decision tree
tree_frame <- decision_tree_model$frame

# Identify leaf nodes (where "var" == "<leaf>")
leaf_nodes <- tree_frame[tree_frame$var == "<leaf>", ]

# Display the number of records in each leaf node
leaf_records <- leaf_nodes$n
leaf_records

#Checking default parameters
decision_tree_model$control

#------------------------------------------
# 7.3 KNN
#------------------------------------------

#========== 7.3.1 STANDARDIZING PARTITION SETS FOR KNN ===========#

# Create copies of datasets to hold the standardized values
train.norm.df <- train_set
valid.norm.df <- validation_set
test.norm.df <- test_set
str(train_set)

# Converting REASON to numeric
train.norm.df$REASON <- ifelse(train_set$REASON == "HomeImp", 1, 0)
valid.norm.df$REASON <- ifelse(validation_set$REASON == "HomeImp", 1, 0)
test.norm.df$REASON <- ifelse(test_set$REASON == "HomeImp", 1, 0)
str(train.norm.df)

# One-Hot Encoding for multi-level categorical variable
train.norm.df <- dummy_cols(train.norm.df, select_columns = "JOB", remove_first_dummy = FALSE)
valid.norm.df <- dummy_cols(valid.norm.df, select_columns = "JOB", remove_first_dummy = FALSE)
test.norm.df <- dummy_cols(test.norm.df, select_columns = "JOB", remove_first_dummy = FALSE)

# Remove original categorical column after encoding 
train.norm.df <- subset(train.norm.df, select = -c(JOB))
valid.norm.df <- subset(valid.norm.df, select = -c(JOB))  
test.norm.df <- subset(test.norm.df, select = -c(JOB))   

# Select columns for standardization
numeric_columns <- setdiff(names(train.norm.df), c("DEFAULT"))

# Apply Z-score standardization using preProcess() 
norm.values <- preProcess(train.norm.df[, numeric_columns], method = c("center", "scale"))

# Apply the standardization to the train, validation, and test sets
train.norm.df[, numeric_columns] <- predict(norm.values, train.norm.df[, numeric_columns])
valid.norm.df[, numeric_columns] <- predict(norm.values, valid.norm.df[, numeric_columns])
test.norm.df[, numeric_columns] <- predict(norm.values, test.norm.df[, numeric_columns])

# Check the summary of the standardized data
summary(train.norm.df)
summary(valid.norm.df)
summary(test.norm.df)

# Check standard deviations for the numeric columns in the training set
apply(train.norm.df[, numeric_columns], 2, sd)
# Check standard deviations for the numeric columns in the validation set
apply(valid.norm.df[, numeric_columns], 2, sd)
# Check standard deviations for the numeric columns in the test set
apply(test.norm.df[, numeric_columns], 2, sd)

#========== 7.3.2 KNN MODEL ===========#

# Define the target and feature sets
train_features <- train.norm.df[, setdiff(names(train.norm.df), "DEFAULT")]
train_target <- train.norm.df$DEFAULT

valid_features <- valid.norm.df[, setdiff(names(valid.norm.df), "DEFAULT")]
valid_target <- valid.norm.df$DEFAULT

test_features <- test.norm.df[, setdiff(names(test.norm.df), "DEFAULT")]
test_target <- test.norm.df$DEFAULT

#Finding Best K
trainControl <- trainControl(method = "cv", number = 10)
tune_knn <- train(DEFAULT ~ ., 
                  data = train.norm.df, 
                  method = "knn", 
                  trControl = trainControl, 
                  tuneGrid = expand.grid(k = 1:20))
print(tune_knn)
plot(tune_knn)

# Apply KNN on the validation set
knn_pred_valid <- knn(train = train_features, 
                      test = valid_features, 
                      cl = train_target, 
                      k = 5,
                      prob=TRUE)

# Extract probabilities for the positive class (e.g., "1")
positive_probs <- ifelse(knn_pred_valid == "1", 
                         attr(knn_pred_valid, "prob"), 
                         1 - attr(knn_pred_valid, "prob"))

#------------------------------------------
# 7.4 RANDOM FOREST
#------------------------------------------
set.seed(123)

# Training the Random Forest Model with class weights
rf_model <- randomForest(DEFAULT ~.-EQUITY, 
                         data = train_set, 
                         ntree = 500,
                         importance = TRUE,
                         classwt = c('0' = 0.3, '1' = 0.7))  # Setting class weights
rf_model

# Get variable importance
importance_values <- importance(rf_model, type = 1)  # type = 1 gives mean decrease in accuracy
importance_values 

# Plot variable importance
varImpPlot(rf_model, type = 1, main = "Variable Importance (Mean Decrease in Accuracy)")

# Print the model summary
print(rf_model)

#==========================================
# Phase 8: PERFORMANCE EVALUATION
#==========================================

#--------------------------------------------------
# 8.1 STEPWISE LOGISTIC REGRESSION CONFUSION MATRIX 
#--------------------------------------------------

#Predicted probabilities
glm_predicted_probs <- predict(stepwise_model, newdata = validation_set, type = "response")

# Setting threshold of 0.2 to obtain class labels 
predicted_classes <- ifelse(glm_predicted_probs > 0.2, 1, 0)

# Make sure actual and predicted classes are factors for confusionMatrix
validation_set$DEFAULT <- as.factor(validation_set$DEFAULT)
predicted_classes<- as.factor(predicted_classes)

# Create confusion matrix
cmatrix_stepwise <- confusionMatrix(predicted_classes, validation_set$DEFAULT, positive = "1")
cmatrix_stepwise 

#------------------------------------------------
# 8.2 DECISION TREE PERFORMANCE CONFUSION MATRIX
#------------------------------------------------

# Decision tree model predictions
tree_predicted_probs <- predict(decision_tree_model, newdata = validation_set, type = "prob")

#probabilities of default
tree_predicted_probs_1<-tree_predicted_probs[,2]

# Converting probabilities to class labels ( 0.5 threshold)
predicted_classes <- ifelse(tree_predicted_probs[, 2] > 0.2, 1, 0)

# Confusion matrix 
cmatrix_tree <- confusionMatrix(as.factor(predicted_classes), 
                                as.factor(validation_set$DEFAULT), 
                                positive = "1")
cmatrix_tree
str(train_set)

#------------------------------------------------
# 8.3 KNN PERFORMANCE CONFUSION MATRIX
#------------------------------------------------

#converting to factors
knn_pred_valid <- factor(knn_pred_valid, levels = c(0, 1))
valid_target <- factor(valid_target, levels = c(0, 1))

# Confusion matrix
cmatrix_knn<- confusionMatrix(knn_pred_valid, valid_target,
                              positive = "1")
cmatrix_knn

#------------------------------------------------
# 8.4 RF CONFUSION MATRIX
#------------------------------------------------

# Predictions and probabilities for validation set
rf_probabilities <- predict(rf_model, newdata = validation_set, type = "prob")[,2]  # Probabilities of default (class 1)

# Class predictions based on a threshold 0.2
rf_predictions <- ifelse(rf_probabilities >= 0.2, 1, 0)


# Confusion matrix 
cmatrix_rf <- confusionMatrix(factor(rf_predictions), factor(validation_set$DEFAULT), positive = "1")
cmatrix_rf

#------------------------------------------------
# 8.5 ROC CURVES
#------------------------------------------------

# Calculate ROC curves for all models
roc_glm <- roc(response = validation_set$DEFAULT, predictor = glm_predicted_probs, plot = FALSE)       # Logistic Regression
roc_tree <- roc(response = validation_set$DEFAULT, predictor = tree_predicted_probs_1, plot = FALSE)  # Decision Tree
roc_rf <- roc(response = validation_set$DEFAULT, predictor = rf_probabilities, plot = FALSE)          # Random Forest
roc_knn <- roc(response = valid_target, predictor = positive_probs, levels = c("0", "1"))             # kNN

# Plot ROC curves
plot(roc_glm, col = "blue", main = "ROC Curves for Multiple Models") # Logistic Regression
plot(roc_tree, col = "red", add = TRUE)                             # Decision Tree
plot(roc_rf, col = "green", add = TRUE)                             # Random Forest
plot(roc_knn, col = "purple", add = TRUE)                           # kNN

# Add legend
legend("bottomright", 
       legend = c("Logistic Regression", "Decision Tree", "Random Forest", "kNN"),
       col = c("blue", "red", "green", "purple"), 
       lwd = 2, 
       cex = 0.8)

# Calculate AUC for all models
auc_glm <- auc(roc_glm)        # Logistic Regression
auc_tree <- auc(roc_tree)      # Decision Tree
auc_knn <- auc(roc_knn)        # kNN
auc_rf <- auc(roc_rf)          # Random Forest


# Print AUC results
cat("AUC for Logistic Regression:", auc_glm, "\n")
cat("AUC for Decision Tree:", auc_tree, "\n")
cat("AUC for kNN:", auc_knn, "\n")
cat("AUC for Random Forest:", auc_rf, "\n")

#==========================================
# Phase 9: ENHANCING RF MODEL TRANSPARENCY 
#==========================================

#------------------------------------------------
# 9.1 SHAP VALUES
#------------------------------------------------

set.seed(123)

predict_model <- function(model, newdata) {
  predict(model, newdata = newdata, type = "prob")[, 2]  # Probabilities for class 1 (default)
}

shap_values <- fastshap::explain(rf_model, 
                                 X = validation_set[, -which(names(validation_set) == "DEFAULT")], 
                                 pred_wrapper = predict_model, 
                                 nsim = 100)

#Creating df
shap_values_df<-as.data.frame(shap_values)

#Adding rf model probs to dataframe
shap_values_df$rf_probabilities<-rf_probabilities

head(shap_values_df,30)

#Obtaining the base probability of default from train set
base_prob <-sum(train_set$DEFAULT=="1")/nrow(train_set)
base_prob

# Sum the SHAP values across rows and add the base probability to each row
shap_values_df$shap_prob <- base_prob + rowSums(shap_values_df[, -which(names(shap_values_df) == "rf_probabilities")])

# View the first few rows to check the results
head(shap_values_df,10)
View (shap_values_df)

#------------------------------------------------
# 9.2 SHAP VS MODEL PREDICTIONS 
#------------------------------------------------

#========= 9.2.1 Mean Absolute Error=======#

# Absolute error
discrepancy <- abs(shap_values_df$rf_probabilities - shap_values_df$shap_prob)

# Mean Absolute Error (MAE)
mae <- mean(discrepancy)
mae

#========= 9.2.2 Plotting Model Probabilities vs SHAP Probabilities ========#

# Creating a data frame 
plot_shap_data <- shap_values_df[, c("rf_probabilities", "shap_prob")]

# Row index to use as the x-axis (record ID)
plot_shap_data$Record_ID <- 1:nrow(plot_shap_data)

# Limit the data to 200 records
plot_data_subset <- plot_shap_data[1:200, ]

# Create the line plot for the subset
ggplot(plot_data_subset, aes(x = Record_ID)) +
  geom_line(aes(y = rf_probabilities, color = "Random Forest Probabilities"), linewidth = 1) +
  geom_line(aes(y = shap_prob, color = "SHAP-Calculated Probabilities"), linewidth = 1) +
  labs(title = "Comparison of Random Forest vs SHAP-Calculated Probabilities (Subset)",
       x = "Record ID",
       y = "Probability") +
  scale_color_manual("", 
                     breaks = c("Random Forest Probabilities", "SHAP-Calculated Probabilities"),
                     values = c("Random Forest Probabilities" = "blue", "SHAP-Calculated Probabilities" = "red")) +
  theme_minimal()

#==========================================
# Phase 10: TEST PHASE
#==========================================

# Predicting probabilities for the test set using the trained Random Forest model
rf_test_probabilities <- predict(rf_model, test_set, type = "prob")[, 2]  # Get probabilities for class 1 (defaulters)

# SETTING THRESHOLD OF 0.2
rf_test_pred <- ifelse(rf_test_probabilities >= 0.2, 1, 0)

#CONFUSION MATRIX
cmatrix_test_set <- confusionMatrix(as.factor(rf_test_pred_threshold), as.factor(test_set$DEFAULT), positive ="1")
cmatrix_test_set

#==========================================
# Phase 11: DATA DRIVEN IMPROVEMENTS
#==========================================

#-----------------------------------------------------
# 11.1 PREVIOUS DEFAULT RATE VS ESTIMATED DEFAULT RATE
#-----------------------------------------------------

# Add predicted classifications to the test set
test_set$Predicted_Class <- rf_test_pred

# Filter for predicted non-default loans
predicted_non_defaults <- test_set[test_set$Predicted_Class == 0, ]

# Calculate the number of actual defaults in the predicted non-default subset
num_actual_defaults <- sum(predicted_non_defaults$DEFAULT == 1)

# Calculate the total number of predicted non-defaults
total_predicted_non_defaults <- nrow(predicted_non_defaults)

# Calculate the default rate among predicted non-defaults
estimated_default_rate<- num_actual_defaults / total_predicted_non_defaults

# Print the default rate
estimated_default_rate

# Create a data frame for plotting
default_rate_comparison <- data.frame(
  Category = c("Actual Default Rate", "Estimated Default Rate in Model-Based Approvals"),
  Default_Rate = c(percent_defaults, estimated_default_rate * 100)  # Ensure estimated_default_rate is converted to percentage
)

# Plot the comparison using ggplot2
library(ggplot2)

default_rate_plot <- ggplot(default_rate_comparison, aes(x = Category, y = Default_Rate, fill = Category)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Actual Default Rate" = "red", "Estimated Default Rate in Model-Based Approvals" = "green")) +
  ggtitle("Comparison of Actual Default Rate vs Estimated Default Rate with Model-Based Loan Approvals") +
  ylab("Default Rate (%)") +
  geom_text(aes(label = paste0(round(Default_Rate, 2), "%")), vjust = -0.3) +
  theme_minimal() +
  theme(legend.position = "none")  # Remove the legend

# Display the plot
print(default_rate_plot)

#-----------------------------------------------------
# 11.2 TOTAL LOAN AMOUNT VS TOTAL DEFAULTED AMOUNT 
#-----------------------------------------------------

# Step 1: Calculate loan amounts (Before Model Implementation)
total_loan_amount_before <- sum(imputed_loan$LOAN)
total_defaulted_loan_amount_before <- sum(imputed_loan$LOAN[imputed_loan$DEFAULT == 1])

# Step 2: Apply the model to the entire dataset (imputed_loan)
rf_probabilities_full <- predict(rf_model, newdata = imputed_loan, type = "prob")[, 2]
imputed_loan$Predicted_Default <- ifelse(rf_probabilities_full >= 0.2, 1, 0)

# Step 3: Calculate loan amounts (After Model Implementation)
total_loan_amount_after <- sum(imputed_loan$LOAN[imputed_loan$Predicted_Default == 0])
total_defaulted_loan_amount_after <- sum(imputed_loan$LOAN[imputed_loan$Predicted_Default == 0 & imputed_loan$DEFAULT == 1], na.rm = TRUE)


# Plot 1: Total Loans Issued vs Defaulted (Pre-Model)
before_plot <- ggplot(data.frame(
  Category = c("Issued Loans", "Defaulted Loans"),
  Amount = c(total_loan_amount_before, total_defaulted_loan_amount_before)
), aes(x = Category, y = Amount, fill = Category)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Issued Loans" = "blue", "Defaulted Loans" = "red")) +
  ggtitle("Total Loans Issued vs Defaulted (Pre-Model)") +
  ylab("Amount (in US Dollars)") +
  geom_text(aes(label = scales::comma(Amount)), vjust = -0.3) +
  theme_minimal() +
  theme(legend.position = "none")

# Plot 2: Estimated Loans Issued vs Defaulted (Post-Model)
after_plot <- ggplot(data.frame(
  Category = c("Issued Loans", "Defaulted Loans"),
  Amount = c(total_loan_amount_after, total_defaulted_loan_amount_after)
), aes(x = Category, y = Amount, fill = Category)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Issued Loans" = "green", "Defaulted Loans" = "orange")) +
  ggtitle("Estimated Loans Issued vs Defaulted (Post-Model)") +
  ylab("Amount (in US Dollars)") +
  geom_text(aes(label = scales::comma(Amount)), vjust = -0.3) +
  theme_minimal() +
  theme(legend.position = "none")

# Arrange plots side by side
grid.arrange(before_plot, after_plot, ncol = 2)

head(imputed_loan)

#-----------------------------------------------------
# 11.3 TOTAL LOAN AMOUNT VS TOTAL DEFAULTED AMOUNT 
#-----------------------------------------------------

# Load required libraries
library(ggplot2)
library(gridExtra)

# Step 1: Calculate totals (before model implementation)
definitive_loss_before <- sum(imputed_loan$DEFINITIVE_LOSS, na.rm = TRUE)
recoverable_loss_before <- sum(imputed_loan$LOAN[imputed_loan$DEFAULT == 1], na.rm = TRUE) - definitive_loss_before

# Step 2: Calculate totals (after model implementation)
definitive_loss_after <- sum(imputed_loan$DEFINITIVE_LOSS[imputed_loan$Predicted_Default == 0], na.rm = TRUE)
recoverable_loss_after <- sum(imputed_loan$LOAN[imputed_loan$DEFAULT == 1 & imputed_loan$Predicted_Default == 0], na.rm = TRUE) - definitive_loss_after

# Step 3: Create the first plot (Before Model)
before_plot <- ggplot(data.frame(
  Metric = c("Definitive Loss", "Recoverable Loss"),
  Amount = c(definitive_loss_before, recoverable_loss_before)
), aes(x = Metric, y = Amount, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Definitive Loss" = "red", "Recoverable Loss" = "blue")) +
  ggtitle("Losses Before Model Implementation") +
  ylab("Amount (in US Dollars)") +
  geom_text(aes(label = scales::comma(Amount)), vjust = -0.3) +
  theme_minimal() +
  theme(legend.position = "none")

# Step 4: Create the second plot (After Model)
after_plot <- ggplot(data.frame(
  Metric = c("Definitive Loss", "Recoverable Loss"),
  Amount = c(definitive_loss_after, recoverable_loss_after)
), aes(x = Metric, y = Amount, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5) +
  scale_fill_manual(values = c("Definitive Loss" = "orange", "Recoverable Loss" = "green")) +
  ggtitle("Losses After Model Implementation") +
  ylab("Amount (in US Dollars)") +
  geom_text(aes(label = scales::comma(Amount)), vjust = -0.3) +
  theme_minimal() +
  theme(legend.position = "none")

# Step 5: Arrange the two plots side by side
grid.arrange(before_plot, after_plot, ncol = 2)


summary(imputed_loan)
str(imputed_loan)
