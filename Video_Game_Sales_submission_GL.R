# Video Games Sales prediction model
# The original data set has many different variables for Sales: "Global","NA","EU","JP","Other"
# Since the Global sales relies on the other region sub-categories, prediction will be for NA Sales levels


#########################################################
# Download Dataset and create training and testing sets #
#########################################################

# Prepare Packages and install them if they are not installed on the user's machine
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(Amelia)) install.packages("Amelia", repos = "http://cran.us.r-project.org")


# Data set originally downloaded from this website and uploaded to GitHub as part of capstone
# https://www.kaggle.com/annavictoria/ml-friendly-public-datasets?utm_medium=email&utm_source=intercom&utm_campaign=data+projects+onboarding
# https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings

data_url <- "https://raw.githubusercontent.com/qquu50/Capstone/master/Video_Games_Sales.csv"


# Download the Video Game Sales into a staging table
# It will then be modified to induce the Scores, Counts, and Year of Release into numerics
# NAs will be introduced by coercion because many games are missing User/Critic scores
raw_vg_table <- read.csv(data_url, stringsAsFactors = FALSE) %>% filter(!is.na(Name))
raw_vg_table$Critic_Score <- round(as.numeric(raw_vg_table$Critic_Score), digits = 0)
raw_vg_table$Critic_Count <- as.numeric(raw_vg_table$Critic_Count)
raw_vg_table$User_Score <- round(as.numeric(raw_vg_table$User_Score), digits = 0)
raw_vg_table$User_Count <- as.numeric(raw_vg_table$User_Count)
raw_vg_table$Year_of_Release <- as.numeric(raw_vg_table$Year_of_Release)


# Add NA Sales Ranking to table
# This ranking will just be to see the count of games for each bucket
# All sales figures are "millions of units"
raw_vg_table <- raw_vg_table %>% mutate(Sales_Rank = ifelse(NA_Sales >= 1, 6,
                                                            ifelse(NA_Sales >= 0.5, 5, 
                                                                   ifelse(NA_Sales >= 0.25, 4, 
                                                                          ifelse(NA_Sales >= 0.1, 3, 
                                                                                 ifelse(NA_Sales >= 0.05, 2, 1))))))

raw_vg_table %>% group_by(Sales_Rank) %>% summarize(n = n())
rank_names <- c("Below 50,000","50,000 - 100,000","100,000 - 250,000","250,000 - 500,000","500,000 - 1 million","Above 1 million units sold")
ranking_list <- raw_vg_table %>% group_by(Sales_Rank) %>% summarize(n = n()) %>% as.data.frame()
rownames(ranking_list) <- rank_names
ranking_list <- ranking_list[order(-ranking_list$Sales_Rank),]
print(ranking_list)
raw_vg_table %>% top_n(10, NA_Sales)

# Setting up Function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Set up training and validation sets 
# Partition Data
# 80/20 split to generate the train(80) and test(20) set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = raw_vg_table$NA_Sales, times = 1, p = 0.2, list = FALSE)
train_set <- raw_vg_table[-test_index,]
test_set <- raw_vg_table[test_index,]

# missmap function maps out the missing values per variable
# In this situation, it's missing many of the User/Critic Scores and Counts so those fields may not be useful in the modelling
missmap(train_set, main = 'Missing Map', col = c('yellow','black'), legend = TRUE, x.cex = 0.8, margins = c(6,4))


# Due to the variability in NA Sales, it will be split into several buckets and a prediction will be made based on these
train_set %>% 
  ggplot(aes(NA_Sales)) + 
  geom_histogram(bins = 10) + 
  xlab("NA Sales") + 
  ylab("Count of Games") +
  ggtitle("Raw NA Sales") +
  theme_economist()
train_set %>% 
  ggplot(aes(Sales_Rank)) + 
  geom_histogram(binwidth = 1) + 
  xlab("NA Sales Ranking") + 
  ylab("Count of Games") +
  ggtitle("Ranking of NA Sales (Higher number means more sales)") +
  theme_economist()
train_set %>% 
  filter(NA_Sales <= 0.5) %>% 
  ggplot(aes(NA_Sales)) + 
  geom_histogram(bins = 10) + 
  xlab("NA Sales") + 
  ylab("Count of Games") +
  ggtitle("Sales below 500,000") + 
  theme_economist()

# Check Correlation between numerics
numeric_cols <- sapply(train_set, is.numeric)
correlation_data <- cor(train_set[,numeric_cols])
corrplot(correlation_data)

# Just the Average
# As a baseline, check the predictive capability if we take the average NA Sales and use that to predict the test set's sales
mu_hat <- mean(train_set$NA_Sales)
naive_rmse <- RMSE(test_set$NA_Sales, mu_hat)
rmse_results <- data_frame(method = "Just the average", RMSE = naive_rmse)

############################
# Platform Effect Model
# Testing the effects that different video game platforms have on sales in the NA region
rmses <- function(){
  mu <- mean(test_set$NA_Sales)
  platform_avg <- test_set %>%
    group_by(Platform) %>%
    summarize(b_i = mean(NA_Sales - mu))
  predicted_sales <- mu + test_set %>%
    left_join(platform_avg, by = "Platform") %>%
    .$b_i
  return(RMSE(predicted_sales, test_set$NA_Sales))
}
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Platform Effect Model",  
                                     RMSE = min(rmses())))

############################
# Platform + Genre Effect Model
# Building on the previous model, check to see if the Genres of games have an impact on sales in NA
rmses <- function(){
  mu <- mean(test_set$NA_Sales)
  platform_avg <- test_set %>%
    group_by(Platform) %>%
    summarize(b_i = mean(NA_Sales - mu))
  genre_avg <- test_set %>%
    left_join(platform_avg, by = 'Platform') %>%
    group_by(Genre) %>%
    summarize(b_g = mean(NA_Sales - mu - b_i))
  predicted_sales <- mu + test_set %>%
    left_join(platform_avg, by = "Platform") %>%
    left_join(genre_avg, by = "Genre") %>%
    mutate(pred = mu + b_i + b_g) %>%
    .$pred
  return(RMSE(predicted_sales, test_set$NA_Sales))
}
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Platform + Genre Effect Model",  
                                     RMSE = min(rmses())))

############################
# Regularized Platform Effect Model
# Regularizing the original model that relies only on Platform variable to predict NA Sales
# Check to see if regularizing will improve predictions
# Train to find the best lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$NA_Sales)
  b_i <- train_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  predicted_ratings <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, train_set$NA_Sales))
})

lambda <- lambdas[which.min(rmses)]

# Test prediction capability with the lambda from the previous step
# Using test set for this function
rmses <- sapply(lambda, function(l){
  mu <- mean(test_set$NA_Sales)
  b_i <- test_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$NA_Sales))
})

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Platform Effect Model",  
                                     RMSE = min(rmses)))

############################
# Regularized Platform + Genre Effect Model
# Regularizing the original model that relies on Platform and Genre variables to predict NA Sales
# Check to see if regularizing will improve predictions
# Train to find the best lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$NA_Sales)
  b_i <- train_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  b_g <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    group_by(Genre) %>%
    summarize(b_g = sum(NA_Sales - b_i - mu)/(n()+l))
  predicted_ratings <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>% 
    mutate(pred = mu + b_i + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, train_set$NA_Sales))
})

lambda <- lambdas[which.min(rmses)]

# Test prediction capability with the lambda from the previous step
# Using test set for this function
rmses <- sapply(lambda, function(l){
  mu <- mean(test_set$NA_Sales)
  b_i <- test_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  b_g <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    group_by(Genre) %>%
    summarize(b_g = sum(NA_Sales - b_i - mu)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    mutate(pred = mu + b_i + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$NA_Sales))
})

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Platform + Genre Effect Model",  
                                     RMSE = min(rmses)))

############################
# Regularized Platform + Genre + Critic Score + User Score Effect Model
# Expanding model to include Critic and User Scores
# Train to find the best lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$NA_Sales)
  b_i <- train_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  b_g <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    group_by(Genre) %>%
    summarize(b_g = sum(NA_Sales - b_i - mu)/(n()+l))
  b_c <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    group_by(Critic_Score) %>%
    summarize(b_c = sum(NA_Sales - b_i - b_g - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    left_join(b_c, by= "Critic_Score") %>%
    group_by(User_Score) %>%
    summarize(b_u = sum(NA_Sales - b_i - b_g - b_c - mu)/(n()+l))
  predicted_ratings <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>% 
    left_join(b_c, by= "Critic_Score") %>%
    left_join(b_u, by= "User_Score") %>%
    mutate(pred = mu + b_i + b_g + b_c + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, train_set$NA_Sales))
})

lambda <- lambdas[which.min(rmses)]

# Test prediction capability with the lambda from the previous step
# Using test set for this function
rmses <- sapply(lambda, function(l){
  mu <- mean(test_set$NA_Sales)
  b_i <- test_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  b_g <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    group_by(Genre) %>%
    summarize(b_g = sum(NA_Sales - b_i - mu)/(n()+l))
  b_c <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    group_by(Critic_Score) %>%
    summarize(b_c = sum(NA_Sales - b_i - b_g - mu)/(n()+l))
  b_u <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    left_join(b_c, by= "Critic_Score") %>%
    group_by(User_Score) %>%
    summarize(b_u = sum(NA_Sales - b_i - b_g - b_c - mu)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>% 
    left_join(b_c, by= "Critic_Score") %>%
    left_join(b_u, by= "User_Score") %>%
    mutate(pred = mu + b_i + b_g + b_c + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$NA_Sales))
})

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Platform + Genre + Critic Score + User Score Effect Model",  
                                     RMSE = min(rmses)))

############################
# Regularized Platform + Genre + Critic Score + User Score + Year of Release Effect Model
# Further expansion of previous data model to take Year of Release into account
# Train to find the best lambda
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$NA_Sales)
  b_i <- train_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  b_g <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    group_by(Genre) %>%
    summarize(b_g = sum(NA_Sales - b_i - mu)/(n()+l))
  b_c <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    group_by(Critic_Score) %>%
    summarize(b_c = sum(NA_Sales - b_i - b_g - mu)/(n()+l))
  b_u <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    left_join(b_c, by= "Critic_Score") %>%
    group_by(User_Score) %>%
    summarize(b_u = sum(NA_Sales - b_i - b_g - b_c - mu)/(n()+l))
  b_y <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    left_join(b_c, by= "Critic_Score") %>%
    left_join(b_u, by= "User_Score") %>%
    group_by(Year_of_Release) %>%
    summarize(b_y = sum(NA_Sales - b_i - b_g - b_c - b_u - mu)/(n()+l))
  predicted_ratings <- train_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>% 
    left_join(b_c, by= "Critic_Score") %>%
    left_join(b_u, by= "User_Score") %>%
    left_join(b_y, by="Year_of_Release") %>%
    mutate(pred = mu + b_i + b_g + b_c + b_u + b_y) %>%
    .$pred
  return(RMSE(predicted_ratings, train_set$NA_Sales))
})

lambda <- lambdas[which.min(rmses)]

# Test prediction capability with the lambda from the previous step
# Using test set for this function
rmses <- sapply(lambda, function(l){
  mu <- mean(test_set$NA_Sales)
  b_i <- test_set %>%
    group_by(Platform) %>%
    summarize(b_i = sum(NA_Sales - mu)/(n()+l))
  b_g <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    group_by(Genre) %>%
    summarize(b_g = sum(NA_Sales - b_i - mu)/(n()+l))
  b_c <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    group_by(Critic_Score) %>%
    summarize(b_c = sum(NA_Sales - b_i - b_g - mu)/(n()+l))
  b_u <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    left_join(b_c, by= "Critic_Score") %>%
    group_by(User_Score) %>%
    summarize(b_u = sum(NA_Sales - b_i - b_g - b_c - mu)/(n()+l))
  b_y <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>%
    left_join(b_c, by= "Critic_Score") %>%
    left_join(b_u, by= "User_Score") %>%
    group_by(Year_of_Release) %>%
    summarize(b_y = sum(NA_Sales - b_i - b_g - b_c - b_u - mu)/(n()+l))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "Platform") %>%
    left_join(b_g, by = "Genre") %>% 
    left_join(b_c, by= "Critic_Score") %>%
    left_join(b_u, by= "User_Score") %>%
    left_join(b_y, by="Year_of_Release") %>%
    mutate(pred = mu + b_i + b_g + b_c + b_u + b_y) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$NA_Sales))
})

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Platform + Genre + Critic Score + User Score + Year of Release Effect Model",  
                                     RMSE = min(rmses)))
rmse_results