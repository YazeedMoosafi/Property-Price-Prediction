##############################
# Name: Prprty_script.R
# This script is used to develop and train machine learning models to predict the property prices in NYC.
##############################


#Install and load the required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Rborist         )) install.packages("Rborist"        , repos = "http://cran.us.r-project.org")
if(!require(randomForest    )) install.packages("randomForest"   , repos = "http://cran.us.r-project.org")
if(!require(lars           )) install.packages("lars"   , repos = "http://cran.us.r-project.org")
if(!require(elasticnet    )) install.packages("elasticnet"   , repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(Rborist         )
library(randomForest         )
library(lars         )
library(elasticnet         )

#Download the data set from github
raw_data <- read.csv("https://raw.githubusercontent.com/YazeedMoosafi/Covid-Dataset/main/nyc-rolling-sales.csv",       sep = ",")

#Data Exploration
plot(as.numeric(raw_data$SALE.PRICE), main = "Checking Sale Price data for outliers")
plot(raw_data$YEAR.BUILT, main = "Checking Built year Data for outliers")
plot(as.numeric(raw_data$GROSS.SQUARE.FEET), main = "Checking Gross Square feet data for outliers")
plot(as.numeric(raw_data$LAND.SQUARE.FEET), main = "Checking Land Square feet data for outliers")
plot(raw_data$BLOCK, main = "Checking Block values for outliers")
plot(raw_data$BOROUGH, main = "Checking Borough values for outliers")
plot(raw_data$TAX.CLASS.AT.TIME.OF.SALE, main = "Checking Tax class values for outliers")
plot(raw_data$ZIP.CODE, main = "Checking Zip code values for outliers")

#Looking for invalid values
head(sort(unique(raw_data$SALE.PRICE))) 
tail(sort(unique(raw_data$SALE.PRICE)))
head(sort(unique(raw_data$YEAR.BUILT))) 
tail(sort(unique(raw_data$YEAR.BUILT)))
head(sort(unique(raw_data$GROSS.SQUARE.FEET))) 
tail(sort(unique(raw_data$GROSS.SQUARE.FEET)))
head(sort(unique(raw_data$LAND.SQUARE.FEET))) 
tail(sort(unique(raw_data$LAND.SQUARE.FEET)))
head(sort(unique(raw_data$ZIP.CODE))) 
tail(sort(unique(raw_data$ZIP.CODE)))

# Data cleansing by removing outliers
# Sale price should be between 50,000 and 10 million
# Remove rows that are having invalid values for predictors as well
# Round the SALE.PRICE values and convert the predictors to numeric values
# Remove invalid values from sale price, built year, gross square feet, land square feet, zip code   
# Gross and land square feet should be less than 50,000  
# Built year should be grated than 1875.  

data <- raw_data %>% filter(!SALE.PRICE %in% " -  " ) %>% filter(as.numeric(SALE.PRICE) > 50000) %>% 
  filter(!YEAR.BUILT %in% c("0", "1111"))  %>%  filter(!GROSS.SQUARE.FEET %in% c("0", " -  ") )  %>% 
  filter(as.numeric(SALE.PRICE) < 10000000) %>% 
  filter(!ZIP.CODE %in% c("0")) %>% filter(!LAND.SQUARE.FEET %in% c(" -  ")) %>% 
  filter(as.numeric(GROSS.SQUARE.FEET) < 50000  & as.numeric(LAND.SQUARE.FEET)< 50000 & 
           as.numeric(LOT)< 1000 & as.numeric(YEAR.BUILT) > 1875 ) %>%
  mutate(GROSS.SQUARE.FEET= as.numeric(GROSS.SQUARE.FEET), SALE.PRICE= round(as.numeric(SALE.PRICE), -1),
         YEAR.BUILT = as.numeric(YEAR.BUILT), SALE.YEAR=as.numeric(substr(SALE.DATE, 1,4)), 
         SALE.MONTH=as.numeric(substr(SALE.DATE, 6,7)),
         BUILDING.CLASS.AT.TIME.OF.SALE=as.factor(BUILDING.CLASS.AT.TIME.OF.SALE),
         ZIP.CODE = as.numeric(ZIP.CODE), NEIGHBORHOOD=as.factor(NEIGHBORHOOD),
         LAND.SQUARE.FEET=as.numeric(LAND.SQUARE.FEET),
         BOROUGH=as.numeric(BOROUGH),  TAX.CLASS.AT.TIME.OF.SALE=as.numeric(TAX.CLASS.AT.TIME.OF.SALE),
         LOT=as.numeric(LOT),BLOCK=as.numeric(BLOCK)
  )

nrow(data)
head(data)


#Check for invalid values and outliers after data cleansing
plot(data$SALE.PRICE, main = "Validating Sale Price data after cleansing")
plot(data$YEAR.BUILT, main = "Validating Built year data after cleansing")
plot(data$GROSS.SQUARE.FEET, main = "Validating Gross Square feet data after cleansing")
plot(data$LAND.SQUARE.FEET, main = "Validating Land Square feet data after cleansing")
plot(data$BLOCK, main = "Validating Block values after cleansing")
plot(data$BOROUGH, main = "Validating Borough values after cleansing")
plot(data$TAX.CLASS.AT.TIME.OF.SALE, main = "Validating Tax class values after cleansing")
plot(data$ZIP.CODE, main = "Validating Zip code values after cleansing")

#Looking for invalid values after data cleansing
head(sort(unique(data$SALE.PRICE))) 
tail(sort(unique(data$SALE.PRICE)))
head(sort(unique(data$YEAR.BUILT))) 
tail(sort(unique(data$YEAR.BUILT)))
head(sort(unique(data$GROSS.SQUARE.FEET))) 
tail(sort(unique(data$GROSS.SQUARE.FEET)))
head(sort(unique(data$LAND.SQUARE.FEET))) 
tail(sort(unique(data$LAND.SQUARE.FEET)))
head(sort(unique(data$ZIP.CODE))) 
tail(sort(unique(data$ZIP.CODE)))



#Split the data set into train_set and test_set
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = data$SALE.PRICE, times = 1,   p = 0.2, list = FALSE)
train_set <- data[-test_index,]
test_set <- data[test_index,]



#list of models we are using
models <- c("lm", "glm", "Rborist", "svmLinear", "knn","lasso", "ridge" )



# Model Training
fits <- lapply(models, function(model){ 
  print(model)
      train(SALE.PRICE ~ GROSS.SQUARE.FEET+YEAR.BUILT+  BOROUGH+ ZIP.CODE+ LAND.SQUARE.FEET+ TAX.CLASS.AT.TIME.OF.SALE+ BLOCK  , method = model, data = train_set
      )
}) 


names(fits) <- models

#Prediction
pred <- sapply(fits, function(object){
  predict(object, newdata = test_set)
})

#Check the accuracy
RMSE(as.data.frame(pred)$Rborist              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$lm              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$glm              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$Rborist              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$svmLinear              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$knn              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$lasso              , test_set$SALE.PRICE)
RMSE(as.data.frame(pred)$ridge              , test_set$SALE.PRICE)

#Tabulate the different RMSE values
rmse_results <- data_frame(Method = "lm", RMSE = RMSE(as.data.frame(pred)$lm , test_set$SALE.PRICE))
rmse_results <- bind_rows(rmse_results, data_frame(Method="glm",       RMSE = RMSE(as.data.frame(pred)$glm , test_set$SALE.PRICE) ))
rmse_results <- bind_rows(rmse_results, data_frame(Method="Rborist",   RMSE = RMSE(as.data.frame(pred)$Rborist , test_set$SALE.PRICE) ))
rmse_results <- bind_rows(rmse_results, data_frame(Method="svmLinear", RMSE = RMSE(as.data.frame(pred)$svmLinear , test_set$SALE.PRICE) ))
rmse_results <- bind_rows(rmse_results, data_frame(Method="knn",       RMSE = RMSE(as.data.frame(pred)$knn , test_set$SALE.PRICE) ))
rmse_results <- bind_rows(rmse_results, data_frame(Method="lasso",     RMSE = RMSE(as.data.frame(pred)$lasso , test_set$SALE.PRICE) ))
rmse_results <- bind_rows(rmse_results, data_frame(Method="ridge",     RMSE = RMSE(as.data.frame(pred)$ridge , test_set$SALE.PRICE) ))


rmse_results %>% knitr::kable()

#Pick the best RMSE
RMSE(as.data.frame(pred)$Rborist, test_set$SALE.PRICE)



#Fine Turning the Rborist model

control <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

# Model Training
fits <- train(SALE.PRICE ~ GROSS.SQUARE.FEET+YEAR.BUILT+  BOROUGH+ ZIP.CODE+ LAND.SQUARE.FEET+ TAX.CLASS.AT.TIME.OF.SALE+ BLOCK  , method = "Rborist", data = train_set,  trControl=control)

#names(fits) <- "Rborist"

#Prediction
pred <- predict(fits, newdata = test_set)

#Check the accuracy
FINAL_RMSE <- RMSE(pred, test_set$SALE.PRICE)

#Print the final RMSE 
FINAL_RMSE


