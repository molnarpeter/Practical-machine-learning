#########################################################################

#Loading the Caret package and the dataset with the option that specify which values should be interpret as NA values

library("caret", lib.loc="C:/R/R-3.0.2/library")

setwd("C:/Users/Studio-XPS/Desktop/coursera/Applied ML")

raw_data <- read.csv("pml-training.csv", na.strings=c("","NA","#DIV/0"))

View(raw_data)

#########################################################################

#As it can be seen the first seven column of the dataset contain information which are not to be used for prediction,
#that's they should be removed:

data <- raw_data[,8:160]

#Still, there are 153 varibale, so the exploratory data analysis would be very hard, fortunately it can be noticed easily
#that there many variables with a lot of NA's and with values equal for all or almost all observations. To select the variables
#in which there is variance, one can use the nearZeroVar command.

poss_preds <- nearZeroVar(data,saveMetrics=TRUE)[nearZeroVar(data,saveMetrics=TRUE)[,"zeroVar"] == FALSE,]

names <- row.names(poss_preds)

#Selecting variables with variance:

predictors <- data[, names]

#Still there are variables which has NA values, and they will be filtered as well:

full_predictors <- (apply(predictors, 2, function(x) { sum(is.na(x)) }) == 0)

full_predictors <- as.list(full_predictors[full_predictors == TRUE])

fin_predictors <- predictors[,names(full_predictors)]

#Now the preliminary data manipulation is ready the next stage is data partition the seed is set 
#because of the reproducibilty issues

set.seed(1989)
partition = createDataPartition(fin_predictors$classe, p=0.8, list=FALSE)
training_set = fin_predictors[partition,]
test_set = fin_predictors[-partition,]

#So the database is fairly large and there is 52 predictors left which is a lot as well. Therefore the possible methods 
#are (in my opinion): classification tree, regularized regression or random forest because these methods can handle the 
#selection of the relevant variables. Another way could be to compress the data with principal component analysis (PCA)
#but I don't really prefer that method due to the loss of information.



model_cart <- train(classe ~ ., data = training_set, method = "rpart", prox = TRUE, trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

prediction_cart <- predict(model_cart, newdata=test_set)

cM_test_cart <- confusionMatrix(prediction_cart, test_set$classe)

model_multinom <- train(classe ~ ., data = training_set, method = "multinom", prox = TRUE, trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

prediction_multinom <- predict(model_multinom, newdata=test_set)

cM_test_multinom <- confusionMatrix(prediction_multinom, test_set$classe)

model_glmnet <- train(classe ~ ., data = training_set, method = "glmnet", prox = TRUE, trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

prediction_glmnet <- predict(model_glmnet, newdata=test_set)

cM_test_glmnet <- confusionMatrix(prediction_glmnet, test_set$classe)

model_rf <- train(classe ~ ., data = training_set, method = "rf", prox = TRUE, trControl = trainControl(method = "cv", number = 4, allowParallel = TRUE))

prediction_rf <- predict(model_rf, newdata=test_set)

cM_test_rf <- confusionMatrix(prediction_rf, test_set$classe)

# From the confusion matrices it can be seen that the random forest is far the best method (sadly the one with the longest computation time as well)
#The random forest is used for the prediction

raw_test <- read.csv("pml-testing.csv", na.strings=c("","NA","#DIV/0"))

test_predictors <- raw_test[,names(full_predictors[1:52])]

test_predictions <- predict(model_rf, newdata = test_predictors)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(test_predictions)

