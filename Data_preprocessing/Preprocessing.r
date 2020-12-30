getwd()
setwd("/home/aj/Desktop")

data = read.csv("/home/aj/Desktop/MachineLearning/Data_preprocessing/Data.csv")
data$age = ifelse(is.na(data$age),ave(data$age,FUN = function(x) mean(x,na.rm=TRUE)),data$age)
data$sallary = ifelse(is.na(data$sallary),ave(data$sallary, FUN = function(x) mean(x,na.rm = TRUE)),data$sallary)

data$country = factor(data$country,levels = c('france','spain','germany'),
                      labels = c(1,2,3))

data$purchased = factor(data$purchased,levels = c('no','yes'),
                      labels = c(4,1))
prcomp(data)
# spliting data training set test set

set.seed(123)
smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

# Feature Scaling
train[,2:3] = scale(train[,2:3])
test[,2:3] = scale(test[,2:3])
