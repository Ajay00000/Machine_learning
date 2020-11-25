
data = read.csv("sallary.csv")

set.seed(123)
smp_size <- floor(2/3 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

# Fitting Simple Regression To the Training Set

regressor = lm(formula = Sallary ~ YearsOfExperience, data=train)

ypred = predict(regressor,newdata = test)

#visualizing Result

library(ggplot2)
ggplot()+
  geom_point(aes(x=train$YearsOfExperience,y=train$Sallary),
             colour = "red") +
  geom_line(aes(x=train$YearsOfExperience,y=predict(regressor,newdata = train)),
            colour="blue") +
  ggtitle(" Sallary Vs Years Of Experience (Training Set)") +
  xlab(" Years Of Experience ") +
  ylab(" Sallary ")

# Visualise on Test Set Data
ggplot()+
  
  geom_point(aes(x=test$YearsOfExperience,y=test$Sallary),
           colour = "red") +
  geom_line(aes(x=train$YearsOfExperience,y=predict(regressor,newdata = train)),
            colour="blue") +
  ggtitle(" Sallary Vs Years Of Experience (Testing Set)") +
  xlab(" Years Of Experience ") +
  ylab(" Sallary ")

