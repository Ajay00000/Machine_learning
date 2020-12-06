# Multiple Linier Regression

data = read.csv("50_Startups.csv")
data$State = factor(data$State,levels=c('New York','California','Florida'),labels=c(1,2,3
))

set.seed(123)
smp_size <- floor(0.8 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
test <- data[-train_ind, ]

regressor = lm(formula = Profit ~. ,data =  train)
y_pred = predict(regressor,newdata = test)
# Linier Regression with All variables
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,data = train)
y_pred = predict(regressor,newdata = test)
summary(regressor)

# Linier regression without state column

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,data = train)
y_pred = predict(regressor,newdata = test)
summary(regressor)

# linier regression without Administrationn
regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend ,data = train)
y_pred = predict(regressor,newdata = test)
summary(regressor)

# linier regression without Marketing.Spend

regressor = lm(formula = Profit ~ R.D.Spend  ,data = train)
y_pred = predict(regressor,newdata = test)
summary(regressor)




