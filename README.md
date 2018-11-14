Predictive-models-for-direct-marketing
Building a good quality maximum profit predictive model for an ad campaign

This project was done in R using the Rpart, Caret, and ROCR packages

Project overview:
A	bank is	marketing	Certificates	of	Deposit	(CD)	to	a	pool	of	1,000	potential	customers.The	bank estimates that	13%	of them are	likely	to	buy	a	CD	if	they	are	contacted.The	cost of	eachcontact	is $10. The	NPV	to	the	bankof a customer	buying	a CD is $50. Clearly,	it is not	profitable	to	contact	all	the	potential	customers. So, instead the	bank	employs	a	predictive	model	to	predict	whether or not a potential	customeris a	likely	buyer. It	has	a	data	set	(bank.csv) of	some customersand	whether	theybought the CD or not.	

# Only sending ads to predicted positive potential customer
# Predicted positive: 130
# Revenue: 130*50 = 6500(1-b)
# cost : 1300(1-b)+8700(a)
# Net Profit:5200-5200(beta)-8700(alpha)

# Set your working directory to wherever you saved the csv file
setwd('C:/Users/Jack/Documents/CIS 417/R')
library(readr)

bank <- read_csv("bank.csv")

# Create a training data set to build the model, and a test data set to test the model
# Training data set contains 2/3 of original data test contains 1/3
set.seed(644)
train = sample(1:nrow(bank),0.66667*nrow(bank))
bank.train = bank[train,]
bank.test = bank[-train,]


# install.packages rpart, and caret in rstudio if not previously installed
library(rpart)
library(caret)

# Growing the biggest tree descriptive tree
#  xval = 10 : cross-sample with 10 folds to determine error rate at each node
#  minsplit = 2  : min number of observations to attempt split
#  cp = 0  : minimum improvement in complexity parameter for splitting
# smaller minsplit and cp result in larger trees
                
# Min split set to 10 cp set to 0 to produce large tree
fit = rpart(y ~ ., 
            data=bank.train,
            control=rpart.control(xval=10, minsplit=10, cp=0))
# 213 nodes
nrow(fit$frame)

# Creation of large tree
plot(fit, 
     uniform=T, 
     branch=0.5, 
     compress=T, 
     main="Large Tree",
     margin=0.0) 
text(fit,  
     splits=F, 
     all=F, 
     use.n=T, 
     pretty=F, 
     cex=0.6) 

# Find the best cp, the one with the smallest xerror
bestcp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]

# Pre prune the tree model with the best CP
# Use zoom tab on chart area for cleaner view of the chart
fit.small2 = rpart(y ~ ., 
                   data=bank.train,
                   control=rpart.control(xval=10, cp=bestcp))
plot(fit.small2, uniform=T, branch=0.5, compress=T,
     main="Tree with best cp", margin=0.1)
text(fit.small2,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2)

# Counts number of nodes
nrow(fit.small2$frame)

# Creates confusion matrix for test data set 
cm <- confusionMatrix(table(pred=predict(fit.small2, bank.test, type="class"),actual = bank.test$y), positive = 'yes')
cm
# Type 1 error alpha
alpha1 <-  (1-cm$byClass["Specificity"][[1]])

# Type 2 error beta
beta1 <-  (1-cm$byClass["Sensitivity"][[1]])

# Expected profit 
# 5200(TPR)-8700(FPR)
Exp_Profit <- 5200 - 5200*(beta1)-8700*(alpha1) 
Exp_Profit


# Creates model using balanced data set of bal by subsampling
# Balanced data has equal number of yes and no's
b.train.yes <- bank.train[bank.train$y =='yes',]
b.train.no <- bank.train[bank.train$y == 'no',]

set.seed(234)
bal = sample(1:nrow(b.train.yes),nrow(b.train.yes))
b.train.no = b.train.no[bal,]

b.bal <- rbind(b.train.no,b.train.yes)


Bfit = rpart(y ~ ., 
            data=b.bal,
            control=rpart.control(xval=10, minsplit=10, cp=0))
# 85 nodes
nrow(Bfit$frame)

# Creation of large tree
plot(Bfit, 
     uniform=T, 
     branch=0.5, 
     compress=T, 
     main="Large Tree",
     margin=0.0) 
text(Bfit,  
     splits=F, 
     all=F, 
     use.n=T, 
     pretty=F, 
     cex=0.6) 


Balbestcp = Bfit$cptable[which.min(fit$cptable[,"xerror"]),"CP"]


fit.smallBal = rpart(y ~ ., 
                   data=b.bal,
                   control=rpart.control(xval=10, cp=Balbestcp))
plot(fit.smallBal, uniform=T, branch=0.5, compress=T,
     main="Tree with best cp", margin=0.1)
text(fit.smallBal,  splits=T, all=F, use.n=T, 
     pretty=T, fancy=F, cex=1.2)

# 15 nodes
nrow(fit.smallBal$frame)

cm2 = confusionMatrix(table(pred=predict(fit.smallBal, bank.test, type="class"),
                                  actual = bank.test$y), positive = 'yes')
cm2
# Type 1 error
alpha2 <- (1-cm2$byClass["Specificity"][[1]])

# Type 2 error
Beta2 <-  (1-cm2$byClass["Sensitivity"][[1]])

Exp_Profit2 <- 5200 -5200*(Beta2)-8700*(alpha2) 
Exp_Profit2  



# Now find the optimal cutoff for the model from the balanced training data set.
library(ROCR)
Ad.pred = as.data.frame(predict(fit.smallBal, b.bal, type="prob"))
head(Ad.pred)

Ad.pred.score =  prediction(Ad.pred[,2], b.bal$y)

Ad.pred.perf = performance(Ad.pred.score, "tpr", "fpr")

plot(Ad.pred.perf, 
     colorize=T, 
     lwd=4) 
abline(0,1)  
abline(h=1) 
abline(v=0)

performance(Ad.pred.score, "auc")@y.values


cm3 = confusionMatrix(table(pred=predict(fit.smallBal, b.bal, type="class"),
                           actual = b.bal$y), positive='yes')

Ad.cost = performance(Ad.pred.score, measure="cost", 
                         cost.fn=5200, cost.fp=8700)

cutoff.best = Ad.cost@x.values[[1]][which.min(Ad.cost@y.values[[1]])]

Ad.pred.test = predict(fit.smallBal, b.bal, type="prob")

Ad.pred.test.cutoff = 
    ifelse(Ad.pred.test[,2] > cutoff.best,'pos','neg') 

cm4 = confusionMatrix(table(pred=Ad.pred.test.cutoff,
                           actual = b.bal$y), positive='yes')
 # Type 1 error
alpha4 <- (1-cm4$byClass["Specificity"][[1]])

# Type 2 error
Beta4 <-  (1-cm4$byClass["Sensitivity"][[1]])

Exp_Profit4 <- 5200 -5200*(Beta4)-8700*(alpha4) 
Exp_Profit4                            
                           
# Model analysis
# The model built using b.test, and b.bal with default cutoff
# Accuracy: 0.7784
# Sensitivity: 0.75145
# Specificity: .78186

# The model built using b.test, and b.bal with cost maximizing cutoff
# Accuracy: 0.7611
# Sensitivity: 0.7861
# Specificity: .7878


# The model built using b.test, and b.bal with cost maximizing cutoff
# Accuracy: 0.7611
# Sensitivity: 0.7861
# Specificity: .7878

# The model built with cost minimizing cutoff has better accuracy, with the expected profits increasing.
                           
