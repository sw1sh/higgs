library(doMC);library(caret);
registerDoMC(cores = 2)

training.y = read.csv('training.csv')
testing = read.csv('test.csv')

training[training==-999] <- NA
testing[testing==-999] <- NA

# Remove Event ID and Weight from training data
training$EventId = NULL
training$Weight = NULL
testing$EventId = NULL

# Imputate with Amelia
library(Amelia)

train_imputed = amelia(training, idvars = c("Label"), m=5, p2s=2)
write.amelia(obj=train_imputed, file.stem = "train_imputed")

test_imputed = amelia(testing, m=5, p2s=2)
write.amelia(obj=test_imputed, file.stem = "test_imputed")

training = read.csv('train_imputed5.csv')[,-1]
testing = read.csv('test_imputed5.csv')[,-1]


# Remove problematic predictors

highCorr = findCorrelation(cor(training[,-31]),0.9)
training = training[,-highCorr]

# Feature Detection

mastProfile = NULL

for (i in 1:50) {

  randSample = training[sample(nrow(training), 1000), ]

  ctrl <- rfeControl( functions = rfFuncs,
                      method = "cv",
                      number = 2,
                      verbose = FALSE)

  y = randSample$Label
  x = randSample
  x$Label = NULL

  Profile <- rfe( x, y,
                  sizes = c(10),
                  rfeControl = ctrl)

  mastProfile <- rbind(mastProfile, predictors(Profile))

}

# Take 12 most featured predictors

predictVars = factor(mastProfile)
predLevels = levels(predictVars)
o = order(table(predictVars), decreasing = TRUE)
predictVars = factor(predictVars, levels = predLevels[o])
predictVars = levels(predictVars)[1:12]

# Fitting
grid <-  expand.grid(interaction.depth = c(2,3),
                     n.trees = c(150),
                     shrinkage = 0.1)

fitControl = trainControl( method = "cv", 
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE, # makes ROC metric to work
                           savePredictions=TRUE,
                           number = 5 ) # 5-fold cross-validation

modelFit = train( Label ~ DER_met_phi_centrality+PRI_met+PRI_tau_pt+DER_mass_transverse_met_lep+DER_mass_vis+
                          DER_pt_ratio_lep_tau+DER_deltar_tau_lep+PRI_met_sumet+PRI_jet_leading_pt+
                          PRI_jet_num+DER_lep_eta_centrality+DER_mass_jet_jet,
                  data = training,
                  method = "gbm",
                  preProcess = c("YeoJohnson","scale"),
                  trControl = fitControl, 
                  tuneGrid = grid,
                  metric = "ROC",
                  verbose = TRUE)

trainPrediction = predict(modelFit, newdata = training)
confusionMatrix(trainPrediction, training$Label)

# Calculate AMS

AMS = function(pred,real,weight)
{
  s = sum(weight[pred == "s" & real == "s"])      # True positive rate
  b = sum(weight[pred == "s" & real == "b"])      # False positive rate
  
  b_tau = 10                                      # Regulator weight
  ans = sqrt(2*((s+b+b_tau)*log(1+s/(b+b_tau))-s))
  return(ans)
}

AMS(pred = as.character(trainPrediction), real = as.character(training$Label), weight = training.y$Weight)

# Make predictions on test set and create submission file

testPrediction = predict(modelFit, newdata = testing)

weightRank = rank(testPrediction, ties.method = "random")


submission = data.frame(EventId = testing$EventId, RankOrder = weightRank, Class = testPrediction)
write.csv(submission, "submission.csv", row.names=FALSE)

