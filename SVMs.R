# Helper packages
library(tidyverse)  # fordata wrangling
library(ggplot2)    # for awesome graphics
library(rsample)    # for data splitting

# Modelling packages
library(lattice)
library(caret)      # for classification and regression training
library(purrr)
library(kernlab)    # for fitting SVMs

# Model interpretability packages
library("pdp")      # for partial dependence plots, etc.
library("vip")      # for variable importance plots

# Job attrition data
setwd("C:/Users/hp/Desktop/Machine Learning/homlr-master/homlr-master/data")
attrition <- read.csv("attrition.csv", header = TRUE)
head(attrition)

# Load attrition data
df <- attrition %>%
  mutate_if(is.ordered, factor, ordered = FALSE)

# Create train set (70%) and test set (30%)
set.seed(123)  # for reproducibility
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition")
churn_train <- training(churn_split)
churn_test <- testing(churn_split)

# Linear (i.e., soft margin classifier)
caret::getModelInfo("svmLinear")$svmLinear$parameters
# Polynomial kernel
caret::getModelInfo("svmPoly")$svmPoly$parameters
#Radial basis kernel
caret::getModelInfo("svmRadial")$svmRadial$parameters
library(mlbench)
?mlbench::mlbench.spirals

# Job Attrition Example 
# Tune an SVM with radial basis kernel
set.seed(1854) # for reproducibility
churn_svm <- train(Attrition ~.,
                   data = churn_train,
                   method = "svmRadial",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "cv", number = 10),
                   tuneLength = 10
                   )
# Plot results
ggplot(churn_svm) + theme_light()
# Print results
churn_svm$results

# Class weights
class.weights = c("No" = 1, "Yes" = 10)

# Class Probabilities
ctrl <- trainControl(method = "cv",
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary  # also neededfor AUC/ROC
                     )

# Tune an SVM
set.seed(5628)   # for reproducibility
churn_svm_auc <- train(Attrition ~.,
                       data = churn_train,
                       method = "svmRadial",
                       preProcess = c("center", "scale"),
                       metric = "ROC",   # area under ROC curve (AUC),
                       trControl = ctrl,
                       tuneLength = 10
                       )
# Print results
churn_svm_auc$results

confusionMatrix(churn_svm_auc)

# Feature Interpretation
prob_yes <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Yes"]
}

# Variable importance plot
set.seed(2827)  # for reproducibility
vip(churn_svm_auc, 
    method = "permute",
    nsim = 5, 
    train = churn_train,
    target = "Attrition",
    metric = "auc",
    reference_class = "Yes",
    pred_wrapper = prob_yes,
    geom = "col",
    all_permutations = TRUE,
    mapping = aes_string(fill = "Variable"),
    aesthetics = list(color = "grey35", size = 0.8)
    )
?vip

?pdp::partial
features <- c("OverTime", "WorkLifeBalance",
              "JobSatisfaction", "JobRole")
pdps <- lapply(features, function(x) {
  partial(churn_svm_auc, pred.var = x, which.class = 2,
          prob = TRUE, plot = TRUE, chull = TRUE, progress = TRUE, plot.engine = "ggplot2") +
    coord_flip()
})
plotPartial(pdps, levelplot = FALSE, zlab = "cmedv", drape = TRUE,
            colorkey = FALSE)
grid.arrange(grobs = pdps, ncol=2)
