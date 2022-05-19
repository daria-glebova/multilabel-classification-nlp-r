#Downloading the required packages
library(tidyverse)
library(readxl)
library(tm)
library(stopwords)
library(SnowballC)
library(text2vec)
library(dplyr)
#install.packages("mldr")
library(mldr)
#install.packages("mlr")
library(mlr)
#install.packages("utiml")
library(utiml)
#install.packages("xgboost")
library(xgboost)
#install.packages("randomForest")
library(randomForest)

################################################################################

# 1. DATABASE WITH COMMENTS

#Loading data
# good = What do you like about the organization
# bad = What needs to be improved
path <- "bigdata.xlsx"

good <- data.frame(read_excel(path, sheet = 1))
bad <- data.frame(read_excel(path, sheet = 2))

good$id <- as.character(good$id)
bad$id <- as.character(bad$id)

#Remove duplicate id
length(unique(good$id)) == nrow(good)
length(unique(bad$id)) == nrow(bad)
good <- good[!duplicated(good$id), ]
bad <- bad[!duplicated(bad$id), ]

#Data cleaning
good_clean <- good %>% 
  mutate(text = tolower(text),
         text = removePunctuation(text),
         text = removeNumbers(text),
         text = stripWhitespace(text))

bad_clean <- bad %>% 
  mutate(text = tolower(text),
         text = removePunctuation(text),
         text = removeNumbers(text),
         text = stripWhitespace(text))

#Stemming
good_stem <- good_clean %>% 
  mutate(text = stemDocument(text, language = "russian"))

bad_stem <- bad_clean %>% 
  mutate(text = stemDocument(text, language = "russian"))

#Tokenizers
good_token_prep <- itoken(good_stem$text,
                          ids = good_stem$id,
                          tokenizer = word_tokenizer)

bad_token_prep <- itoken(bad_stem$text,
                         ids = bad_stem$id,
                         tokenizer = word_tokenizer)

#Loading stop words
stop_words <- c(stopwords(language = "ru", source = "snowball"), "«", "»") %>% 
  stemDocument(., language = "russian")

#Dictionaries with mono- and bigrams
good_vocab <- prune_vocabulary(create_vocabulary(good_token_prep, 
                                                 stopwords = stop_words, 
                                                 ngram = c(1L,2L)),
                               doc_proportion_min = 0.005,
                               doc_proportion_max = 0.90)


bad_vocab <- prune_vocabulary(create_vocabulary(bad_token_prep, 
                                                stopwords = stop_words, 
                                                ngram = c(1L,2L)),
                              doc_proportion_min = 0.005,
                              doc_proportion_max = 0.90)

#Document-term matrix
good_term_document <- create_dtm(good_token_prep, 
                                 vocab_vectorizer(good_vocab)) %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "id")

bad_term_document <- create_dtm(bad_token_prep,
                                vocab_vectorizer(bad_vocab)) %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "id")

################################################################################

# 2. DATABASE WITH CATEGORIES

#Loading data
# g = What do you like about the organization
# b = What needs to be improved
p <- "categories.xlsx"

g <- data.frame(read_excel(p, sheet = 1))
b <- data.frame(read_excel(p, sheet = 2))

g$id <- as.character(g$id)
b$id <- as.character(b$id)

#Удаляю повторяющиеся id
length(unique(g$id)) == nrow(g)
length(unique(b$id)) == nrow(b)
g <- g[!duplicated(g$id), ]
b <- b[!duplicated(b$id), ]

#Меняю NA на 0
g[is.na(g)] <- 0
b[is.na(b)] <- 0

################################################################################

# 3. COMBINING DATABASES

#One big table - categories and term-document
good_employer <- merge(x = g, y = good_term_document)
development_zones <- merge(x = b, y = bad_term_document)

#Remove id so they don't become predictors in models
good_employer <- good_employer[,-1] 
development_zones <- development_zones[,-1] 

#Converting one large table to Multilabel Datasets (MLD) + deleting observations without categories
good_employer_mldr <- mldr_from_dataframe(good_employer, 
                                          labelIndices = c(1:13), 
                                          name = "testMLDR") %>% 
  remove_unlabeled_instances()
development_zones_mldr <- mldr_from_dataframe(development_zones, 
                                              labelIndices = c(1:12), 
                                              name = "testMLDR") %>% 
  remove_unlabeled_instances()

#Small 5% samples (for sample models)
small_good_employer <- good_employer %>% sample_frac(size = 0.05)
small_development_zones <- development_zones %>% sample_frac(size = 0.05)

#Convert small 5% samples to MLD
small_good_employer_mldr <- mldr_from_dataframe(small_good_employer, 
                                                labelIndices = c(1:13), 
                                                name = "testMLDR")
small_development_zones_mldr <- mldr_from_dataframe(small_development_zones, 
                                                    labelIndices = c(1:12), 
                                                    name = "testMLDR")

#Create training samples and validation samples
data_small <- create_holdout_partition(small_good_employer_mldr, 
                                       c(train=0.7, test=0.3))
data <- create_holdout_partition(good_employer_mldr, 
                                 c(train=0.7, test=0.3))

data_b_small <- create_holdout_partition(small_development_zones_mldr, 
                                         c(train=0.7, test=0.3))
data_b <- create_holdout_partition(development_zones_mldr, 
                                   c(train=0.7, test=0.3))

################################################################################

# 4. DESCRIPTIVE STATISTICS AND GRAPHS

#Category names = numbers, for nicer charts
goemp <- good_employer

goemp$`1` <- good_employer$`Заработная.плата`
goemp$`2` <- good_employer$`Нематериальная.мотивация`
goemp$`3` <- good_employer$`Соблюдение.обязательств.и.Трудового.Кодекса.работодателем`
goemp$`4` <- good_employer$`График.и.условия.труда`
goemp$`5` <- good_employer$`Обучение..профессиональное.развитие`
goemp$`6` <- good_employer$`Руководство`
goemp$`7` <- good_employer$`Атмосфера.и.коллеги`
goemp$`8` <- good_employer$`Внимательное.отношение.к.сотрудникам`
goemp$`9` <- good_employer$`Бизнес.процессы.и.организация.работы`
goemp$`10` <- good_employer$`Карьерные.возможности`
goemp$`11` <- good_employer$`Интересная.работа`
goemp$`12` <- good_employer$`Положение.компании.на.рынке.труда..репутация.и.бренд` 
goemp$`13` <- good_employer$`Профессиональная.среда`

goemp <- goemp[,-c(1:13)]

devzo <- development_zones

devzo$`1` <- development_zones$`Заработная.плата`
devzo$`2` <- development_zones$`Нематериальная.мотивация`
devzo$`3` <- development_zones$`График.и.условия.труда`
devzo$`4` <- development_zones$`Обучение..профессиональное.развитие`
devzo$`5` <- development_zones$`Руководство`
devzo$`6` <- development_zones$`Атмосфера.и.коллеги`
devzo$`7` <- development_zones$`Взаимодействие.между.подразделениями`
devzo$`8` <- development_zones$`Бизнес.процессы.и.организация.работы`
devzo$`9` <- development_zones$`Информирование..наличие.обратной.связи`
devzo$`10` <- development_zones$`Карьерные.возможности`
devzo$`11` <- development_zones$`Кадры.и.HR.практики`
devzo$`12` <- development_zones$`Оборудование..техническая.оснащенность`

devzo <- devzo[,-c(1:12)]

gemp_mldr <- mldr_from_dataframe(goemp, 
                                 labelIndices = c(190:201), 
                                 name = "testMLDR") %>% 
  remove_unlabeled_instances()

devzo_mldr <- mldr_from_dataframe(devzo, 
                                  labelIndices = c(187:198), 
                                  name = "testMLDR") %>% 
  remove_unlabeled_instances()

#Descriptive statistics for MLD
summary(gemp_mldr)
summary(devzo_mldr)

#Separately descriptive category statistics in MLD
gemp_mldr$labels
devzo_mldr$labels

#Plots
plot(gemp_mldr, type = "LC")
plot(gemp_mldr, type = "LB")

plot(devzo_mldr, type = "LC")
plot(devzo_mldr, type = "LB")

################################################################################

# 5. BUILD MODELS

#Random forest

#1
rf_model <- br(data$train, "RF", importance = TRUE)
rf_pred <- predict(rf_model, data$test)
rf_res <- multilabel_evaluate(data$test, rf_pred)
round(rf_res, 4)

rf_model_b <- br(data_b$train, "RF", importance = TRUE)
rf_pred_b <- predict(rf_model_b, data_b$test)
rf_res_b <- multilabel_evaluate(data_b$test, rf_pred_b)
round(rf_res_b, 4)

#2
rf_model_2 <- br(data$train, "RF", ntree = 2000, importance = TRUE)
rf_pred_2 <- predict(rf_model_2, data$test)
rf_res_2 <- multilabel_evaluate(data$test, rf_pred_2)
round(rf_res_2, 4)

rf_model_b_2 <- br(data_b$train, "RF", ntree = 2000, importance = TRUE)
rf_pred_b_2 <- predict(rf_model_b_2, data_b$test)
rf_res_b_2 <- multilabel_evaluate(data_b$test, rf_pred_b_2)
round(rf_res_b_2, 4)

#3
rf_model_3 <- br(data$train, "RF", ntree = 2500, importance = TRUE)
rf_pred_3 <- predict(rf_model_3, data$test)
rf_res_3 <- multilabel_evaluate(data$test, rf_pred_3)
round(rf_res_3, 4)

rf_model_b_3 <- br(data_b$train, "RF", ntree = 1500, importance = TRUE)
rf_pred_b_3 <- predict(rf_model_b_3, data_b$test)
rf_res_b_3 <- multilabel_evaluate(data_b$test, rf_pred_b_3)
round(rf_res_b_3, 4)

#4
rf_model_4 <- br(data$train, "RF", ntree = 2750, importance = TRUE)
rf_pred_4 <- predict(rf_model_4, data$test)
rf_res_4 <- multilabel_evaluate(data$test, rf_pred_4)
round(rf_res_4, 4)

################################################################################

# XGBOOST

#1
xgb_model <- br(data$train, "XGB")
xgb_pred <- predict(xgb_model, data$test)
xgb_res <- multilabel_evaluate(data$test, xgb_pred)
round(xgb_res, 4)

xgb_model_b <- br(data_b$train, "XGB")
xgb_pred_b <- predict(xgb_model_b, data_b$test)
xgb_res_b <- multilabel_evaluate(data_b$test, xgb_pred_b)
round(xgb_res_b, 4)

#2
xgb_model_2 <- br(data$train, "XGB", 
                  max.depth = 8, 
                  eta = 0.03, 
                  objective = "binary:logistic", 
                  nrounds = 1000)
xgb_pred_2 <- predict(xgb_model_2, data$test)
xgb_res_2 <- multilabel_evaluate(data$test, xgb_pred_2)
round(xgb_res_2, 4)

xgb_model_b_2 <- br(data_b$train, "XGB", 
                    max.depth = 8, 
                    eta = 0.03, 
                    objective = "binary:logistic", 
                    nrounds = 1000)
xgb_pred_b_2 <- predict(xgb_model_b_2, data_b$test)
xgb_res_b_2 <- multilabel_evaluate(data_b$test, xgb_pred_b_2)
round(xgb_res_b_2, 4)

#3
xgb_model_3 <- br(data$train, "XGB", 
                  max.depth = 9, 
                  eta = 0.01, 
                  objective = "binary:logistic", 
                  nrounds = 1000)
xgb_pred_3 <- predict(xgb_model_3, data$test)
xgb_res_b_3 <- multilabel_evaluate(data$test, xgb_pred_3)
round(xgb_res_3, 4)

xgb_model_b_3 <- br(data_b$train, "XGB", 
                    max.depth = 8, 
                    eta = 0.01, 
                    objective = "binary:logistic", 
                    nrounds = 1500)
xgb_pred_b_3 <- predict(xgb_model_b_3, data_b$test)
xgb_res_b_3 <- multilabel_evaluate(data_b$test, xgb_pred_b_3)
round(xgb_res_b_3, 4)

#4
xgb_model_b_4 <- br(data_b$train, "XGB", 
                    max.depth = 8, 
                    eta = 0.02, 
                    objective = "binary:logistic", 
                    nrounds = 1000)
xgb_pred_b_4 <- predict(xgb_model_b_4, data_b$test)
xgb_res_b_4 <- multilabel_evaluate(data_b$test, xgb_pred_b_4)
round(xgb_res_b_4, 4)

################################################################################

#Naive Bayes

nb_model <- br(data$train, "NB")
nb_pred <- predict(nb_model, data$test)
nb_res <- multilabel_evaluate(data$test, nb_pred)
round(nb_res, 4)

nb_model_b <- br(data_b$train, "NB")
nb_pred_b <- predict(nb_model_b, data_b$test)
nb_res_b <- multilabel_evaluate(data_b$test, nb_pred_b)
round(nb_res_b, 4)

################################################################################

#Accuracy RF models separately for each category
cm_g_1_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred)))
prec_g_1_rf <- (cm_g_1_rf$TP + cm_g_1_rf$TN)  / (cm_g_1_rf$TP + cm_g_1_rf$FP + cm_g_1_rf$TN + cm_g_1_rf$FN)
round(prec_g_1_rf, 4)

cm_b_1_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, rf_pred_b)))
prec_b_1_rf <- (cm_b_1_rf$TP + cm_b_1_rf$TN)  / (cm_b_1_rf$TP + cm_b_1_rf$FP + cm_b_1_rf$TN + cm_b_1_rf$FN)
round(prec_b_1_rf, 4)

cm_g_2_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred_2)))
prec_g_2_rf <- (cm_g_2_rf$TP + cm_g_2_rf$TN)  / (cm_g_2_rf$TP + cm_g_2_rf$FP + cm_g_2_rf$TN + cm_g_2_rf$FN)
round(prec_g_2_rf, 4)

cm_b_2_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, rf_pred_b_2)))
prec_b_2_rf <- (cm_b_2_rf$TP + cm_b_2_rf$TN)  / (cm_b_2_rf$TP + cm_b_2_rf$FP + cm_b_2_rf$TN + cm_b_2_rf$FN)
round(prec_b_2_rf, 4)

cm_g_3_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred_3)))
prec_g_3_rf <- (cm_g_3_rf$TP + cm_g_3_rf$TN)  / (cm_g_3_rf$TP + cm_g_3_rf$FP + cm_g_3_rf$TN + cm_g_3_rf$FN)
round(prec_g_3_rf, 4)

cm_b_3_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, rf_pred_b_3)))
prec_b_3_rf <- (cm_b_3_rf$TP + cm_b_3_rf$TN)  / (cm_b_3_rf$TP + cm_b_3_rf$FP + cm_b_3_rf$TN + cm_b_3_rf$FN)
round(prec_b_3_rf, 4)

cm_g_4_rf <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred_4)))
prec_g_4_rf <- (cm_g_4_rf$TP + cm_g_4_rf$TN)  / (cm_g_4_rf$TP + cm_g_4_rf$FP + cm_g_4_rf$TN + cm_g_4_rf$FN)
round(prec_g_4_rf, 4)

#Accuracy XGB models separately for each category
cm_g_1_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, xgb_pred)))
prec_g_1_xgb <- (cm_g_1_xgb$TP + cm_g_1_xgb$TN) / (cm_g_1_xgb$TP + cm_g_1_xgb$FP + cm_g_1_xgb$TN + cm_g_1_xgb$FN)
round(prec_g_1_xgb, 4)

cm_b_1_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b)))
prec_b_1_xgb <- (cm_b_1_xgb$TP + cm_b_1_xgb$TN) / (cm_b_1_xgb$TP + cm_b_1_xgb$FP + cm_b_1_xgb$TN + cm_b_1_xgb$FN)
round(prec_b_1_xgb, 4)

cm_g_2_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, xgb_pred_2)))
prec_g_2_xgb <- (cm_g_2_xgb$TP + cm_g_2_xgb$TN) / (cm_g_2_xgb$TP + cm_g_2_xgb$FP + cm_g_2_xgb$TN + cm_g_2_xgb$FN)
round(prec_g_2_xgb, 4)

cm_b_2_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b_2)))
prec_b_2_xgb <- (cm_b_2_xgb$TP + cm_b_2_xgb$TN) / (cm_b_2_xgb$TP + cm_b_2_xgb$FP + cm_b_2_xgb$TN + cm_b_2_xgb$FN)
round(prec_b_2_xgb, 4)

cm_g_3_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, xgb_pred_3)))
prec_g_3_xgb <- (cm_g_3_xgb$TP + cm_g_3_xgb$TN) / (cm_g_3_xgb$TP + cm_g_3_xgb$FP + cm_g_3_xgb$TN + cm_g_3_xgb$FN)
round(prec_g_3_xgb, 4)

cm_b_3_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b_3)))
prec_b_3_xgb <- (cm_b_3_xgb$TP + cm_b_3_xgb$TN) / (cm_b_3_xgb$TP + cm_b_3_xgb$FP + cm_b_3_xgb$TN + cm_b_3_xgb$FN)
round(prec_b_3_xgb, 4)

cm_b_4_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b_4)))
prec_b_4_xgb <- (cm_b_4_xgb$TP + cm_b_4_xgb$TN) / (cm_b_4_xgb$TP + cm_b_4_xgb$FP + cm_b_4_xgb$TN + cm_b_4_xgb$FN)
round(prec_b_4_xgb, 4)

#Accuracy NB models separately for each category
cm_g_nb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, nb_pred)))
prec_g_nb <- (cm_g_nb$TP + cm_g_nb$TN) / (cm_g_nb$TP + cm_g_nb$FP + cm_g_nb$TN + cm_g_nb$FN)
round(prec_g_nb, 4)

cm_b_nb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, nb_pred_b)))
prec_b_nb <- (cm_b_nb$TP + cm_b_nb$TN) / (cm_b_nb$TP + cm_b_nb$FP + cm_b_nb$TN + cm_b_nb$FN)
round(prec_b_nb, 4)

################################################################################

# Assessing the importance of predictors (substitute: 1.desired model name and 2.category name)
imp <- as.data.frame(rf_model_b$models$График.и.условия.труда$importanceSD)
imp <- cbind(vars = rownames(imp), imp)
imp <-  imp[order(imp$MeanDecreaseAccuracy), ]
imp$vars <- factor(imp$vars, levels = unique(imp$vars))

#The feature importance (Mean Decrease Accuracy) plot
dotchart(imp$MeanDecreaseAccuracy, 
         imp$vars, xlim = c(0,max(imp$MeanDecreaseAccuracy)), pch = 16)

################################################################################