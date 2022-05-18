#Загрузка необходимых пакетов
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

getwd()
setwd("C:/Users/marus/Desktop/rrrrrr/")

################################################################################

# 1. БАЗА С КОММЕНТАРИЯМИ

#Загрузка данных
#    good = Что Вам нравится в организации
#    bad = Что необходимо улучшить
path <- "bigdata.xlsx"

good <- data.frame(read_excel(path, sheet = 1))
bad <- data.frame(read_excel(path, sheet = 2))

good$id <- as.character(good$id)
bad$id <- as.character(bad$id)

#Удаляю повторяющиеся id
length(unique(good$id)) == nrow(good)
length(unique(bad$id)) == nrow(bad)
good <- good[!duplicated(good$id), ]
bad <- bad[!duplicated(bad$id), ]

#Чистка данных от лишнего
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

#Стеммирование
good_stem <- good_clean %>% 
  mutate(text = stemDocument(text, language = "russian"))

bad_stem <- bad_clean %>% 
  mutate(text = stemDocument(text, language = "russian"))

#Токенайзеры
good_token_prep <- itoken(good_stem$text,
                     ids = good_stem$id,
                     tokenizer = word_tokenizer)

bad_token_prep <- itoken(bad_stem$text,
                          ids = bad_stem$id,
                          tokenizer = word_tokenizer)

#Загружаю стоп-стова
stop_words <- c(stopwords(language = "ru", source = "snowball"), "«", "»") %>% 
  stemDocument(., language = "russian")

#Словари с моно- и биграммами
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

#Матрица term-document
good_term_document <- create_dtm(good_token_prep, vocab_vectorizer(good_vocab)) %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "id")

bad_term_document <- create_dtm(bad_token_prep, vocab_vectorizer(bad_vocab)) %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "id")

################################################################################

# 2. БАЗА С КАТЕГОРИЯМИ

#Загрузка данных
#    g = Что Вам нравится в орагнизации
#    b = Что необходимо улучшить
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

# 3. ОБЪЕДИНЕНИЕ БАЗ ДАННЫХ

#Одна большая таблица - категории и term-document
good_employer <- merge(x = g, y = good_term_document)
development_zones <- merge(x = b, y = bad_term_document)

#Удаляю id, чтобы они не стали предикторами в моделях
good_employer <- good_employer[,-1] 
development_zones <- development_zones[,-1] 

#Конвертирование одной большой таблицы в Multilabel Datasets (MLD) + удаление наблюдений без категорий
good_employer_mldr <- mldr_from_dataframe(good_employer, labelIndices = c(1:13), name = "testMLDR") %>% 
  remove_unlabeled_instances()
development_zones_mldr <- mldr_from_dataframe(development_zones, labelIndices = c(1:12), name = "testMLDR") %>% 
  remove_unlabeled_instances()

#Маленькие 5%-выборки (для проб моделей)
small_good_employer <- good_employer %>% sample_frac(size = 0.05)
small_development_zones <- development_zones %>% sample_frac(size = 0.05)

#Конвертирование маленьких 5%-выборок в MLD
small_good_employer_mldr <- mldr_from_dataframe(small_good_employer, labelIndices = c(1:13), name = "testMLDR")
small_development_zones_mldr <- mldr_from_dataframe(small_development_zones, labelIndices = c(1:12), name = "testMLDR")

#Создание выборок обучения и выборок проверки
data_small <- create_holdout_partition(small_good_employer_mldr, c(train=0.7, test=0.3))
data <- create_holdout_partition(good_employer_mldr, c(train=0.7, test=0.3))

data_b_small <- create_holdout_partition(small_development_zones_mldr, c(train=0.7, test=0.3))
data_b <- create_holdout_partition(development_zones_mldr, c(train=0.7, test=0.3))

################################################################################

#4. ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ И ГРАФИКИ

#Названия категорий = цифры, для более приятных графиков
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

gemp_mldr <- mldr_from_dataframe(goemp, labelIndices = c(190:201), name = "testMLDR") %>% 
  remove_unlabeled_instances()
devzo_mldr <- mldr_from_dataframe(devzo, labelIndices = c(187:198), name = "testMLDR") %>% 
  remove_unlabeled_instances()

#Описательные статистики для MLD
summary(gemp_mldr)
summary(devzo_mldr)

#Отдельно описательные статистики категорий в MLD
gemp_mldr$labels
devzo_mldr$labels

#Графики
plot(gemp_mldr, type = "LC")
plot(gemp_mldr, type = "LB")

plot(devzo_mldr, type = "LC")
plot(devzo_mldr, type = "LB")

################################################################################

# 5. ПОСТРОЕНИЕ МОДЕЛЕЙ 

#Модель Random forest

#1
rf_model <- br(data$train, "RF", importance = TRUE)
rf_pred <- predict(rf_model, data$test)
rf_res <- multilabel_evaluate(data$test, rf_pred)
round(rf_res, 4)
#accuracy average-precision               clp          coverage 
#0.7991            0.8988            0.0000            1.8469 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8327            0.0433            0.9276            0.7479 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8416            0.6922            1.0431            0.9524 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8437            0.8872            0.8042            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1092            0.8708            0.0595            0.8217 
#subset-accuracy               wlp 
#0.6767            0.0000 

rf_model_b <- br(data_b$train, "RF", importance = TRUE)
rf_pred_b <- predict(rf_model_b, data_b$test)
rf_res_b <- multilabel_evaluate(data_b$test, rf_pred_b)
round(rf_res_b, 4)
#accuracy average-precision               clp          coverage 
#0.6991            0.8772            0.0000            1.7829 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7523            0.0657            0.8742            0.6292 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7536            0.5656            1.1062            0.9325 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7762            0.7893            0.7636            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1469            0.7748            0.0755            0.7850 
#subset-accuracy               wlp 
#0.5335            0.0000 

#2
rf_model_2 <- br(data$train, "RF", ntree = 2000, importance = TRUE)
rf_pred_2 <- predict(rf_model_2, data$test)
rf_res_2 <- multilabel_evaluate(data$test, rf_pred_2)
round(rf_res_2, 4)
#accuracy average-precision               clp          coverage 
#0.8007            0.8995            0.0000            1.8274 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8338            0.0430            0.9281            0.7485 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8449            0.6919            1.0260            0.9527 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8448            0.8893            0.8045            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1095            0.8724            0.0586            0.8220 
#subset-accuracy               wlp 
#0.6799            0.0000 

getMultilabelBinaryPerformances(cm, measures = list(acc, ppv, tpr, fpr, fnr, tnr))


rf_model_b_2 <- br(data_b$train, "RF", ntree = 2000, importance = TRUE)
rf_pred_b_2 <- predict(rf_model_b_2, data_b$test)
rf_res_b_2 <- multilabel_evaluate(data_b$test, rf_pred_b_2)
round(rf_res_b_2, 4)
#accuracy average-precision               clp          coverage 
#0.6949            0.8759            0.0000            1.7801 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7481            0.0664            0.8773            0.6281 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7558            0.5637            1.1016            0.9352 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7736            0.7871            0.7605            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1501            0.7710            0.0764            0.7806 
#subset-accuracy               wlp 
#0.5289            0.0000

#3
rf_model_3 <- br(data$train, "RF", ntree = 2500, importance = TRUE)
rf_pred_3 <- predict(rf_model_3, data$test)
rf_res_3 <- multilabel_evaluate(data$test, rf_pred_3)
round(rf_res_3, 4)
#accuracy average-precision               clp          coverage 
#0.8020            0.8993            0.0000            1.7821 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8351            0.0424            0.9266            0.7508 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8263            0.7002            1.0126            0.9521 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8441            0.8844            0.8073            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1124            0.8705            0.0582            0.8256 
#subset-accuracy               wlp 
#0.6820            0.0000

rf_model_b_3 <- br(data_b$train, "RF", ntree = 1500, importance = TRUE)
rf_pred_b_3 <- predict(rf_model_b_3, data_b$test)
rf_res_b_3 <- multilabel_evaluate(data_b$test, rf_pred_b_3)
round(rf_res_b_3, 4)
#accuracy average-precision               clp          coverage 
#0.7058            0.8894            0.0000            1.6933 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7595            0.0650            0.8872            0.6274 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7618            0.5558            0.9917            0.9437 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7806            0.7963            0.7655            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1335            0.7823            0.0659            0.7913 
#subset-accuracy               wlp 
#0.5423            0.0000 

#4
rf_model_4 <- br(data$train, "RF", ntree = 2750, importance = TRUE)
rf_pred_4 <- predict(rf_model_4, data$test)
rf_res_4 <- multilabel_evaluate(data$test, rf_pred_4)
round(rf_res_4, 4)
#accuracy average-precision               clp          coverage 
#0.8014            0.8975            0.0000            1.7885 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8342            0.0425            0.9262            0.7505 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8321            0.6998            1.0182            0.9520 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8439            0.8840            0.8073            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1138            0.8696            0.0590            0.8247 
#subset-accuracy               wlp 
#0.6820            0.0000
################################################################################

#Модель XGBOOST

xgb_model <- br(data$train, "XGB")
xgb_pred <- predict(xgb_model, data$test)
xgb_res <- multilabel_evaluate(data$test, xgb_pred)
round(xgb_res, 4)
#accuracy average-precision               clp          coverage 
#0.7861            0.8909            0.0000            1.8180 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8214            0.0461            0.8989            0.7409 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8237            0.6923            1.0260            0.9462 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8338            0.8761            0.7954            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1186            0.8617            0.0567            0.8096 
#subset-accuracy               wlp 
#0.6614            0.0000

xgb_model_b <- br(data_b$train, "XGB")
xgb_pred_b <- predict(xgb_model_b, data_b$test)
xgb_res_b <- multilabel_evaluate(data_b$test, xgb_pred_b)
round(xgb_res_b, 4)
#accuracy average-precision               clp          coverage 
#0.6503            0.8567            0.0000            1.7954 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7079            0.0744            0.8256            0.6165 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7369            0.5540            1.1275            0.9305 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7413            0.7704            0.7143            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1801            0.7475            0.0780            0.7280 
#subset-accuracy               wlp 
#0.4730            0.0000 

#2
xgb_model_2 <- br(data$train, "XGB", max.depth = 8, eta = 0.03, objective = "binary:logistic", nrounds = 1000)
xgb_pred_2 <- predict(xgb_model_2, data$test)
xgb_res_2 <- multilabel_evaluate(data$test, xgb_pred_2)
round(xgb_res_2, 4)
#accuracy average-precision               clp          coverage 
#0.7985            0.9036            0.0000            1.7034 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8333            0.0444            0.9355            0.7505 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8203            0.7004            0.9063            0.9591 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8404            0.8805            0.8038            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1076            0.8700            0.0487            0.8235 
#subset-accuracy               wlp 
#0.6748            0.0000 

xgb_model_b_2 <- br(data_b$train, "XGB", max.depth = 8, eta = 0.03, objective = "binary:logistic", nrounds = 1000)
xgb_pred_b_2 <- predict(xgb_model_b_2, data_b$test)
xgb_res_b_2 <- multilabel_evaluate(data_b$test, xgb_pred_b_2)
round(xgb_res_b_2, 4)
#accuracy average-precision               clp          coverage 
#0.7019            0.8864            0.0000            1.5875 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7558            0.0657            0.9095            0.6419 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7347            0.5850            0.9062            0.9551 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7751            0.7919            0.7590            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1418            0.7819            0.0615            0.7850 
#subset-accuracy               wlp 
#0.5353            0.0000 

#3
xgb_model_b_3 <- br(data_b$train, "XGB", max.depth = 8, eta = 0.01, objective = "binary:logistic", nrounds = 1000)
xgb_pred_b_3 <- predict(xgb_model_b_3, data_b$test)
xgb_res_b_3 <- multilabel_evaluate(data_b$test, xgb_pred_b_3)
round(xgb_res_b_3, 4)
#accuracy average-precision               clp          coverage 
#0.6999            0.8885            0.0000            1.5912 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7542            0.0648            0.9220            0.6293 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7581            0.5545            0.8804            0.9605 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7775            0.8070            0.7502            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1363            0.7924            0.0592            0.7740 
#subset-accuracy               wlp 
#0.5358            0.0000

xgb_model_b_3 <- br(data_b$train, "XGB", max.depth = 9, eta = 0.01, objective = "binary:logistic", nrounds = 1500)
xgb_pred_b_3 <- predict(xgb_model_b_3, data_b$test)
xgb_res_b_3 <- multilabel_evaluate(data_b$test, xgb_pred_b_3)
round(xgb_res_b_3, 4)
#accuracy average-precision               clp          coverage 
#0.7011            0.8918            0.0000            1.5723 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7569            0.0647            0.9201            0.6334 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7525            0.5617            0.8619            0.9602 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7782            0.8072            0.7512            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1335            0.7947            0.0580            0.7773 
#subset-accuracy               wlp 
#0.5321            0.0000 

xgb_model_b_4 <- br(data_b$train, "XGB", max.depth = 8, eta = 0.02, objective = "binary:logistic", nrounds = 1000)
xgb_pred_b_4 <- predict(xgb_model_b_4, data_b$test)
xgb_res_b_4 <- multilabel_evaluate(data_b$test, xgb_pred_b_4)
round(xgb_res_b_4, 4)
#accuracy average-precision               clp          coverage 
#0.7023            0.8917            0.0000            1.5714 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.7574            0.0648            0.9182            0.6325 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.7474            0.5624            0.8624            0.9593 
#micro-F1   micro-precision      micro-recall               mlp 
#0.7777            0.8065            0.7510            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1344            0.7947            0.0580            0.7776 
#subset-accuracy               wlp 
#0.5349            0.0000 

xgb_model_3 <- br(data$train, "XGB", max.depth = 8, eta = 0.01, objective = "binary:logistic", nrounds = 1500)
xgb_pred_3 <- predict(xgb_model_3, data$test)
xgb_res_3 <- multilabel_evaluate(data$test, xgb_pred_3)
round(xgb_res_3, 4)
#accuracy average-precision               clp          coverage 
#0.8026            0.9058            0.0000            1.6154 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.8364            0.0428            0.9391            0.7522 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.8164            0.7061            0.8506            0.9633 
#micro-F1   micro-precision      micro-recall               mlp 
#0.8430            0.8814            0.8079            0.0000 
#one-error         precision      ranking-loss            recall 
#0.1089            0.8700            0.0461            0.8277 
#subset-accuracy               wlp 
#0.6777            0.0000

################################################################################

#Модель Naive Bayes

nb_model <- br(data$train, "NB")
nb_pred <- predict(nb_model, data$test)
nb_res <- multilabel_evaluate(data$test, nb_pred)
round(nb_res, 4)
#accuracy average-precision               clp          coverage 
#0.3879            0.7004            0.0000            2.9772 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.5121            0.2356            0.8394            0.4792 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.3880            0.8183            2.3654            0.8229 
#micro-F1   micro-precision      micro-recall               mlp 
#0.5013            0.3599            0.8257            0.0000 
#one-error         precision      ranking-loss            recall 
#0.4069            0.4041            0.1389            0.8215 
#subset-accuracy               wlp 
#0.0476            0.0000

nb_model_b <- br(data_b$train, "NB")
nb_pred_b <- predict(nb_model_b, data_b$test)
nb_res_b <- multilabel_evaluate(data_b$test, nb_pred_b)
round(nb_res_b, 4)
#accuracy average-precision               clp          coverage 
#0.2359            0.6162            0.0000            4.2079 
#F1      hamming-loss         macro-AUC          macro-F1 
#0.3627            0.4332            0.7751            0.3406 
#macro-precision      macro-recall       margin-loss         micro-AUC 
#0.2674            0.8574            3.7206            0.6992 
#micro-F1   micro-precision      micro-recall               mlp 
#0.3632            0.2323            0.8324            0.0000 
#one-error         precision      ranking-loss            recall 
#0.3524            0.2420            0.2449            0.8418 
#subset-accuracy               wlp 
#0.0005            0.0000 

################################################################################

# Оценка важности предикторов (подставить нужное имя модели RF и имя категории)
imp <- as.data.frame(rf_model_2$models$Заработная.плата$importanceSD)
imp <- cbind(vars = rownames(imp), imp)
imp <-  imp[order(imp$MeanDecreaseAccuracy), ]
imp$vars <- factor(imp$vars, levels = unique(imp$vars))

#График важности предикторов
dotchart(imp$MeanDecreaseAccuracy, imp$vars, xlim = c(0,max(imp$MeanDecreaseAccuracy)), pch = 16)
#2368 x 2080

################################################################################

#Точность моделей RF отдельно для каждой категории
cm_g_1 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred)))
cm_b_1 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, rf_pred_b)))

prec_g_1 <- cm_g_1$TP / (cm_g_1$TP + cm_g_1$FP)
round(prec_g_1, 4)
prec_b_1 <- cm_b_1$TP / (cm_b_1$TP + cm_b_1$FP)
round(prec_b_1, 4)

cm_g_2 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred_2)))
#cm_b_2 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, rf_pred_b_2)))

prec_g_2 <- cm_g_2$TP / (cm_g_2$TP + cm_g_2$FP)
round(prec_g_2, 4)
#prec_b_2 <- cm_b_2$TP / (cm_b_2$TP + cm_b_2$FP)
#round(prec_b_2, 4)

cm_g_3 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred_3)))
cm_b_3 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, rf_pred_b_3)))

prec_g_3 <- cm_g_3$TP / (cm_g_3$TP + cm_g_3$FP)
round(prec_g_3, 4)
prec_b_3 <- cm_b_3$TP / (cm_b_3$TP + cm_b_3$FP)
round(prec_b_3, 4)

#cm_g_4 <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, rf_pred_4)))
#
#prec_g_4 <- cm_g_4$TP / (cm_g_4$TP + cm_g_4$FP)
#round(prec_g_4, 4)

#Точность моделей XGB отдельно для каждой категории
cm_g_1_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, xgb_pred)))
cm_b_1_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b)))

prec_g_1_xgb <- cm_g_1_xgb$TP / (cm_g_1_xgb$TP + cm_g_1_xgb$FP)
round(prec_g_1_xgb, 4)
prec_b_1_xgb <- cm_b_1_xgb$TP / (cm_b_1_xgb$TP + cm_b_1_xgb$FP)
round(prec_b_1_xgb, 4)

cm_g_2_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, xgb_pred_2)))
cm_b_2_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b_2)))

prec_g_2_xgb <- cm_g_2_xgb$TP / (cm_g_2_xgb$TP + cm_g_2_xgb$FP)
round(prec_g_2_xgb, 4)
prec_b_2_xgb <- cm_b_2_xgb$TP / (cm_b_2_xgb$TP + cm_b_2_xgb$FP)
round(prec_b_2_xgb, 4)

cm_g_3_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, xgb_pred_3)))
cm_b_3_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b_3)))

prec_g_3_xgb <- cm_g_3_xgb$TP / (cm_g_3_xgb$TP + cm_g_3_xgb$FP)
round(prec_g_3_xgb, 4)
prec_b_3_xgb <- cm_b_3_xgb$TP / (cm_b_3_xgb$TP + cm_b_3_xgb$FP)
round(prec_b_3_xgb, 4)

#cm_b_4_xgb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, xgb_pred_b_4)))
#
#prec_b_4_xgb <- cm_b_4_xgb$TP / (cm_b_4_xgb$TP + cm_b_4_xgb$FP)
#round(prec_b_4_xgb, 4)

#Точность модели NB отдельно для каждой категории
cm_g_nb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data$test, nb_pred)))
cm_b_nb <- as.data.frame(as.matrix(multilabel_confusion_matrix(data_b$test, nb_pred_b)))

prec_g_nb <- cm_g_nb$TP / (cm_g_nb$TP + cm_g_nb$FP)
round(prec_g_nb, 4)
prec_b_nb <- cm_b_nb$TP / (cm_b_nb$TP + cm_b_nb$FP)
round(prec_b_nb, 4)

################################################################################
rf_model_b_2 <- br(data_b$train, "RF", ntree = 2000, importance = TRUE)
rf_pred_b_2 <- predict(rf_model_b_2, data_b$test)
rf_res_b_2 <- multilabel_evaluate(data_b$test, rf_pred_b_2)

xgb_model_b_4 <- br(data_b$train, "XGB", max.depth = 8, eta = 0.02, objective = "binary:logistic", nrounds = 1000)
xgb_pred_b_4 <- predict(xgb_model_b_4, data_b$test)
xgb_res_b_4 <- multilabel_evaluate(data_b$test, xgb_pred_b_4)

rf_model_4 <- br(data$train, "RF", ntree = 2750, importance = TRUE)
rf_pred_4 <- predict(rf_model_4, data$test)
rf_res_4 <- multilabel_evaluate(data$test, rf_pred_4)