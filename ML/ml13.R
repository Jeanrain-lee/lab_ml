# 모델 성능 평가

# model < - naiveBayes(훈련 데이터, ...)
# predict <- predict(model, 테스트 데이터, type = "...")

# Naive Bayes 알고리즘에서 스팸 메시지 분류 결과/확률 정리
sms_results <- read.csv(file = "mlwr/sms_results.csv")
head(sms_results)

library(dplyr)
# spam/ham 분류에서 예측 확률이 50% 근처인 경우 - 예측하기 애매한 경우
# 모델이 잘못 예측할 가능성이 크다.
sms_results %>% 
  filter(prob_spam > 0.4 & prob_spam < 0.6) %>% 
  head(n = 10)

# 실제 값과 예측 값이 다른 경우
# 모델이 굉장히 확신(99%)함에도 불구하고 예측이 틀리는 경우도 있음.
sms_results %>% 
  filter(actual_type != predict_type) %>% 
  tail(n = 10)


# 혼동 행렬(confusion matrix)
table(sms_results$actual_type, sms_results$predict_type)

library(gmodels)
CrossTable(sms_results$actual_type, sms_results$predict_type)

# kappa 통계량 계산
# Pr(a): 실제 일치(actual agreement) 비율
# TN + TP
pr_a <- 0.865 + 0.109
pr_a
# Pr(e): 예상 일치(expected agreement) 비율
# 독립 사건이라는 가정 아래에서
# P(실제 스팸) x P(예측 스팸) + P(실제 햄) x P(예측 햄)
pr_e <- 0.132 * 0.112 +  0.868 * 0.888
kappa <- (pr_a - pr_e) / (1 - pr_e)

# caret 패키지: Classification And REgression Training 
install.packages("caret")
library(caret)

CrossTable(sms_results$actual_type, sms_results$predict_type)
confusionMatrix(data = sms_results$predict_type, 
                reference = sms_results$actual_type,
                positive = "spam")
  # data = 예측 결과, reference = 실제 결과 
# Positive Predictive Value -> 정밀도(precision)
  # positive = 관심 클래스

# 민감도
sensitivity(data = sms_results$predict_type,
            reference = sms_results$actual_type,
            positive = "spam")
# 특이도
specificity(data = sms_results$predict_type,
            reference = sms_results$actual_type,
            negative = "ham")
precision(data = sms_results$predict_type,
          reference = sms_results$actual_type,
          relevant = "spam")
# F-척도 = (2 * precision * recall) / (precsion + recall)
F_meas(data = sms_results$predict_type,
       reference = sms_results$actual_type,
       relevant = "spam")
f <- (2 * 0.974359 * 0.8306011) / (0.974359 + 0.8306011)
f

# ROC(Receiver Operation Characteristic) 곡선
install.packages("pROC")
library(pROC)

sms_roc <- roc(response = sms_results$actual_type,
               predictor = sms_results$prob_spam)
plot(sms_roc, col = "blue", lwd = 3)

sms_knn <- read.csv(file = "mlwr/sms_results_knn.csv")
head(sms_knn)
sms_knn_roc <- roc(response = sms_results$actual_type,
                   predictor = sms_knn$p_spam)
plot(sms_knn_roc, col = "red", lwd = 3, add = T)

# AUC: Area Under the Curve
auc(sms_roc)
auc(sms_knn_roc)

# k-fold CV(Cross Validation, 교차 검증)
# caret::createFolds()
