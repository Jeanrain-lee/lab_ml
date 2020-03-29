# Regression Tree와 Model Tree

# 1. 데이터 준비
wine <- read.csv(file = "mlwr/whitewines.csv")

# 2. 데이터 확인, 전처리
str(wine)
  # 4,898 obs.(예시), 12 variables(특징) - white wine 데이터
summary(wine)  # 기술 통계량
# 종속 변수(quality)의 분포
hist(wine$quality)

# regression tree를 사용하기 위한 패키지
# rpart: recursive partioning
install.packages("rpart")
library(rpart)

# 3. 모델 학습
# 학습 데이터 세트(75%)/테스트 데이터 세트(25%)
head(wine)
tail(wine)
4898 * 0.75
wine_train <- wine[1:3674, ]
wine_test <- wine[3675:4898, ]

# 학습 데이터를 rpart 패키지를 사용해서 학습시킴
wine_rpart <- rpart(formula = quality ~ ., data = wine_train)
wine_rpart
summary(wine_rpart)

# rpart (회귀 트리) 결과를 시각적으로 보여주는 패키지: rpart.plot
install.packages("rpart.plot")
library(rpart.plot)

rpart.plot(x = wine_rpart, digits = 3)
rpart.plot(wine_rpart, digits = 4, fallen.leaves = F)

# 4. 모델 평가 - regression tree가 테스트 데이터를 얼마나 잘 설명?
wine_predict <- predict(wine_rpart, wine_test)
head(wine_predict)
tail(wine_predict)
summary(wine_predict)       # 예측 quality의 기술 통계량
summary(wine_test$quality)  # 실제 quality의 기술 통계량

# 모델 성능 평가 1)
# 상관 계수(correlation coefficient): -1 <= cor <= 1
cor(wine_predict, wine_test$quality)  # 0.54

# 모델 성능 평가 2)
# MAE(Mean Absolute Error): 평균 절대 오차
# 오차(실제값 - 예측값)들의 절대값의 평균
MAE <- function(actual, predict) {
  return(mean(abs(actual - predict)))
}

# 함수 테스트
MAE(actual = c(1, 2, 3), predict = c(1.1, 1.9, 3.0))

MAE(wine_test$quality, wine_predict)  # 0.6


# 5. 모델 성능 향상
# 모델 트리(Model Tree): 
# Regression Tree(분류) + Regression Modeling(회귀 모델 적용)
# 교재: RWeka 패키지의 M5P 함수 사용
# Cubist 패키지: 규칙 학습 기반 분류 + M5P알고리즘 회귀 모델 적용
install.packages("Cubist")
library(Cubist)

# cubist(x = 훈련데이터, y = 훈련 데이터의 결과)
# 훈련 데이터에는 레이블(클래스), 즉 종속 변수가 포함되면 안됨.
# 훈련 데이터에는 분류에 필요한 독립 변수들만 포함시킴.
wine_cubist <- cubist(x = wine_train[-12], y = wine_train$quality)
wine_cubist
summary(wine_cubist)

# 모델 트리의 성능 테스트
wine_predict2 <- predict(wine_cubist, wine_test)
head(wine_predict2)

# 기술 통계량
summary(wine_predict2)
summary(wine_test$quality)

# 상관 계수
cor(wine_predict2, wine_test$quality)  # 0.64

# MAE: 평균 절대 오차
MAE(wine_predict2, wine_test$quality) # 0.54

# 회귀 트리와 비교해서 상관 계수는 커졌고, MAE는 작아짐.
# -> 성능이 좋아졌다고 볼 수 있음.


rm(list = ls())

# red wine 데이터를 사용한 Regression Tree와 Model Tree
redwine <- read.csv(file = "mlwr/redwines.csv")
str(redwine)
summary(redwine)
head(redwine)
tail(redwine)
hist(redwine$quality)

sample_count <- round(nrow(redwine) * 0.75)
redwine_train <- redwine[1:sample_count, ]
redwine_test <- redwine[(sample_count+1):1599, ]

model_rpart <- rpart(formula = quality ~ ., data = redwine_train)
model_rpart
rpart.plot(model_rpart, digits = 3, cex = 0.55)

predict_rpart <- predict(model_rpart, redwine_test)
summary(predict_rpart)
summary(redwine_test$quality)

cor(predict_rpart, redwine_test$quality)  # 0.598
MAE(predict_rpart, redwine_test$quality)  # 0.537

model_cubist <- cubist(x = redwine_train[-12], y = redwine_train$quality)
model_cubist
summary(model_cubist)

predict_cubist <- predict(model_cubist, redwine_test)
summary(predict_cubist)
summary(redwine_test$quality)

cor(predict_cubist, redwine_test$quality)  # 0.671
MAE(predict_cubist, redwine_test$quality)  # 0.453
