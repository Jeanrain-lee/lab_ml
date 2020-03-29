rm(list = ls())

# SVM(Support Vector Machine)을 이용한 분류

# 1. 데이터 준비
letters <- read.csv(file = "mlwr/letterdata.csv")

# 2. 데이터 확인, 전처리 
str(letters)
head(letters)
table(letters$letter)

# 학습 데이터(80%) / 테스트 데이터(20%) 세트
letters_train <- letters[1:16000, ]
letters_test <- letters[16001:20000, ]

table(letters_train$letter)
table(letters_test$letter)

# 3. 모델 생성 - SVM
# kernlab 패키지
install.packages("kernlab")
library(kernlab)
search()
# detach("package:neuralnet")

# SVM 알고리즘 모델을 생성 
letter_classifier <- ksvm(letter ~ .,
                          data = letters_train,
                          kernel = "vanilladot")

# 4. 모델 평가
letters_predict <- predict(letter_classifier, letters_test)
head(letters_predict)
table(letters_predict, letters_test$letter)
prop.table(table(letters_predict, letters_test$letter))
letters_predict[1]
letters_test$letter[1]
letters_predict[1] == letters_test$letter[1]

correct <- ifelse(letters_predict == letters_test$letter, 1, 0)
correct_count <- sum(correct)
correct_count  # SVM 모델이 문자들을 제대로 구분한 갯수
correct_ratio <- correct_count / 4000
correct_ratio  # 약 0.84

# 5. 모델 수정 -> 재평가 -> 성능 개선
classifier2 <- ksvm(letter ~ .,
                    data = letters_train,
                    kernel = "rbfdot")

predict2 <- predict(classifier2, letters_test)
head(predict2, n = 10)
head(letters_test$letter, n = 10)

table(predict2, letters_test$letter)

correct2 <- ifelse(predict2 == letters_test$letter, 1, 0)
correct2_count <- sum(correct2)
correct2_count
correct2_ratio <- correct2_count / 4000
correct2_ratio  # 0.93
