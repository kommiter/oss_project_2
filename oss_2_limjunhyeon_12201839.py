import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt

# 정렬
def sort_dataset(dataset_df):
    return dataset_df.sort_values(by='year', ascending=1)

# 데이터셋 분할
def split_dataset(dataset_df):
    dataset_df['salary'] = dataset_df['salary']*0.001
    train_df = dataset_df.iloc[:1718]
    test_df = dataset_df.iloc[1718:]
    return train_df.drop('salary', axis=1), test_df.drop('salary', axis=1), train_df['salary'], test_df['salary']

# 추출
def extract_numerical_cols(dataset_df):
    numerical_cols = ['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']
    return dataset_df[numerical_cols]

# 모델 실행
def train_predict_decision_tree(X_train, Y_train, X_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    return model.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
    model = SVR()
    model.fit(X_train, Y_train)
    return model.predict(X_test)

# RMSE 계산
def calculate_RMSE(labels, predictions):
    return sqrt(mean_squared_error(labels, predictions))

# 데이터셋 가공
data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
sorted_df = sort_dataset(data_df)
X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
X_train = extract_numerical_cols(X_train)
X_test = extract_numerical_cols(X_test)

# 모델 예측
dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
svm_predictions = train_predict_svm(X_train, Y_train, X_test)

# 출력부
print("  의사 결정 트리 모델 RMSE: ", calculate_RMSE(Y_test, dt_predictions))
print("   랜덤 포레스트 모델 RMSE: ", calculate_RMSE(Y_test, rf_predictions))
print("서포트 벡터 머신 모델 RMSE: ", calculate_RMSE(Y_test, svm_predictions))