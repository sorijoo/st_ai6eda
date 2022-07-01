import streamlit as st 
import pandas as pd 
import numpy as np 
from html_module import section, callout, line_break, title
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



st.title("MAKE YOUR OWN MACHINE LEARNING MODEL!")

st.write("""
# Employee Future Prediction
this app predict Employee's Future
""")


def ConvertEducation(x):
    if x == "Bachelors":
        return 1
    elif x == "Masters":
        return 2
    else :
        return 3         

def ConvertCity(x):
    if x == "Bangalore":
        return 1
    elif x == "Pune":
        return 2
    else :
        return 3    


def scoreModel(model, X_train, X_valid, y_train, y_valid):
    '''
        머신러닝 모델과 X_train, X_valid, y_train, y_valid 변수를 받아서
        모델명, 학습용 세트 정확도(R2 score), 테스트 세트 정확도(R2 score)를 출력하는 함수
    '''
    print("모델 : {}".format(model))
    print("학습용 세트 정확도: {:.3f}".format(model.score(X_train, y_train)))
    valid_score = model.score(X_valid, y_valid)
    print("검증 세트 정확도: {:.3f}".format(valid_score))
    return valid_score        





df = pd.read_csv("data/Employee.csv")



#데이터 전처리

df["Gender_n"] = df["Gender"].apply(lambda x : int(x=="Female"))
df["EverBenched_n"] = df["EverBenched"].apply(lambda x : int(x=="Yes"))
df["Education_n"] = df["Education"].map(lambda x :ConvertEducation(x))
df["City_n"] = df["City"].map(lambda x :ConvertCity(x))


#예측 데이터 정하기

label_name = "LeaveOrNot"
feature_names = df.columns.tolist()

no = [label_name, 'Education', 'City', 'Gender','EverBenched']
for i in no:
    feature_names.remove(i) 

X = df[feature_names]
y = df[label_name]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 42)


## 모델 선택 - 로지스틱 리그레션

section('Logistic Regression', 300)
col1, col2 = st.columns(2)
with col1:
    # metric = st.radio('metric', ['mae', 'mse', 'rmse'])
    penalty = st.selectbox('Penalty', ('l2', 'none'))
    line_break()
    c = float(st.slider('c', 0.0, 100.0, 10.0))

with col2:
    solver = st.selectbox('solver', ('newton-cg','lbfgs', 'liblinear','sag','saga'))
    line_break()
    max_iter = float(st.slider('max_iter', 500.0, 10000.0, 7000.0))
 

line_break()

model_lr = st.button('Logistic Regression 모델링 Start')

if model_lr:
    # lr_params = set_lr_params(penalty, c, solver, max_iter)
    lr = LogisticRegression(penalty=penalty, C=c, solver=solver, max_iter=max_iter)

    # my_bar = st.progress(0)
    lr_model_state = st.text('인생을 예측하는 데에는 시간이 걸립니다')
    lr_model = lr.fit(X_train, y_train)
    # for percent_complete in range(100):
    # time.sleep(0.1)
    # my_bar.progress(percent_complete + 1)
    lr_model_state.success("모델링 완료")

    y_predict = lr_model.predict(X_test)
    score = scoreModel(lr_model, X_train, X_test, y_train, y_test)
    st.write("이 모델의 예측력은 : ", round(score,3))
   

## 모델 선택 - 로지스틱 리그레션

section('Random Forest', 300)
col1, col2 = st.columns(2)
with col1:
    # metric = st.radio('metric', ['mae', 'mse', 'rmse'])
    n_estimators = int(st.slider('N estimators', 5, 30, 10))
    line_break()
    min_samples_split = int(st.slider('Min samples split', 1, 30, 10))
    line_break()
    min_samples_leaf = float(st.slider('Min samples leaf', 0.0, 0.5, 0.3))


with col2:
    max_depth= int(st.slider('Max depth', 1, 10, 3))
    line_break()
    max_leaf_nodes= int(st.slider('Max leaf nodes', 1, 10, 3))
 

line_break()

model_rf = st.button('Random Forest 모델링 Start')

if model_rf:
    # lr_params = set_lr_params(penalty, c, solver, max_iter)
    rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

    # my_bar = st.progress(0)
    rf_model_state = st.text('인생을 예측하는 데에는 시간이 걸립니다')
    rf_model = rf.fit(X_train, y_train)
    # for percent_complete in range(100):
    # time.sleep(0.1)
    # my_bar.progress(percent_complete + 1)
    rf_model_state.success("모델링 완료")

    y_predict = rf_model.predict(X_test)
    score = scoreModel(rf_model, X_train, X_test, y_train, y_test)
    st.write("이 모델의 예측력은 : ", round(score,3))






# clf=RandomForestClassifier()
# clf.fit(X,Y)

# y_predict = lr.fit(X_train, y_train).predict(X_test)
# # prediction_proba=clf.predict_proba(df)

# st.subheader('class label and there corresponding index number ')
# st.write(iris.target_names)

# st.subheader('prediction ')
# st.write(iris.target_names[prediction])

# st.subheader('prediction probability ')
# st.write(prediction_proba)
