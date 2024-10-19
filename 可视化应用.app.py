
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import streamlit as st
import matplotlib.pyplot as plt

# 数据加载和处理函数
def load_data():
    # 加载 CSV 文件
    data = pd.read_csv("well_data.csv")
    # 查看数据的前几行
    print(data)
    return data

# 训练模型的函数
def train_models(X_train, y_train):
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 随机森林
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # 梯度提升
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X_train_scaled, y_train)

    # ANN模型
    ann_model = Sequential()
    ann_model.add(Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    for _ in range(9):
        ann_model.add(Dense(10, activation='relu'))
    ann_model.add(Dense(1))
    ann_model.compile(optimizer='adam', loss='mse')
    ann_model.fit(X_train_scaled, y_train, epochs=100, verbose=0)

    # RNN模型
    X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    rnn_model = Sequential()
    rnn_model.add(LSTM(50, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    rnn_model.add(Dense(1))
    rnn_model.compile(optimizer='adam', loss='mse')
    rnn_model.fit(X_train_rnn, y_train, epochs=100, verbose=0)

    return rf_model, gb_model, ann_model, rnn_model, scaler

# 预测函数
def make_predictions(models, X_test):
    rf_model, gb_model, ann_model, rnn_model, scaler = models
    X_test_scaled = scaler.transform(X_test)
    X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    ann_pred = ann_model.predict(X_test_scaled)
    rnn_pred = rnn_model.predict(X_test_rnn)

    return rf_pred[0], gb_pred[0], ann_pred[0][0], rnn_pred[0][0]

# Streamlit界面
def main():
    st.title('P-wave Velocity Prediction')

    st.sidebar.header('Input Parameters')
    cone_resistance = st.sidebar.number_input('Cone Resistance', min_value=0.0, value=1.0)
    side_friction = st.sidebar.number_input('Side Friction', min_value=0.0, value=1.0)
    depth = st.sidebar.number_input('Depth (m)', min_value=0.0, value=1.0)

    # 加载数据
    data = load_data()
    X = data[['Cone Resistance', 'Side Friction', 'Depth']]
    y = data['P-wave Velocity']

    # 模型训练
    models = train_models(X, y)

    # 输入的数据点
    X_test = np.array([[cone_resistance, side_friction, depth]])

    # 进行预测
    rf_pred, gb_pred, ann_pred, rnn_pred = make_predictions(models, X_test)

    # 展示预测结果
    st.write(f"Random Forest Prediction: {rf_pred:.2f}")
    st.write(f"Gradient Boosting Prediction: {gb_pred:.2f}")
    st.write(f"ANN Prediction: {ann_pred:.2f}")
    st.write(f"RNN Prediction: {rnn_pred:.2f}")

    # 绘图展示预测值
    plt.figure(figsize=(10, 6))
    methods = ['Random Forest', 'Gradient Boosting', 'ANN', 'RNN']
    predictions = [rf_pred, gb_pred, ann_pred, rnn_pred]

    # 打印预测值和它们的形状用于调试
    print("Predictions:", predictions)
    print("Shapes:", [np.shape(pred) for pred in predictions])

    plt.bar(methods, predictions, color=['blue', 'green', 'orange', 'red'])
    plt.title('Prediction Results from Different Models')
    plt.ylabel('P-wave Velocity (m/s)')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
