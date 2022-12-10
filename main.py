import yaml
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance

# Authentication
with open('./config/config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

# Login Form
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == None:
    st.warning("Please enter your name an password")
elif authentication_status == False:
    st.error("Username/Password is Incorrect")
elif authentication_status:
    authenticator.logout("Logout", "sidebar")

    # Main Page
    
    # App Title
    st.title("Onion Price Prediction using XGBoost")

    # Sidebar Title
    st.sidebar.header("Settings")

    def get_dataset(choose_data):
        if choose_data == 'Boyolali':
            df = pd.read_csv('./datasets/harga-boyolali-2018-2022.csv')
        elif choose_data == 'Brebes':
            df = pd.read_csv('./datasets/harga-brebes-2018-2022.csv')
        elif choose_data == 'Demak':
            df = pd.read_csv('./datasets/harga-demak-2018-2022.csv')
        elif choose_data == 'Kendal':
            df = pd.read_csv('./datasets/harga-kendal-2018-2022.csv')
        elif choose_data == 'Pati':
            df = pd.read_csv('./datasets/harga-pati-2018-2022.csv')
        elif choose_data == 'Tegal':
            df = pd.read_csv('./datasets/harga-tegal-2018-2022.csv')
        elif choose_data == 'Temanggung':
            df = pd.read_csv('./datasets/harga-temanggung-2018-2022.csv')
        
        return df

    def get_chart(choose_chart):
        if choose_chart == "Lineplots":
            plot = px.line(data_frame=df, x=df['Date'], y=df['Price'])
        elif choose_chart == "Scatterplots":
            plot = px.scatter(data_frame=df, x=df['Date'], y=df['Price'])

        return plot

    def train_test_split(dataframe, split_date):
        split_idx = dataframe.index[dataframe['Date'] == split_date][0]
        data_train = dataframe[:split_idx]
        data_test = dataframe[split_idx:]
        return data_train, data_test

    def create_features(dataframe, target_variable):
        dataframe = dataframe.dropna()
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe = dataframe.set_index('Date')
        dataframe['Date'] = dataframe.index
        dataframe['DayOfWeek'] = dataframe['Date'].dt.dayofweek
        dataframe['Quarter'] = dataframe['Date'].dt.quarter
        dataframe['Month'] = dataframe['Date'].dt.month
        dataframe['Year'] = dataframe['Date'].dt.year
        dataframe['DayOfYear'] = dataframe['Date'].dt.dayofyear
        dataframe['DayOfMonth'] = dataframe['Date'].dt.day
        dataframe['WeekOfYear'] = dataframe['Date'].dt.weekofyear

        X = dataframe[['DayOfWeek', 'Quarter', 'Month', 'Year',
            'DayOfYear', 'DayOfMonth', 'WeekOfYear']]

        if target_variable:
            y = dataframe[target_variable]
            return X, y
        return X

    # MAPE
    def mean_absolute_percentage_error_func(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Evaluation Metrics
    def timeseries_evaluation_metrics_func(y_true, y_pred):
        st.write(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
        st.write(f'MAE is : {metrics.mean_absolute_error(y_true, y_pred)}')
        st.write(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))/506}')
        st.write(f'MAPE is : {mean_absolute_percentage_error_func(y_true, y_pred)}')
        st.write(f'R2 is : {metrics.r2_score(y_true, y_pred)}')

    # Chart Accuracy
    def chart_acc(actual, predict):
        data_acc = {'Actual': list(actual), 'Predict': list(predict)}
        df_acc = pd.DataFrame(data_acc)

        graph_0 = go.Scatter(
            x = df_acc.index,
            y = df_acc['Actual'],
            name = "Actual",
            mode = 'lines'
        )
        graph_1 = go.Scatter(
            x = df_acc.index,
            y = df_acc['Predict'],
            name = "Predict",
            mode = 'lines',
        )
        layout = go.Layout(title = "Graph")
        plot_acc = go.Figure(data=[graph_0, graph_1], layout=layout)

        return plot_acc

    # Chart Comparison
    def chart_comp(graph0, graph1):
        graph_0 = go.Scatter(
            x = graph0["Date"],
            y = graph0["Price"],
            name = "Past",
            mode = 'lines'
        )
        graph_1 = go.Scatter(
            x = graph1["Date"],
            y = graph1["Price"],
            name = "Future",
            mode = 'lines',
        )
        layout = go.Layout(title = "Graph")
        plot_acc = go.Figure(data=[graph_0, graph_1], layout=layout)

        return plot_acc

    # Choosing Dataset
    choose_data = st.sidebar.selectbox("Choose a Dataset", options=['Boyolali', 'Brebes', 'Demak', 'Kendal', 'Pati', 'Tegal', 'Temanggung'])
    st.header(choose_data)
    st.subheader("Dataframe")
    df = get_dataset(choose_data)
    st.write(df)

    # Choosing Chart
    choose_chart = st.sidebar.selectbox(label="Select the chart type", options = ["Lineplots", "Scatterplots"])
    st.subheader("Chart: {}".format(choose_chart))
    plot = get_chart(choose_chart)
    st.plotly_chart(plot)

    # Train Test Split
    split_date = '2020-12-30'
    data_train, data_test = train_test_split(df, split_date)
    col0_1, col0_2 = st.columns(2)
    col0_1.subheader("Data Train")
    col0_1.write(data_train)
    col0_2.subheader("Data Test")
    col0_2.write(data_test)

    # Create Features
    X_train, y_train = create_features(data_train, target_variable='Price')
    X_test, y_test = create_features(data_test, target_variable='Price')

    # Session State
    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    # Fit Model
    st.subheader("Accuracy Model")
    ph_train = st.sidebar.empty()
    btn_train = ph_train.button("Start Training")
    if btn_train or st.session_state.load_state:
        st.session_state.load_state = True
        ph_train.empty()

        # Train Model (Past)
        xgb = XGBRegressor(objective='reg:linear', n_estimator=1000)
        xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0)

        # Predict (Past)
        predict = xgb.predict(X_test)
        
        # Evaluation Metrics
        timeseries_evaluation_metrics_func(y_test, predict)
        plot_acc = chart_acc(y_test, predict)
        st.plotly_chart(plot_acc)

        # -------------------------------------------------------- #
        # Predict Future
        st.subheader("Predict Future")
        ph_pred = st.sidebar.empty()
        btn_pred = ph_pred.button("Start Predicting")
        if btn_pred:
            st.session_state.load_state = False
            ph_pred.empty()

            future = pd.date_range('2022-10-01','2023-09-30')
            data_future = {'Date': future, 'Price': 0}
            df_future = pd.DataFrame(data_future)

            col1_1, col1_2 = st.columns(2)
            col1_1.subheader("Data Past")
            col1_1.write(df)
            col1_2.subheader("Data Future")
            col1_2.write(df_future)

            X_past, y_past = create_features(df, target_variable='Price')
            X_future, y_future = create_features(df_future, target_variable='Price')

            # Train Model (Future)
            xgb = XGBRegressor(objective='reg:linear', n_estimator=1000)
            xgb.fit(X_past, y_past, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=0)

            # Predict (Future)
            df_future['Price'] = xgb.predict(X_future)

            # Comparison
            st.subheader("Result")
            col2_1, col2_2 = st.columns(2)
            col2_1.subheader("Data Past")
            col2_1.write(df)
            col2_2.subheader("Data Future")
            col2_2.write(df_future)

            plot_comp = chart_comp(df, df_future)
            st.plotly_chart(plot_comp)
        else:
            st.write("Click \"Start Predict\" on the sidebar to show predictions")
    else:
        st.write("Click \"Start Training\" on the sidebar to show accuracy")

