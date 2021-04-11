import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

from PIL import Image

image= Image.open('arc.jpg')
st.image(image,width=200)


START_from = "2017-01-01"
TODAY_now = date.today().strftime("%Y-%m-%d")


st.write("""
# Arcondis Make the Healthcare Better 
 WE OFFER INNOVATIVE SOLUTIONS "STRONG ACCURACY IN SAELS FORECASTING WITH AI" 
""")

#st.title('AI Effective Accuracy for Sales Forecasting')

sales = ('BTC-USD', 'GOOG', 'AAPL')
selected_sales = st.selectbox('Select dataset for the prediction', sales)

num_years = st.slider('Years of prediction:', 1, 5)
t_period = num_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START_from, TODAY_now)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_sales)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name="Sale_High"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name="Sale_Low"))
	fig.layout.update(title_text='Time Series data ', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()


# Predict forecast with Prophet.
df_train = data[['Date','Low']]
df_train = df_train.rename(columns={"Date": "ds", "Low": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=t_period)
forecast = m.predict(future)


st.subheader('AI Accuracy, Automated, Data Cleansing, Real-Time, AI self-learning ')


# Show and plot forecast
st.subheader('High Accuracy in Sales Forecast Data with AI')
st.write(forecast.tail())

st.write(f'Forecast plot for {num_years} month')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)