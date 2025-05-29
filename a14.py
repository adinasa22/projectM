import plotly.graph_objects as go
import pandas as pd

# Sample data (Date, Open, High, Low, Close)
data = {
    'Date': ['2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05'],
    'Open': [100, 102, 101, 105, 107],
    'High': [105, 106, 104, 108, 110],
    'Low': [99, 101, 99, 104, 106],
    'Close': [104, 103, 102, 107, 109]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

fig.update_layout(title='Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
fig.show()