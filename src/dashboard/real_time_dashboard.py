import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

class RealTimeDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Stock Trading Dashboard'),

            # Stock selection
            html.Div([
                html.Label('Stock Symbol:'),
                dcc.Input(id='stock-input', value='AAPL', type='text'),
                html.Button('Load', id='load-button')
            ]),

            # Charts
            html.Div([
                dcc.Graph(id='candlestick-chart'),
                dcc.Graph(id='volume-chart'),
                dcc.Graph(id='indicators-chart')
            ]),

            # Technical indicators
            html.Div([
                html.H3('Technical Indicators'),
                html.Div(id='technical-indicators')
            ]),

            # Auto refresh
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # 1 minute
                n_intervals=0
            )
        ])

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def setup_callbacks(self):
        @self.app.callback(
            [Output('candlestick-chart', 'figure'),
             Output('volume-chart', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('technical-indicators', 'children')],
            [Input('load-button', 'n_clicks'),
             Input('stock-input', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_charts(n_clicks, symbol, n_intervals):
            if not symbol:
                return {}, {}, {}, ''

            # Get data
            stock = yf.Ticker(symbol)
            df = stock.history(period='1mo', interval='1d')
            df = self.calculate_indicators(df)

            # Candlestick chart
            candlestick = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close']
            )])
            candlestick.update_layout(title=f'{symbol} Price')

            # Volume chart
            volume = go.Figure(data=[go.Bar(
                x=df.index,
                y=df['Volume']
            )])
            volume.update_layout(title='Volume')

            # Indicators chart
            indicators = go.Figure()
            indicators.add_trace(go.Scatter(
                x=df.index, y=df['SMA_20'],
                name='SMA 20'
            ))
            indicators.add_trace(go.Scatter(
                x=df.index, y=df['SMA_50'],
                name='SMA 50'
            ))
            indicators.update_layout(title='Technical Indicators')

            # Technical indicators text
            current_rsi = df['RSI'].iloc[-1]
            current_sma20 = df['SMA_20'].iloc[-1]
            current_sma50 = df['SMA_50'].iloc[-1]

            indicators_text = html.Div([
                html.P(f'RSI: {current_rsi:.2f}'),
                html.P(f'SMA 20: {current_sma20:.2f}'),
                html.P(f'SMA 50: {current_sma50:.2f}')
            ])

            return candlestick, volume, indicators, indicators_text

    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    dashboard = RealTimeDashboard()
    dashboard.run()