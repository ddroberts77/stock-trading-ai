import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class StockDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Stock Trading Dashboard'),
            
            html.Div([
                html.Label('Stock Symbol:'),
                dcc.Input(
                    id='stock-input',
                    value='AAPL',
                    type='text'
                ),
                html.Button('Load Data', id='load-button', n_clicks=0)
            ]),
            
            dcc.Graph(id='price-chart'),
            
            html.Div(id='stock-info'),
            
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # updates every minute
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('stock-info', 'children')],
            [Input('load-button', 'n_clicks'),
             Input('stock-input', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_data(n_clicks, symbol, n_intervals):
            if not symbol:
                return {}, ''
            
            # Get stock data
            stock = yf.Ticker(symbol)
            df = stock.history(period='1mo')
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ))
            
            fig.update_layout(
                title=f'{symbol} Stock Price',
                yaxis_title='Price',
                xaxis_title='Date'
            )
            
            # Get current info
            info = stock.info
            current_price = info.get('regularMarketPrice', 'N/A')
            market_cap = info.get('marketCap', 'N/A')
            
            info_div = html.Div([
                html.H3('Stock Information'),
                html.P(f'Current Price: ${current_price}'),
                html.P(f'Market Cap: ${market_cap:,}' if isinstance(market_cap, (int, float)) else f'Market Cap: {market_cap}')
            ])
            
            return fig, info_div
    
    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    dashboard = StockDashboard()
    dashboard.run()