import dash
from dash import html, dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from datetime import datetime
import pandas as pd
import os
import json

class StockDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.data_path = 'data/stocks/'
        self.setup_layout()
        self.setup_callbacks()
    
    def load_available_stocks(self):
        """Load all available stock data"""
        stocks = []
        for file in os.listdir(self.data_path):
            if file.endswith('_meta.json'):
                with open(os.path.join(self.data_path, file)) as f:
                    meta = json.load(f)
                    stocks.append({
                        'label': f"{meta['company_name']} ({meta['symbol']})",
                        'value': meta['symbol']
                    })
        return stocks
    
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Stock Trading Dashboard'),
            
            # Stock selector
            dcc.Dropdown(
                id='stock-selector',
                options=self.load_available_stocks(),
                value=None,
                placeholder='Select a stock...'
            ),
            
            # Price chart
            dcc.Graph(id='price-chart'),
            
            # Stock info
            html.Div(id='stock-info'),
            
            # Refresh button
            html.Button('Refresh Data', id='refresh-button'),
            
            # Auto refresh interval
            dcc.Interval(
                id='interval-component',
                interval=300*1000,  # 5 minutes in milliseconds
                n_intervals=0
            )
        ])
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('stock-info', 'children')],
            [Input('stock-selector', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_graph(selected_stock, n):
            if not selected_stock:
                return {}, ''
            
            # Load data
            df = pd.read_csv(f'{self.data_path}{selected_stock}_data.csv')
            with open(f'{self.data_path}{selected_stock}_meta.json') as f:
                meta = json.load(f)
            
            # Create figure
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines',
                name='Close Price'
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA_20'],
                mode='lines',
                name='20 Day MA',
                line=dict(dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA_50'],
                mode='lines',
                name='50 Day MA',
                line=dict(dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{meta['company_name']} Stock Price",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified'
            )
            
            # Create info div
            info = html.Div([
                html.H3('Stock Information'),
                html.P(f"Current Price: ${meta['current_price']:.2f}"),
                html.P(f"Last Updated: {meta['last_updated']}")
            ])
            
            return fig, info
    
    def run_server(self, debug=True):
        self.app.run_server(debug=debug)

if __name__ == '__main__':
    dashboard = StockDashboard()
    dashboard.run_server()