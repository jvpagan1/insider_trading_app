#  The app is a tool for identifying insider trading in the stock market.  The app is based on
#  the work of Eugene Fama (1969) and Craig Mackinlay (1997).  
#
#  See references below:
#    1)Fama, Fisher, Jensen and Roll, The Adjustment of Stock Prices to New Information, 1969, and
#    2)MacKinlay, Craig, Event Studies in Economics and Finance, 1997
#
#  Certain tables and charts were inspired on the work of Jean-Baptiste Lemaire on his API for 
#  event study analysis.  Many thanks to Jean-Baptiste Lemaire for his excelent website.  
#  For more information, please visit https://github.com/LemaireJean-Baptiste/eventstudy.
# 
#  The app is written in Python, Flask and Bokeh. 

from flask import Flask, render_template, request, redirect
import requests
import pandas as pd
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html, components
from bokeh.layouts import gridplot

import os
from ediblepickle import checkpoint
from retrying import retry
import time
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
from sklearn.metrics import r2_score
import math
from scipy.stats import t
import statsmodels.api as sm
from bokeh.models import Band, ColumnDataSource, Span, Range1d,HoverTool
from bokeh.models import Legend
import numpy as np

import yfinance as yf

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

# @checkpoint(key=lambda args, kwargs: args[0] + "_" + args[1] + '.p', work_dir=cache_dir)
@retry
def getRequest(stock_symbol, benchmark_symbol):
    return yf.download([stock_symbol,benchmark_symbol],period="max")["Adj Close"]

def get_date1(date,stock_returns):   
#   Returns first trading date commencing on date    
    new_date = stock_returns[date:].index[0]
    return datetime.date.isoformat(new_date)

def get_date2(date,n,stock_returns):
#   Returns stock market trading date commencing n periods after date.
    if n > 0:
        new_date = stock_returns[date:].index[n]
    else:
        new_date = stock_returns[:date].index[n-1]
    return datetime.date.isoformat(new_date)



def get_AR(stock_returns,stock_symbol,benchmark_symbol
           ,event_date,event_window_before,
           event_window_after,estimation_period,model="mkt"):
    #  Calculate abnormal returns
    
    # Date of announcement
    print("eventdate0: ",stock_returns[event_date:])
    event_date = get_date1(event_date,stock_returns)
    # First date in event window
    print("eventdate1: ",event_date)
    print("window_before: ",event_window_before)
    print("stock_returns",stock_returns)
    begin_date = get_date2(event_date,
                           event_window_before,stock_returns)
    # Last date in event window
    end_date = get_date2(event_date,event_window_after,stock_returns)
    # First date in estimation period
    begin_date_eval = get_date2(event_date,event_window_before
                                - estimation_period,stock_returns)
    # Last date in estimation period
    end_date_eval = get_date2(event_date,
                              event_window_after-1,stock_returns)
    # Event time in periods
    time=range(event_window_before, event_window_after + 1)
    
    data_eval = (stock_returns[[stock_symbol]+
                    [benchmark_symbol]][begin_date_eval:end_date_eval])
    data = (stock_returns[[stock_symbol]+
                          [benchmark_symbol]][begin_date:end_date])
    
    X_eval = data_eval[[benchmark_symbol]]
    y_eval = data_eval[stock_symbol]
    X = data[[benchmark_symbol]]
    y = data[stock_symbol]
    if model == "mean":
        y_eval_pred = [np.mean(y_eval)]*len(y_eval)
        y_pred = np.mean(y)
        AR_eval = y_eval - y_pred
    elif model == "mkt":
        lr = LinearRegression().fit(X_eval,y_eval)
        y_eval_pred = lr.predict(X_eval)
        y_pred = lr.predict(X)
        AR_eval = y_eval - y_eval_pred
    AR = y - y_pred
    AR.index = time
    r2 = r2_score(y_eval,y_eval_pred)
    return AR,np.std(AR_eval),r2
    return 0,0,0


def plot_stock_prices_and_returns(stock_symbol,benchmark_symbol,stock_prices, stock_returns,r2, event_date):
    source = ColumnDataSource(data=dict(
        x=pd.to_datetime(stock_prices.index),
        prices0=stock_prices[stock_symbol],
        prices1=stock_prices[benchmark_symbol],
        returns0=stock_returns[stock_symbol],
        returns1=stock_returns[benchmark_symbol]))
        
    graph1 = figure(x_axis_type = "datetime", sizing_mode="scale_width",
               title = stock_symbol
               + " " + "Stock Prices", plot_width=550, plot_height=300) 

    graph1.line('x','prices0',source=source, 
            legend_label = stock_symbol,
            color = "blue")
    graph1.legend.location = "top_left"
    graph1.line('x','prices1',source=source,  
            legend_label = benchmark_symbol,
            color = "red")
    graph1.xaxis.axis_label = 'Date'
    graph1.yaxis.axis_label = 'Adjusted Closing Price'
    graph1.add_tools(HoverTool(
        tooltips=[
            ( 'Date',   '$x{%F}'          ),
            ( stock_symbol,  '@prices0{6.2f}' ), # use @{ } for field names with spaces
            ( benchmark_symbol,'@prices1{6.2f}' ),
        ],

        formatters={
            '$x'      : 'datetime', # use 'datetime' formatter for 'date' field
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    ))
    
    graph2 = figure(x_axis_type = "datetime",x_range=graph1.x_range,
           title = f'{stock_symbol} Stock Returns (R2={r2:.2f})',
                    plot_width = 550, plot_height = 300) 
    graph2.line('x','returns0',source=source, 
            legend_label = stock_symbol,
            color = "blue")
    graph2.line('x','returns1',source=source, 
            legend_label = benchmark_symbol,
            color = "red")
    graph2.xaxis.axis_label = 'Date'
    graph2.yaxis.axis_label = 'Daily Return'
    graph2.add_tools(HoverTool(
        tooltips=[
            ( 'Date',   '$x{%F}'          ),
            ( stock_symbol,  '@returns0{.2f}' ), # use @{ } for field names with spaces
            ( benchmark_symbol,'@returns1{.2f}' ),
        ],

        formatters={
            '$x'      : 'datetime', # use 'datetime' formatter for 'date' field
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    ))


    return gridplot([graph1,graph2],ncols=2)



def plot_event(df,stock_symbol,event_date,model):
    if model == "mkt":
        model_name = "Market Model"
    elif model == "mean":
        model_name = "Constant Mean Return Model"
    y_max=max(list(df.y)+list(df.upper)+list(df.lower))
    y_min=min(list(df.y)+list(df.upper)+list(df.lower))
    yrange = Range1d(y_min*1.5,y_max*1.5)
    
    # instantiating the figure object 
    graph = figure(x_axis_type = "linear", title = stock_symbol
                   + " " + "Stock Prices") 

    source = ColumnDataSource(df.reset_index())
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    graph = figure(tools=TOOLS,y_range = yrange)
    graph.line(x='x', 
                y='y', source=source, 
    #             legend_label = CAR.columns[0],
                color = 'blue'
              ) 
    graph.vbar(x='x', 
            top='AR', source=source, 
            width = .1,
            fill_color='blue',
            fill_alpha = 1)

    band = Band(base='x', lower='lower', upper='upper', 
                source=source, level='underlay',
                fill_alpha=1.0, fill_color='lightgrey',
                line_width=1, line_color=None)
    graph.add_layout(band)
    vline = Span(location=0, dimension='height', line_color='gray',
                 line_width=2, line_dash='dashed')
    # Horizontal line
    hline = Span(location=0, dimension='width', line_color='black',
                 line_width=2)
    graph.renderers.extend([vline, hline])
    graph.title.text = (stock_symbol + " Cummulative Abnormal Returns ("
                        +model_name+")")
    graph.xgrid[0].grid_line_color=None
    graph.ygrid[0].grid_line_alpha=0.5
    graph.xaxis.axis_label = 'Peiods (0='+event_date+')'
    graph.yaxis.axis_label = 'Daily Pct Returns'
    return graph


app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/get_stock_data')


@app.route('/get_stock_data')
def get_stock_data():


    try:
        error_msg = ""
        stock_symbol = request.args.get('stock_symbol','AAPL')
        benchmark_symbol = request.args.get('benchmark_symbol','SPY')
        event_date = request.args.get('event_date','2000-09-28')
        event_window_before = int(request.args.get(
            'event_window_before','-5'))
        event_window_after = int(request.args.get(
            'event_window_after','20'))
        estimation_period = int(request.args.get(
            'estimation_period','150'))
        model =  request.args.get('model','mkt')
        print(event_window_before,event_window_after)

        stock_prices=getRequest(stock_symbol,benchmark_symbol)
        if stock_prices.size !=2:
            error_msg = error_msg+"<br>Incorrect stock symbol or benchmark symbol"

            

        print(stock_prices[1:])

        stock_returns = stock_prices.pct_change().iloc[1:]
        print(stock_returns)




        AR,stderr,r2 = get_AR(stock_returns,stock_symbol,
                           benchmark_symbol,event_date,event_window_before,
                        event_window_after,estimation_period,model)    

        p = plot_stock_prices_and_returns(stock_symbol,
                benchmark_symbol,stock_prices[1:], stock_returns,r2,event_date)





    #     print(AR)
        CAR = AR.cumsum() 
    #     print(CAR)
    #     print(stderr)
        confidence = .90 # confidence level   
        deg_freedom = estimation_period-1.0 # degrees of freedom
        # abnormal returns (event window)
        # array with standard deviation of abnormal returns for estimation period
        stderr_AR = np.array([stderr]*len(AR))
        stderr_CAR = np.sqrt(np.cumsum(stderr_AR**2))
        tstat = CAR/stderr_CAR
        pvalue = 1.0 - t.cdf(np.abs(tstat), deg_freedom)
#         pvalue = 1.0 - t.cdf((-tstat), deg_freedom)
        notes = ['*'* sum(v<(0.01,.05,.1)) for v in pvalue]  # used to show 90%, 95% and 99% confidence

        delta = stderr_CAR * t.ppf(confidence, deg_freedom)
        upper = CAR + delta
        lower =  CAR - delta
        df = pd.DataFrame(CAR)
        df['upper'] = + delta
        df['lower'] =  - delta
        df['AR'] = AR
        df = df.reset_index()
        df.columns = ["x","y","upper","lower","AR"]
        graph=plot_event(df,stock_symbol,event_date,model)

        table = pd.DataFrame()
        table['AR'] = np.round(AR,3)
        table['Std. E. AR'] = np.round(stderr_AR,5)
        table['CAR'] = np.round(CAR,3)
        table['Std.E. CAR'] = np.round(stderr_CAR,5)
        table['t-stat'] = np.round(tstat,2)
        table['p-value'] = np.round(pvalue,2)
        table["notes"] = notes
        table.index.names = ["Period"]

        print("test event date: ",type(event_date),event_date)
        print(table)

        scriptp, divp = components(p)
        script, div = components(graph)



    except:
        print("Input Error")
        wrapper =   """<html>
        <br><br>OOPs!  There seems to be an input error. Please go back to 
         <a class="active" href="/">Home</a><br>      
    </html>""" + error_msg
        return wrapper
    else:
        return render_template('index4.html',
           scriptp = scriptp, divp = divp,
            script = script, div = div,
            stock_symbol = stock_symbol,
            benchmark_symbol = benchmark_symbol,
            event_date = event_date,
            event_window_before = (event_window_before),
            event_window_after = (event_window_after),
            estimation_period = (estimation_period),
            model=model,
            tables=[table.to_html(col_space=50,
                justify='right', border=0,index_names=True, header=True)])



@app.route('/back_to_index', methods=['POST'])
def back_to_index():
    return redirect('/')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port, debug=True)
