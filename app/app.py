import os
import datetime
from flask import Flask, render_template, request, redirect
import pandas as pd
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import Band, ColumnDataSource, Span, Range1d,HoverTool
from retrying import retry
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import t
import numpy as np
from ediblepickle import checkpoint


CACHE_DIR = 'cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

@checkpoint(key=lambda args, kwargs: args[0] + "_" + args[1] + datetime.datetime.now().date().isoformat() + '.p', work_dir=CACHE_DIR)
@retry
def get_request(stock_symbol, benchmark_symbol):
    """ Get request from Yahoo Finance

    Parameters:

        stock_symbol (str): stock ticker symbol
        benchmark_symbol (str): ticker symbol of ETF to be used as proxy for market.  Ex. 'SPY' or 'RYT'.

    Returns:

        Returns pandas dataframe with historical stock prices.

    """
    return yf.download([stock_symbol,benchmark_symbol],period="max")["Adj Close"]



def get_next_date(date,stock_returns):
#   
    """ Get next trading date

    Parameters:

        date (str): date in "YYYY-MM-DD" format
        stock_returns (pandas dataframe): contains historical stock returns.  Index is date.

    Returns:

        Returns first trading date commencing on date
    """
    new_date = stock_returns[date:].index[0]
    return datetime.date.isoformat(new_date)

def get_offset_date(date,offset,stock_returns):
#   Returns stock market trading date commencing n periods after date.
    """ Get trading data from date plus offset

    Parameters:

        date (str): date in "YYYY-MM-DD" format
        offset (int): number of trading days to offset date by
        stock_returns (pandas dataframe): contains historical stock returns.  Index is date.

    Returns:

        Returns stock market trading date offset periods from date.
    """
    if offset > 0:
        new_date = stock_returns[date:].index[offset]
    else:
        new_date = stock_returns[:date].index[offset-1]
    return datetime.date.isoformat(new_date)

def get_stock_returns_data(stock_returns,stock_symbol,benchmark_symbol
           ,event_date,event_window_before,
           event_window_after,estimation_period):
    """ Get stock returns data

    Parameters:

        stock_returns (pandas dataframe): contains historical stock returns.  Index is date.
        stock_symbol (str): stock ticker symbol
        benchmark_symbol (str): ticker symbol of ETF to be used as proxy for market.  Ex. 'SPY' or 'RYT'.
        event_date (str): date in "YYYY-MM-DD" format
        event_window_before (int): number of trading days prior to event to include in event window.  Ex. -5
        event_window_after (int): number of trading days after event to include in event window.  Ex. -5 
        estimation_period (int): number of trading days prior to event window to use in estimation window.

    Returns:

        event_time (range) : Periods in event window.  
            Ex.: [-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        data_eval (pandas dataframe): contains returns for stock and benchmark during evaluation window
        data (pandas dataframe): contains returns for stock and benchmark during event window

    """
    # Date of event announcement
    event_date = get_next_date(event_date,stock_returns)
    # First date in event window
    begin_date = get_offset_date(event_date,
                           event_window_before,stock_returns)
    # Last date in event window
    end_date = get_offset_date(event_date,event_window_after,stock_returns)
    # First date in estimation period
    begin_date_eval = get_offset_date(event_date,event_window_before
                                - estimation_period,stock_returns)
    # Last date in estimation period
    end_date_eval = get_offset_date(event_date,
                              event_window_after-1,stock_returns)
    # Event time in periods
    event_time=range(event_window_before, event_window_after + 1)

    data_eval = (stock_returns[[stock_symbol]+
                    [benchmark_symbol]][begin_date_eval:end_date_eval])
    data = (stock_returns[[stock_symbol]+
                          [benchmark_symbol]][begin_date:end_date])
    return event_time, data_eval, data

def get_abnormal_returns(stock_symbol,benchmark_symbol,event_time, data_eval, data,model="mkt"):
    """ Get abnormal returns data

    Parameters:

        stock_symbol (str): stock ticker symbol
        benchmark_symbol (str): ticker symbol of ETF to be used as proxy for market.  Ex. 'SPY' or 'RYT'.
        event_time (range) : Periods in event window.  Ex.: [-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        data_eval (pandas dataframe): contains returns for stock and benchmark during evaluation window
        data (pandas dataframe): contains returns for stock and benchmark during event window
        model (str): "market" or "mean" to indicate the type of model used to calculate estimated (normal) returns

    Return:

        AR (pandas dataframe): contains daily abnormal returns during event window
        stderr (float): stderr of abnormal returns during evealuation period
        r2 (float) : coefficient of determination of linear regression used to estimate stock expected (normal) 
            returns based on benchmark returns during evaluation period.

    """
    eval_X = data_eval[[benchmark_symbol]]
    eval_y = data_eval[stock_symbol]
    X = data[[benchmark_symbol]]
    y = data[stock_symbol]
    if model == "mean":
        eval_y_pred = [np.mean(eval_y)]*len(eval_y)
        y_pred = np.mean(eval_y) # evaluation period mean
        AR_eval = eval_y - y_pred
    elif model == "mkt":
        lr = LinearRegression().fit(eval_X,eval_y)
        eval_y_pred = lr.predict(eval_X)
        y_pred = lr.predict(X)
        AR_eval = eval_y - eval_y_pred
    AR = y - y_pred
    AR.index = event_time
    r2 = r2_score(eval_y,eval_y_pred)
    return AR,np.std(AR_eval),r2


def plot_stock_prices_and_returns(stock_symbol,benchmark_symbol,stock_prices, stock_returns,r2):
    """ Plot historical stock prices and returns

    Parameters:

        stock_symbol (str): stock ticker symbol
        benchmark_symbol (str): ticker symbol of ETF to be used as proxy for market.  Ex. 'SPY' or 'RYT'.
        stock_prices (pandas dataframe): contains historical stock prices.  Index is date.
        stock_returns (pandas dataframe): contains historical stock returns.  Index is date.
        r2 (float): coefficient of determination of linear regression to predict expected stock returns
            based on benchmark returns.

    Return:

        Two synchronized Bokeh graphs with historical daily prices and returns for the stock and the benchmark.

    """
    source = ColumnDataSource(data=dict(
            x=pd.to_datetime(stock_prices.index),
            prices0=stock_prices[stock_symbol],
            prices1=stock_prices[benchmark_symbol],
            returns0=stock_returns[stock_symbol],
            returns1=stock_returns[benchmark_symbol]))

    graph1 = figure(x_axis_type = "datetime", sizing_mode="scale_width",
               title = stock_symbol + " " + "Stock Prices", plot_width=550, plot_height=300)

    graph1.line('x','prices0',source=source,legend_label = stock_symbol,color = "blue")
    graph1.legend.location = "top_left"
    graph1.line('x','prices1',source=source,legend_label = benchmark_symbol,color = "red")
    graph1.xaxis.axis_label = 'Date'
    graph1.yaxis.axis_label = 'Adjusted Closing Price'
    graph1.add_tools(HoverTool(
        tooltips=[( 'Date','$x{%F}'), (stock_symbol, '@prices0{6.2f}'),(benchmark_symbol,'@prices1{6.2f}'),],
        formatters={'$x': 'datetime'}))

    graph2 = figure(x_axis_type = "datetime",x_range=graph1.x_range,
           title = f'{stock_symbol} Stock Returns (R2={r2:.2f})',plot_width = 550, plot_height = 300)
    graph2.line('x','returns0',source=source,legend_label = stock_symbol,color = "blue")
    graph2.line('x','returns1',source=source,legend_label = benchmark_symbol,color = "red")
    graph2.xaxis.axis_label = 'Date'
    graph2.yaxis.axis_label = 'Daily Return'
    graph2.add_tools(HoverTool(
        tooltips=[( 'Date','$x{%F}'), (stock_symbol, '@returns0{6.2f}'),(benchmark_symbol,'@returns1{6.2f}'),],
        formatters={'$x': 'datetime'}))


    return gridplot([graph1,graph2],ncols=2)



def plot_event(df,stock_symbol,event_date,model):
    """ Plot event study results

    Parameters:

        df (pandas dataframe): contains daily abnormal returns and cumulative returns for the event windos
        stock_symbol (str): stock ticker symbol
        event_date (str): date in "YYYY-MM-DD" format        
        model (str): "market" or "mean" to indicate the type of model used to calculate 
            estimated (normal) returns

    Return:

        Bokeh graph with abnormal returns (bar chart) and cumulative abnormal returns (line chart) 
            for event window.

    """
    # Set model name variable
    if model == "mkt":
        model_name = "Market Model"
    elif model == "mean":
        model_name = "Constant Mean Return Model"
        
    # Set y margins
    y_max=max(list(df.y)+list(df.upper)+list(df.lower))
    y_min=min(list(df.y)+list(df.upper)+list(df.lower))
    yrange = Range1d(y_min*1.5,y_max*1.5)

    # instantiating the figure object
    graph = figure(x_axis_type = "linear", title = stock_symbol+ " " + "Stock Prices")

    source = ColumnDataSource(df.reset_index())
    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    graph = figure(tools=TOOLS,y_range = yrange)
    graph.line(x='x', y='y', source=source, color = 'blue')
    graph.vbar(x='x',top='AR', source=source,width = .1,fill_color='blue',fill_alpha = 1)
    
    # Plot shaded area for 90% confidence
    band = Band(base='x', lower='lower', upper='upper',source=source, level='underlay',
                fill_alpha=1.0, fill_color='lightgrey',line_width=1, line_color=None)
    graph.add_layout(band)
    # Add dashed vertical line on event date 
    vline = Span(location=0, dimension='height', line_color='gray',line_width=2, line_dash='dashed')
    hline = Span(location=0, dimension='width', line_color='black',line_width=2) 
    graph.renderers.extend([vline, hline])
    
    graph.title.text = (stock_symbol + " Cummulative Abnormal Returns ("+model_name+")")
    graph.xgrid[0].grid_line_color=None
    graph.ygrid[0].grid_line_alpha=0.5
    graph.xaxis.axis_label = 'Peiods (0='+event_date+')'
    graph.yaxis.axis_label = 'Daily Returns'
    return graph


app = Flask(__name__)

@app.route('/')
def index():
    return redirect('/get_stock_data')


@app.route('/get_stock_data')
def get_stock_data():
    """ Returns event study based on user defined inputs.

    Parameters:
    
        None
        

    Return:

        1) Two synchronized Bokeh graphs with historical daily prices and returns for the stock and the benchmark.
        2) A Bokeh graph with abnormal returns (bar chart) and cumulative abnormal returns (line chart) 
            for event window.
        3) A results table with abnormal returns, cumulative abnormal returns, t-statistics and p-values for the event window.

    """

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

        stock_prices=get_request(stock_symbol,benchmark_symbol)
        if stock_prices.size !=2:
            error_msg = error_msg+"<br>Incorrect stock symbol or benchmark symbol"

        stock_returns = stock_prices.pct_change().iloc[1:]

        event_time, data_eval, data = get_stock_returns_data(stock_returns,stock_symbol,benchmark_symbol,
                        event_date,event_window_before,event_window_after,estimation_period)

        AR,stderr,r2 = get_abnormal_returns(stock_symbol,benchmark_symbol,event_time, 
                        data_eval, data, model)

        p = plot_stock_prices_and_returns(stock_symbol,
                benchmark_symbol,stock_prices[1:], stock_returns,r2)


        CAR = AR.cumsum()
        confidence = .90 # confidence level
        deg_freedom = estimation_period-1.0 # degrees of freedom
        # abnormal returns (event window)
        # array with standard deviation of abnormal returns for estimation period
        stderr_AR = np.array([stderr]*len(AR))
        stderr_CAR = np.sqrt(np.cumsum(stderr_AR**2))
        tstat = CAR/stderr_CAR
        pvalue = 1.0 - t.cdf(np.abs(tstat), deg_freedom)
        # pvalue = 1.0 - t.cdf((-tstat), deg_freedom)
        # used to show 90%, 95% and 99% confidence
        notes = ['*'* sum(v<(0.01,.05,.1)) for v in pvalue]
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
        
        scriptp, divp = components(p)
        script, div = components(graph)

    except:
        # On exception, return error message page
        wrapper =  error_msg + """<html>
        <br><br>OOPs!  There seems to be an input error. Please go back to
         <a class="active" href="/">Home</a><br></html>""" 
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


@app.route('/references')
def references():
    return render_template('references.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='127.0.0.1', port=port)
