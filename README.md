# Insider Trading App

This [app](http://jpagan-event-study-app.herokuapp.com/) is a tool for identifying insider trading in the stock market using the event study 
methodology introduced by Eugene Fama in 1969, as further developed by Craig Mackinlay in 1997.
Certain tables and charts were inspired by the work of Jean-Baptiste Lemaire on his API for
event study analysis.  See references below.

The app analyzes an event for a given stock on a given date.  A benchmark stock (SPY, RYT, etc.) is used
to extract the market variance from stock returns when using the "market" model option. Linear regression is used
to estimate expected (normal) returns based on benchmark returns.  The model is trained with data from the
evaluation period and used to predict stock returns during the event window.
Alternatively, the "mean" model option may be used to set expected returns to the mean return of the stock
during the evaluation period.  An event window and an evaluation period must also be selected.  The evaluation 
window and the event window must be within dates for which data is available for the stock.  If the "market"
model is used, benchmark data must also be available for the evaluation and event windows.  

Two synchronized stock price and daily return charts are provided for easy identification of 
abnormal stock price movements using the Bokeh toolbar and hover tools.  The event study results 
are provided in the form of a chart showing abnormal returns (bar chart)
and cumulative abnormal returns (line chart) during the event period.  Cumulative abnormal 
returns prior to the event date that are outside the shaded area in the chart are indicative of 
potential insider trading.

The event study table provides t-statistics and p-values for the abnormal returns during
the event window.  Cumulative abnormal returns with p-values under 10% prior to the event date are
indicative of potential insider trading.  The level of confidence that a cummulative return is
abnormal is indicated by one, two or three asterisks for 90%, 95% and 99% confidence, respectively.

See a summary of the methodology used below.

The app is written in Python, Flask and Bokeh.  


# Event Studies in Finance

- Event studies use financial market data to measure the effects of an economic event on the value of a firm.
- Given rationality in the marketplace, the effects of an event can be observed over a short time period since they are reflected immediately in security prices. 
- The event study has many applications in finance research and has been applied to a variety of firm specific and economy wide events including studies of market efficiency and insider trading.
- In this presentation, we use event studies to evidence the legitimacy (or not) of vulnerability disclosures.

# Event Study Methodology

## Identification of the Event and its Date

#### Type of Event 
- Disclosure made public.

#### Date of the Event t=0
- Date in which the news about the disclosure is published.
- The event window expanded to 5 days before and 20 days after the event date to examine periods surrounding the event.

#### Selection of Sample
- Must be representative.
- Daily Adjusted Closing Stock Prices (Yahoo Finance).
- Compute daily actual returns as the percentage change of daily stock prices.

## Modelling the Return Generating Process

#### Abnormal Returns
	ARt = Rt – E(Rt|Xt) 
where ARt , Rt and E(Rt|Xt) are the abnormal, actual, and normal returns respectively for time period t.  Xt is the conditioning information for the normal return model.

#### Measure Normal Returns: expected returns if no event happened
- Constant Mean Model where Xt is constant
- Using the Market Model where Xt includes the market return
- Control portfolio: Xt is the return on a portfolio of similar firms

#### The Estimation Window
- Period prior to the event window, 150 days.

## Testing the Hypothesis

#### H0 – the null hypothesis
- The null hypothesis is that the abnormal returns are not significantly different from zero.

#### Estimate the benchmark model during the estimation period using linear regression
	𝑅𝑡=⍺+𝛽𝑅𝑚𝑡+ et	 where	E(et) = 0  and   e  ~  N ( 0, σ2e )
Where Rt and 𝑅𝑚𝑡 are  the period-t returns on the security and the market portfolio, respectively, and et is the time period t disturbance term for the security with and expectation of zero and variance sigma squared (σ2e).

#### Calculate the abnormal returns during the event period using the linear regression coefficients
	 𝐴𝑅𝑡=𝑅𝑡− ⍺ −𝛽𝑅𝑚𝑡	 ~  N ( 0, var(𝐴𝑅𝑡))

#### Calculate cumulative abnormal returns over time
	CAR [t-t1:t+t2] = ∑t=t-t1:t+t2 ARt

## Analyzing Abnormal Returns

#### Null Hypothesis
- The abnormal returns are not significantly different from zero and the event has no impact on the firm’s value.

#### T-statistic 
- The T-statistic is used to accept or reject the null hypothesis. 
- Assumes data points are independent and follow a normal distribution.
- Use one tailed test to test whether the abnormal return is significantly less than zero. Positive deviations from zero are not considered.
- The test is carried out with a 90% confidence level and a significance level of 0.10 with n-1 degrees of freedom, where n is the number of days in the estimation window. 

#### p-value 
- Measures the probability that the t-statistic will produce values at least as extreme as the t-score produced form the sample. 

The null hypothesis is rejected if the p-value is less than 0.10. 

# References

- Fama, Fisher, Jensen and Roll, The Adjustment of Stock Prices to New Information, 1969
- MacKinlay, Craig, Event Studies in Economics and Finance, 1997
- Lemiere, Jean-Baptiste, Event Study Package, Github, 2019
- National Evaluation Series, Event Study Analysis
- M.Lejdelin, P. Lindén, Insider trading on the Stockholm Stock Exchange, 2006 

