 # GARCH Trading Model
 
 1. TOC
{:toc}

# Introduction:

In our Econometrics course we had to come up with a algorithmic trading strategy either using a pairs trading or GARCH approach.We opted for he GARCH approach using R code. 

 GARCH is a econometric method to estimate the behavior of a sequence of financial data we will refer to as time series data.

 

The assumption is that the value of a financial asset may have a strong relationship to the past. This relationship simply means that the price today is a function of its historical behavior alone.

 

However, we generally want to model processes that are known as stationary. In other words, data that does not have a defined pattern. Such processes tend not to have a visible trend in behavior and can be classified as noise. This is an ideal situation and is not always obtained in raw data.

 

As a result, we need to transform our data in several ways and then test whether the resultant process satisfies the required statistical properties. Such transformations aim to remove or quantify seasonal effects, long term trends and any information useful in the forecasting.

 

At best, the residual error (difference between forecast and actual) should be random and have no trend i.e stationary. That then means that our model would have captured all information that impacts the data.

 

For instance, if we let $P(n)$ to be the price today then

### Auto regressive model: $AR(n)$

$P(t)=P(t-1)+a(2)P(t-2)+...a(n)P(t-n)+e_n$

This means the current price depends on the past n prices

### Moving Average: $MA(m)$

The price depends on the past $m$ moving averages where $e_n$ represents the error or the residuals. Simply put it is a linear regression of today's price with yesterday's priced as input variables. Now the GARCH model goes further to model an ARMA type mode on the residuals.

 
## Rational

We have to choose the right lags for the current data in the following way.

 

1. Convert stock price data into return data by differencing

2. Test that data for stationarity in two ways

    A. Visualise the transformed data
    B. Plot an autocorelation and partial auto correlation graph to get a feel of the data.
    This stage enables us to see which lags are significant

3. Use uGarch program in R to determine an ideal model structure.

In addition, we used an iterating function to determine several models and compare their relative significance

4. Statistically test if the proposed relationship using the following

    Augmented Dickey Fuller Test - Tests if the proposed model fits the data
    Information criterion tests (AIC and BIC) - these determine of the model parameters explain the real data or not. Also, it quantifies the statistical significance.

5. Choose the model that best suits the data according to the AIC/BIC

 

By modeling the data we can extrapolate within a certain level of confidence to possible price tomorrow.

 

This leads us to the following strategy:

 

## TRADING STRATEGY:

    If the forecast price>Today price
        BUY
    If forecast price < Today price
        SELL
    Else do nothing

The method was too static so we needed a rolling model that re calibrates daily according to the data.

 

So the above processes were enclosed in a loop that reevaluated the position after the end of a trading day.
Implementation
# Results
# Conclusions 

# Possible improvements:
 
# References
