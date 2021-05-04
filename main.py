# For division
from __future__ import division

# Standard
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
from scipy import stats

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# For reading stock data from yahoo
from pandas_datareader import data

# For time stamps
import datetime

# Tech stocks used far analysis
tech_list = ['AAPL','GOOG','MSFT','AMZN']

# Set up Start and End times for grabbing data (One year ago to now)
end = datetime.datetime.now()
start = datetime.datetime(end.year - 1, end.month, end.day)

# For loop for grabbing finance data and entering it into a DataFrame
for stock in tech_list:
    # Set DataFrame as stock ticker
    globals()[stock] = data.DataReader(stock,'yahoo',start,end)

# Summary statistics and general info printed
print(AAPL.describe())
print(AAPL.info())
print(AAPL.head())

# Plot volume  of stock being traded each day for the past 5 years
# and  closing price of AAPL stocks
AAPL['Volume'].plot(legend=True, figsize=(10,4))
plt.show()
AAPL['Adj Close'].plot(legend=True, figsize=(10,4))
plt.show()

# Use pandas to calculate moving average of stock
# Setup plot for several moving averages
ma_day = [10,50,200]
for ma in ma_day:
    column_name = 'MA for {} days'.format(str(ma))
    AAPL[column_name] = pd.DataFrame.rolling(AAPL['Adj Close'],ma).mean()

# Plot all moving averages
AAPL[['Adj Close', 'MA for 10 days','MA for 50 days','MA for 200 days']].plot(subplots=False,figsize=(10,4))
plt.show()

# After baseline analysis of AAPL stock, analyze risk of stock on daily basis
# Use DataFrame.percent_change and then plot
AAPL["Daily Return"] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(12,4),legend=True,linestyle='--',marker='o')
plt.show()

# Use seaborn to create histogram of newly created 'Daily Return' data
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
AAPL['Daily Return'].hist() # alternate option to create histogram
plt.show()

# Analyze the returns of all stocks in tech stock list by building DataFrame
adjclose_df = data.DataReader(['AAPL','GOOG','MSFT','AMZN'],'yahoo',start,end)['Adj Close']
print(adjclose_df.head())

# Make a new DataFrame grabbing the daily returns for closing prices of all stocks
tech_returns = adjclose_df.pct_change()

# Create jointplot to compare daily returns between Google and Microsoft
sns.jointplot(x='GOOG',y='MSFT',data=tech_returns,kind='scatter',legend=True)
plt.show()

# Call pairplot on adjClose DataFrame to view all comparisons
sns.pairplot(adjclose_df.dropna())
plt.show()

# Use sns.pairGrid() with parameters to obtain more descriptive stats

# Set up new figure and call pairGrid() onto the DataFrame
returns_figures = sns.PairGrid(tech_returns.dropna())
# Specify what upper triangle will look like with map_upper
returns_figures.map_upper(plt.scatter, color='purple')
# Define lower triangle of figure, including plot type (kde) and color
returns_figures.map_lower(sns.kdeplot,cmap='cool_d')
# Define the diagonal figures as a series of histogram plots of the daily return
returns_figures.map_diag(plt.hist,bins=30)
plt.show()

# Do same analysis on closing prices
returns_figures = sns.PairGrid(adjclose_df)
returns_figures.map_upper(plt.scatter, color='purple')
returns_figures.map_lower(sns.kdeplot,cmap='cool_d')
returns_figures.map_diag(plt.hist,bins=30)
plt.show()

# Set up correlation plot for daily returns using seaborn to see which
# stock indexes had similar growth correlation
corr_plot = tech_returns.corr()
sns.heatmap(corr_plot.dropna(),annot=True)
plt.show()

########################################################################
# Risk Analysis
# Do basic risk analysis by comparing expected return with stdv of the daily returns

# Define new DataFrame with dropna() for null values
new_returns = tech_returns.dropna()
area = np.pi * 20
plt.scatter(new_returns.mean(),new_returns.std(),alpha=0.5,s=area)
plt.ylim([0.016,0.0275])
plt.xlim([0.001,0.004])
plt.xlabel('Expected Returns')
plt.ylabel('Risk')

for label, x, y in zip(new_returns.columns, new_returns.mean(), new_returns.std()):
    plt.annotate(label, xy= (x,y), xytext = (50,50),
                 textcoords='offset points', ha = 'right', va = 'bottom',
                 arrowprops=dict(arrowstyle = '-', color='red',connectionstyle = 'arc3,rad=-0.3'))

plt.show()

########################################################################
# Value at Risk "Bootstrap" Method
# Start by recreating histogram of AAPL stock for visual
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
# Use quantile to get 95% confidence interval of all stocks
print(new_returns['AAPL'].quantile(0.05))
# Output = -0.03320439878572928
# With 95% confidence, worst daily loss will not exceed 3.32%
print(new_returns['AMZN'].quantile(0.05))
# Output = -0.029947223838873865
# With 95% confidence, worst daily loss will not exceed 2.99%
print(new_returns['GOOG'].quantile(0.05))
# Output = -0.028556537120439873
# With 95% confidence, worst daily loss will not exceed 2.86%
print(new_returns['MSFT'].quantile(0.05))
# Output = -0.027551462832725956
# With 95% confidence, worst daily loss will not exceed 2.76%

# Value at Risk "Monte Carlo" Method on AAPL stock

# Set up time horizon for GBM equation
days = 365

# Set up delta for GBM equation
dt = 1/days

# Grab "drift" for GBM equation (daily average return)
mu = new_returns.mean()['AAPL']

# Grab volatility of stock using std() of average return
sigma = new_returns.std()['AAPL']

# Create function that takes in starting price and number of days,
# and uses mu and sigma from earlier calulations
def monte_carlo_stock(start_price,days,mu,sigma):
    # This function takes in starting stock price, days of simulation,
    # mu, sigma, and returns simulated price array

    # Define price array
    price = np.zeros(days)
    price[0] = start_price

    # Shock and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)

    # Run price array for number of days
    for x in range(1,days):

        # Calculate shock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate drift
        drift[x] = mu * dt
        # Calculate price
        price[x] = price[x-1] + (price[x-1]) * (drift[x] + shock[x])

    return price

start_price = 72.292503

for run in range(100):
    plt.plot(monte_carlo_stock(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for AAPL')
plt.show()

#####################################################################
# Generate histogram of end results for AAPL for a larger run size

# Set large number of runs
runs = 10000

# Create empty matrix the hold the end price data
simulations = np.zeros(runs)

# Suppress output by limiting display to only 0-5 points
np.set_printoptions(threshold=5)

for run in range(runs):
    # Set the simulation data point as the last stock price for that run
    simulations[run] = monte_carlo_stock(start_price,days,mu,sigma)[days-1]

# Plot histogram of arrays
# Define q as the 1% emperical value
q = np.percentile(simulations, 1)

# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold')
plt.show()

# Looked at the 1% empirical quantile of the final price distribution
# to estimate the Value at Risk for the Apple stock, which looks to be
# $13.25 for every investment of 569.85 (the price of one initial google stock).

