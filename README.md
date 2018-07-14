# Deep learning LSTM prediction model for stock returns

Python implementation for "Deep learning with long short-term memory networks for financial market predictions" working paper by Thomas Fischer and Christopher Krauss.
https://www.econstor.eu/bitstream/10419/157808/1/886576210.pdf

Simply collect and load three years of data for the S&P 500 constituents. 
You can do this easily using https://github.com/nicolasvianavega/stock_price_time_series.

The output consists of:

* Trained Keras model (json)
* Model weights (h5)
* Pandas data frame consisting of normalized stock returns and up/down probabilities for each stock at any given day for a test dataset

## Example

```python
# Collect S&P 500 Constituents data using stock_price_time_series
symbol_list = ['AAPL','GOOGL','TSLA']
since = '2015-01-01'
until = '2018-01-05'
value = 'adj_close' #recommended

results = data_collect(symbol_list, since, until, value)
results
```

| date       | AAPL     | GOOGL     | TSLA     |
|------------|----------|-----------|----------|
| 2015-01-02 | 170.9074 | 1073.2100 | 320.5300 |
| 2015-01-03 | 170.8776 | 1091.5200 | 317.2500 |
|	 ... | 	    ...	|       ... | 	   ... |
| 2018-01-05 | 173.6259 | 1110.2900 | 316.5800 |

After collecting, just load the data into the script


