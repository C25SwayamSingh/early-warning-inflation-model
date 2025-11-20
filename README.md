# Early Warning Inflation Model

## Project Overview

This project explores whether alternative, real-time data sources can serve as early indicators of U.S. inflation. Using the Federal Reserve Bank of St. Louis’s FRED API for official CPI data and Google Trends for search interest in everyday costs (e.g., “rent,” “gas prices,” “food prices”), our goal is to build a data-driven model that detects inflation signals before official reports are released.

## Team Members

- [Swayam Singh] - [Data Cleaning, Model Development, Visualization] - [linkedin.com/in/swayam-singh-connect]
- [Name] - [Role/Contribution] - [Optional: LinkedIn/Email]


## Data Description

We combine two datasets:
FRED (St. Louis Fed) — Monthly U.S. CPI (series: CPIAUCSL). Pulled via fredapi.
Time period: 1948–present (your last run fetched through 2025-09).
Size: ~933 monthly observations (grows each month).
Google Trends (pytrends) — U.S. search interest for “rent”, “gas prices”, “food prices”.
Time period: rolling last 5 years (“today 5-y”), weekly values averaged to monthly.
Size: ~60 monthly observations per keyword (depends on the run date).

How the data is collected: 
FRED via HTTPS API using your local FRED_API_KEY.
Google Trends via pytrends with retries/backoff; results cached per keyword to avoid rate limits.

Preprocessing / cleaning
CPI: compute month-over-month (MoM) and year-over-year (YoY) percent changes; resample to month-end.
Trends: drop isPartial, resample weekly → monthly mean, cache to data/google_trends_cache/.
Merge: inner join on month-end dates; drop rows with any missing values.
No scaling is applied to saved data (only temporary normalization for plotting).
Outputs saved: merged_inflation_data.csv, correlation_results.csv, and results/correlation_analysis.png.
Merged dataset size
~60 monthly rows (intersection of CPI with the last 5-year Trends window) and 6 features (CPI level, MoM, YoY, and three trend columns), plus the date index.


### Key Variables

| Variable Name     | Description                                             | Data Type | Units/Format           | Notes                                            |
| :---------------- | :------------------------------------------------------ | :-------- | :--------------------- | :----------------------------------------------- |
| date              | Month-end timestamp (index)                             | datetime  | YYYY-MM-DD (month-end) | From resampling to monthly.                      |
| CPI               | CPI for All Urban Consumers (headline level)            | float     | Index (1982–84=100)    | FRED series CPIAUCSL.                          |
| CPI_MoM           | Month-over-month % change in CPI                        | float     | Percent                | 100 * pct_change(1); early NaNs dropped.       |
| CPI_YoY           | Year-over-year % change in CPI                          | float     | Percent                | 100 * pct_change(12); first 12 months dropped. |
| rent_trend        | Google search interest for “rent” (monthly mean)        | float     | 0–100 (Google Trends)  | US, “today 5-y”; weekly → monthly mean.          |
| gas_prices_trend  | Google search interest for “gas prices” (monthly mean)  | float     | 0–100 (Google Trends)  | Same settings as above.                          |
| food_prices_trend | Google search interest for “food prices” (monthly mean) | float     | 0–100 (Google Trends)  | Same settings as above.                          |


## Methodology

Data Cleaning & EDA: Merged CPI and Google Trends data by month; checked for missing values, outliers, and correlations.
Exploratory Analysis: Visualized search interest vs. CPI changes to assess relationships.
Modeling: Used linear regression and XGBoost to test predictive strength of trends for next-month CPI changes.
Evaluation: Measured model performance using correlation coefficients, RMSE, and SHAP feature importance.

Outline the approach taken to analyze the data, including:

- Exploratory data analysis techniques
- Statistical methods applied
- Machine learning models used (if applicable)
- Evaluation metrics


## Key Findings

Search trends for “gas prices” and “rent” show moderate leading correlations with monthly CPI changes.
Google Trends data can act as a short-term inflation signal, but strength varies by time period and keyword frequency.
The FRED CPI data provides stability, while Trends data adds timeliness and early-warning potential.

Summarize the main discoveries and insights from your analysis. Include:

- Important patterns or trends identified
- Unexpected results
- Answers to the initial research questions
- Visualizations of key findings (reference to files in the repository)


## Installation and Setup

Instructions for setting up the project environment:

```bash
# Example installation commands
pip install -r requirements.txt
```
**note:** A requriements.txt is often used to list all the packages (e.g., pandas, numpy, scikit-learn) that are needed to run the project. You can create this file by running `pip freeze > requirements.txt` in your terminal.

## Project Structure

pip install -r requirements.txt

├── data/               # Raw and processed data files  
├── notebooks/          # EDA and testing notebooks  
├── src/                # Source code (API integration, modeling)  
├── results/            # Visualizations and summary outputs  
├── requirements.txt    # Dependencies (pandas, pytrends, matplotlib, xgboost, shap)  
└── README.md           # This file  


Explain the organization of files and directories in your repository:

```
├── data/               # Raw and processed data files
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
├── results/            # Output files and visualizations
├── requirements.txt    # Required packages
└── README.md           # This file
```
**Note:** To get the above project structure (without needing to format it all manually), look up the [tree command](https://www.geeksforgeeks.org/tree-command-unixlinux/). You can use `tree` in the terminal in any directory to recursively list folders/files. I would suggest using `tree -L 1` (as shown above) to limit the depth of the tree to 1 level, as project codebases can get quite complex.

## Usage

# Run the main script
python main.py

# The script pulls CPI and Google Trends data, merges them, runs correlations,
# and generates plots saved in the /results folder.

Provide instructions on how to run your code and reproduce your results:

```python
# Example code snippet
from src.model import train_model
from src.visualization import visualize_results
model = train_model(data_path='data/processed/training_data.csv')
output = model.predict(new_data)
visualize_results(output)
```

## Future Work

Expand search keywords (e.g., “housing costs,” “grocery inflation”) to increase coverage.
Compare predictive accuracy against traditional macro models (e.g., ARIMA, VAR).
Incorporate additional alternative data sources like shipping or wage data.

Outline potential next steps or improvements for the project:

- Additional analyses that could be performed
- Features that could be added
- Ways to improve the current models or methods

## Acknowledgments

Data provided by Federal Reserve Bank of St. Louis (FRED) and Google Trends.
Built with Python 3.11, pandas, pytrends, scikit-learn, and matplotlib.




## Contact Information

Swayam Singh – swayam.singh@illinois.edu