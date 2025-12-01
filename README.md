# Early-Warning Inflation Model

This project builds an early-warning inflation model by combining official CPI data from the Federal Reserve Economic Data (FRED) API with Google Trends search data for inflation-related terms.

## Features

- **FRED API Integration**: Fetches monthly Consumer Price Index (CPI) data
- **Google Trends Analysis**: Collects search interest data for terms like "rent," "gas prices," "food prices," etc.
- **Data Merging**: Aligns and merges datasets by date
- **Correlation Analysis**: Identifies which search terms have the strongest relationship with CPI changes
- **Visualizations**: Creates comprehensive plots showing correlations and time series
- **Data Export**: Saves merged datasets and analysis results to CSV files

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get a FRED API Key:**
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up for a free account and get your API key

3. **Set the API key as an environment variable:**
   ```bash
   export FRED_API_KEY='your_key_here'
   ```
   
   Or on Windows:
   ```cmd
   set FRED_API_KEY=your_key_here
   ```

## Usage

Run the main script:

```bash
python main.py
```

The script will:
1. Fetch CPI data from FRED (default: from 2010 to present)
2. Fetch Google Trends data for specified search terms
3. Merge and clean the datasets
4. Calculate correlation coefficients
5. Generate visualizations
6. Save results to:
   - `merged_inflation_data.csv` - Merged dataset
   - `correlation_results.csv` - Correlation analysis results
   - `correlation_analysis.png` - Visualization plots

## Output Files

- **merged_inflation_data.csv**: Complete merged dataset with CPI and Google Trends data
- **correlation_results.csv**: Correlation coefficients between search terms and CPI metrics
- **correlation_analysis.png**: Visualizations including:
  - Time series comparison
  - Correlation heatmap
  - Scatter plots for top correlations

## Notes

- Google Trends has rate limits - if you encounter errors, wait a few minutes and try again
- The model analyzes correlations with both month-over-month (MoM) and year-over-year (YoY) CPI changes
- Search terms can be customized in the `main()` function

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

