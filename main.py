"""
Early-Warning Inflation Model using Alternative Data Sources

This script builds an inflation prediction model by combining:
- Official CPI data from FRED (Federal Reserve Economic Data)
- Google Trends search data for inflation-related terms

The model analyzes correlations between search trends and actual CPI changes
to identify early warning signals of inflation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from fredapi import Fred
from pytrends.request import TrendReq
from pytrends import exceptions
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class InflationEarlyWarningModel:
    """
    A class to build and analyze an early-warning inflation model
    using FRED CPI data and Google Trends data.
    """
    
    def __init__(self, fred_api_key=None):
        """
        Initialize the model with API credentials.
        
        Args:
            fred_api_key: FRED API key. If None, will try to get from environment variable.
        """
        # Initialize FRED API client
        if fred_api_key is None:
            fred_api_key = os.getenv('FRED_API_KEY')
            if fred_api_key is None:
                raise ValueError(
                    "FRED API key required. Set FRED_API_KEY environment variable "
                    "or pass it as an argument. Get your key from: https://fred.stlouisfed.org/docs/api/api_key.html"
                )
        
        self.fred = Fred(api_key=fred_api_key)
        
        # Initialize Google Trends client with retry/backoff settings
        self.pytrends = TrendReq(
            hl='en-US',
            tz=360,
            retries=5,
            backoff_factor=60,
            timeout=(10, 30)
        )
        
        # Storage for data
        self.cpi_data = None
        self.trends_data = None
        self.merged_data = None
        
    def fetch_cpi_data(self, start_date='2010-01-01', series_id='CPIAUCSL'):
        """
        Fetch Consumer Price Index (CPI) data from FRED API.
        
        Args:
            start_date: Start date for data retrieval (YYYY-MM-DD format)
            series_id: FRED series ID for CPI (default: CPIAUCSL - CPI for All Urban Consumers)
        
        Returns:
            DataFrame with CPI data indexed by date
        """
        print(f"Fetching CPI data from FRED (Series: {series_id})...")
        
        try:
            # Fetch data from FRED
            cpi = self.fred.get_series(series_id, start=start_date)
            
            # Convert to DataFrame
            cpi_df = pd.DataFrame({
                'date': cpi.index,
                'CPI': cpi.values
            })
            
            # Calculate month-over-month and year-over-year changes
            cpi_df['CPI_MoM'] = cpi_df['CPI'].pct_change(1) * 100  # Monthly change in %
            cpi_df['CPI_YoY'] = cpi_df['CPI'].pct_change(12) * 100  # Year-over-year change in %
            
            # Set date as index
            cpi_df.set_index('date', inplace=True)
            
            # Resample to monthly (in case of daily data)
            cpi_df = cpi_df.resample('M').last()
            
            # Remove NaN values from the beginning
            cpi_df = cpi_df.dropna()
            
            self.cpi_data = cpi_df
            print(f"✓ Successfully fetched {len(cpi_df)} months of CPI data")
            print(f"  Date range: {cpi_df.index.min()} to {cpi_df.index.max()}")
            
            return cpi_df
            
        except Exception as e:
            print(f"Error fetching CPI data: {e}")
            raise
    
    def fetch_google_trends(self, keywords, start_date='2010-01-01', end_date=None, fetch_one_by_one=True):
        """
        Fetch Google Trends data for specified keywords.

        Args:
            keywords: List of search terms to analyze
            start_date: Start date for data retrieval (YYYY-MM-DD format)
            end_date: End date for data retrieval (YYYY-MM-DD format). If None, uses today.
            fetch_one_by_one: If True, fetch keywords individually with delays to avoid rate limits.

        Returns:
            DataFrame with Google Trends data indexed by date.
        """
        print(f"\nFetching Google Trends data for: {', '.join(keywords)}...")

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        timeframe_str = "today 5-y"

        cache_dir = Path("data/google_trends_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _cache_path(keyword: str) -> Path:
            safe = keyword.replace(" ", "_").lower()
            return cache_dir / f"{safe}_{start_dt.strftime('%Y%m')}_{end_dt.strftime('%Y%m')}.csv"

        def _load_from_cache(keyword: str) -> Optional[pd.DataFrame]:
            path = _cache_path(keyword)
            if path.exists():
                print(f"  Using cached data for '{keyword}'")
                df = pd.read_csv(path, parse_dates=['date'], index_col='date')
                return df
            return None

        def _fetch_keyword(keyword: str) -> pd.DataFrame:
            cached_df = _load_from_cache(keyword)
            if cached_df is not None:
                return cached_df

            safe_col = keyword.replace(" ", "_").lower() + "_trend"
            max_attempts = 5
            wait_seconds = 45

            for attempt in range(1, max_attempts + 1):
                try:
                    print(f"  Fetching '{keyword}' (attempt {attempt}/{max_attempts})...")
                    if attempt > 1:
                        print(f"    Waiting {int(wait_seconds)} seconds before retry...")
                        time.sleep(wait_seconds)
                        wait_seconds *= 1.5
                    else:
                        time.sleep(3)

                    self.pytrends.build_payload(
                        [keyword],
                        cat=0,
                        timeframe=timeframe_str,
                        geo='US',
                        gprop=''
                    )

                    kw_df = self.pytrends.interest_over_time()

                    if kw_df.empty:
                        raise RuntimeError("empty data returned")

                    if 'isPartial' in kw_df.columns:
                        kw_df = kw_df.drop(columns=['isPartial'])

                    kw_df = kw_df.rename(columns={kw_df.columns[0]: safe_col})
                    kw_df = kw_df[[safe_col]]

                    cache_path = _cache_path(keyword)
                    kw_df.reset_index().to_csv(cache_path, index=False)
                    return kw_df

                except exceptions.TooManyRequestsError as e:
                    print(f"    Rate limit hit: {e}")
                    continue
                except Exception as e:
                    print(f"    Error fetching '{keyword}': {e}")
                    continue

            print(f"    ❌ Failed to fetch '{keyword}' after multiple attempts.")
            return pd.DataFrame()

        trend_frames = []

        if fetch_one_by_one:
            for idx, keyword in enumerate(keywords, start=1):
                print(f"\n  [{idx}/{len(keywords)}] Processing keyword: '{keyword}'")
                df_kw = _fetch_keyword(keyword)
                if not df_kw.empty:
                    trend_frames.append(df_kw)
                else:
                    print(f"    ⚠️  Skipping '{keyword}' due to unavailable data.")
                time.sleep(5)
        else:
            print("  Fetching all keywords in a single request...")
            try:
                time.sleep(3)
                self.pytrends.build_payload(
                    keywords,
                    cat=0,
                    timeframe=timeframe_str,
                    geo='US',
                    gprop=''
                )
                combined = self.pytrends.interest_over_time()
                if combined.empty:
                    raise RuntimeError("empty data returned")
                if 'isPartial' in combined.columns:
                    combined = combined.drop(columns=['isPartial'])
                combined.columns = [f"{col.replace(' ', '_').lower()}_trend" for col in combined.columns]
                trend_frames.append(combined)
            except Exception as e:
                print(f"  ❌ Failed to fetch combined Google Trends data: {e}")

        if not trend_frames:
            raise RuntimeError("❌ No Google Trends data fetched. Try again later or with fewer keywords.")

        trends_df = pd.concat(trend_frames, axis=1)
        trends_df = trends_df.sort_index()
        trends_monthly = trends_df.resample('M').mean()
        trends_monthly = trends_monthly.dropna(how='all')

        self.trends_data = trends_monthly

        print(f"\n✓ Successfully fetched Google Trends data for {len(self.trends_data.columns)} keyword(s)")
        print(f"  Date range: {self.trends_data.index.min()} to {self.trends_data.index.max()}")
        return self.trends_data
    
    def merge_datasets(self):
        """
        Merge CPI and Google Trends datasets by date.
        
        Returns:
            Merged DataFrame with both CPI and trends data
        """
        if self.cpi_data is None or self.trends_data is None:
            raise ValueError("Both CPI and Trends data must be fetched before merging")
        
        print("\nMerging CPI and Google Trends datasets...")
        
        # Merge on date index
        merged = pd.merge(
            self.cpi_data,
            self.trends_data,
            left_index=True,
            right_index=True,
            how='inner'
        )
        
        # Remove any rows with NaN values
        merged = merged.dropna()
        
        self.merged_data = merged
        print(f"✓ Successfully merged datasets")
        print(f"  Final dataset: {len(merged)} months of data")
        print(f"  Date range: {merged.index.min()} to {merged.index.max()}")
        
        return merged
    
    def calculate_correlations(self):
        """
        Calculate correlation coefficients between Google Trends and CPI changes.
        
        Returns:
            DataFrame with correlation results
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before calculating correlations")
        
        print("\nCalculating correlations...")
        
        # Get trend columns
        trend_cols = [col for col in self.merged_data.columns if '_trend' in col]
        
        # Calculate correlations with CPI changes
        correlations = {}
        
        for trend_col in trend_cols:
            # Correlation with month-over-month CPI change
            corr_mom = self.merged_data[trend_col].corr(self.merged_data['CPI_MoM'])
            
            # Correlation with year-over-year CPI change
            corr_yoy = self.merged_data[trend_col].corr(self.merged_data['CPI_YoY'])
            
            # Correlation with CPI level (with lag - trends might predict future CPI)
            # Try different lags (0, 1, 2, 3 months)
            max_corr = 0
            best_lag = 0
            for lag in range(4):
                shifted_cpi = self.merged_data['CPI'].shift(-lag)
                corr = self.merged_data[trend_col].corr(shifted_cpi)
                if abs(corr) > abs(max_corr):
                    max_corr = corr
                    best_lag = lag
            
            correlations[trend_col] = {
                'CPI_MoM': corr_mom,
                'CPI_YoY': corr_yoy,
                'CPI_Level_Best_Lag': max_corr,
                'Best_Lag_Months': best_lag
            }
        
        # Convert to DataFrame for easier viewing
        corr_df = pd.DataFrame(correlations).T
        corr_df = corr_df.sort_values('CPI_YoY', key=abs, ascending=False)
        
        print("\nCorrelation Results:")
        print(corr_df.round(3))
        
        return corr_df
    
    def visualize_correlations(self, save_path='correlation_analysis.png'):
        """
        Create visualizations of correlations and time series.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before visualization")
        
        print(f"\nCreating visualizations...")
        
        # Get trend columns
        trend_cols = [col for col in self.merged_data.columns if '_trend' in col]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. Time series plot
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Plot CPI YoY change
        ax1.plot(self.merged_data.index, self.merged_data['CPI_YoY'], 
                'b-', linewidth=2, label='CPI YoY % Change')
        ax1.set_ylabel('CPI Year-over-Year % Change', color='b', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('CPI vs Google Trends Over Time', fontsize=12, fontweight='bold')
        
        # Plot normalized trends (for comparison)
        for trend_col in trend_cols:
            normalized = (self.merged_data[trend_col] - self.merged_data[trend_col].min()) / \
                        (self.merged_data[trend_col].max() - self.merged_data[trend_col].min()) * 10
            ax1_twin.plot(self.merged_data.index, normalized, 
                         alpha=0.6, label=trend_col.replace('_trend', ''))
        
        ax1_twin.set_ylabel('Normalized Search Interest (0-10 scale)', color='g', fontsize=10)
        ax1_twin.tick_params(axis='y', labelcolor='g')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. Correlation heatmap
        ax2 = axes[1]
        corr_data = self.merged_data[['CPI_MoM', 'CPI_YoY'] + trend_cols].corr()
        corr_subset = corr_data.loc[['CPI_MoM', 'CPI_YoY'], trend_cols]
        
        sns.heatmap(corr_subset, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   center=0, vmin=-1, vmax=1, ax=ax2, cbar_kws={'label': 'Correlation'})
        ax2.set_title('Correlation Matrix: CPI Changes vs Google Trends', 
                     fontsize=12, fontweight='bold')
        ax2.set_ylabel('CPI Metrics')
        ax2.set_xlabel('Google Trends Search Terms')
        
        # 3. Scatter plots for top correlations
        ax3 = axes[2]
        
        # Find the trend with highest absolute correlation with CPI_YoY
        corr_with_yoy = self.merged_data[trend_cols].corrwith(self.merged_data['CPI_YoY'])
        best_trend = corr_with_yoy.abs().idxmax()
        
        ax3.scatter(self.merged_data[best_trend], self.merged_data['CPI_YoY'], 
                   alpha=0.6, s=50)
        ax3.set_xlabel(f'{best_trend.replace("_trend", "")} Search Interest', fontsize=10)
        ax3.set_ylabel('CPI Year-over-Year % Change', fontsize=10)
        ax3.set_title(f'Scatter Plot: {best_trend.replace("_trend", "")} vs CPI YoY\n'
                     f'Correlation: {corr_with_yoy[best_trend]:.3f}', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.merged_data[best_trend], self.merged_data['CPI_YoY'], 1)
        p = np.poly1d(z)
        ax3.plot(self.merged_data[best_trend], p(self.merged_data[best_trend]), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_data(self, filepath='merged_inflation_data.csv'):
        """
        Save the merged dataset to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        if self.merged_data is None:
            raise ValueError("No merged data to save")
        
        self.merged_data.to_csv(filepath)
        print(f"✓ Merged dataset saved to: {filepath}")


def main():
    """
    Main function to run the inflation early-warning model.
    """
    print("=" * 60)
    print("Early-Warning Inflation Model")
    print("Using FRED CPI Data + Google Trends")
    print("=" * 60)
    
    # Initialize model
    # Note: You need to set FRED_API_KEY environment variable
    # Get your key from: https://fred.stlouisfed.org/docs/api/api_key.html
    try:
        model = InflationEarlyWarningModel()
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo fix this:")
        print("1. Get a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Set it as an environment variable:")
        print("   export FRED_API_KEY='your_key_here'")
        return
    
    # Define search terms to analyze
    # Reduced to 3 most important terms to avoid rate limits
    # You can add more later: 'inflation', 'grocery prices'
    search_terms = ['rent', 'gas prices', 'food prices']
    
    try:
        # Fetch CPI data (last 10+ years)
        model.fetch_cpi_data(start_date='2010-01-01')
        
        # Try to fetch Google Trends data
        # Note: Google Trends has rate limits, so this may fail
        trends_success = False
        try:
            print("\nAttempting to fetch Google Trends data...")
            model.fetch_google_trends(
                keywords=search_terms,
                start_date='2010-01-01'
            )
            trends_success = True
        except Exception as trends_error:
            print(f"\n⚠️  Google Trends unavailable: {trends_error}")
            print("\n" + "=" * 60)
            print("OPTION 1: Wait 10-15 minutes and try again")
            print("OPTION 2: Continue with CPI-only analysis (see below)")
            print("=" * 60)
            
            # Ask user if they want to continue with sample data or just CPI
            print("\nGenerating CPI-only analysis and saving data...")
            
            # Save CPI data
            model.cpi_data.to_csv('cpi_data_only.csv')
            print(f"✓ CPI data saved to: cpi_data_only.csv")
            
            # Create a simple CPI visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(model.cpi_data.index, model.cpi_data['CPI_YoY'], 
                   'b-', linewidth=2, label='CPI Year-over-Year % Change')
            ax.set_ylabel('CPI Year-over-Year % Change', fontsize=12)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_title('Consumer Price Index (CPI) - Year-over-Year Change\n(Google Trends data unavailable due to rate limits)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            plt.savefig('cpi_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ CPI visualization saved to: cpi_analysis.png")
            plt.close()
            
            print("\n" + "=" * 60)
            print("CPI-Only Analysis Complete!")
            print("=" * 60)
            print(f"\n  • CPI data saved to 'cpi_data_only.csv'")
            print(f"  • Visualization saved to 'cpi_analysis.png'")
            print(f"\n  To get full analysis with Google Trends:")
            print(f"  • Wait 10-15 minutes for rate limit to reset")
            print(f"  • Then run the script again: python main.py")
            return
        
        # If Google Trends succeeded, continue with full analysis
        if trends_success:
            # Merge datasets
            model.merge_datasets()
            
            # Calculate correlations
            correlations = model.calculate_correlations()
            
            # Create visualizations
            model.visualize_correlations(save_path='correlation_analysis.png')
            
            # Save merged dataset
            model.save_data(filepath='merged_inflation_data.csv')
            
            # Save correlation results
            correlations.to_csv('correlation_results.csv')
            print(f"✓ Correlation results saved to: correlation_results.csv")
            
            print("\n" + "=" * 60)
            print("Analysis Complete!")
            print("=" * 60)
            print("\nKey Findings:")
            print(f"  • Strongest predictor (by absolute correlation with CPI YoY):")
            best_predictor = correlations['CPI_YoY'].abs().idxmax()
            best_corr = correlations.loc[best_predictor, 'CPI_YoY']
            print(f"    {best_predictor.replace('_trend', '')}: {best_corr:.3f}")
            print(f"\n  • All correlation results saved to 'correlation_results.csv'")
            print(f"  • Visualizations saved to 'correlation_analysis.png'")
            print(f"  • Merged data saved to 'merged_inflation_data.csv'")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()