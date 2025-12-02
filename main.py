"""
Early-Warning Inflation Model using Alternative Data Sources

This script builds an inflation prediction model by combining:
- Official CPI data from FRED (Federal Reserve Economic Data)
- Google Trends search data for inflation-related terms

The model analyzes correlations between search trends and actual CPI changes
to identify early warning signals of inflation.
"""

import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from fredapi import Fred
from pytrends.request import TrendReq
from pytrends import exceptions
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pycaret.regression import setup, compare_models, pull, save_model


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
        timeframe_str = f'{start_dt.strftime("%Y-%m-%d")} {end_dt.strftime("%Y-%m-%d")}'

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

    def build_food_index(self, food_terms: List[str], drop_components: bool = True):
        """
        Build a composite food index from provided food-related search terms.

        Args:
            food_terms: List of strings representing food-related search terms.
            drop_components: If True, drop the individual component columns once the index is built.
        """
        if self.trends_data is None:
            raise ValueError("Trends data must be available before building a food index")

        component_columns = []
        for term in food_terms:
            col_name = f"{term.replace(' ', '_').lower()}_trend"
            if col_name in self.trends_data.columns:
                component_columns.append(col_name)

        if not component_columns:
            print("⚠️  Warning: Could not build 'food_index_trend' (no food components present)")
            return

        standardized_series = []
        for col in component_columns:
            series = self.trends_data[col]
            std = series.std()
            if std is None or np.isnan(std) or std == 0:
                continue
            standardized_series.append((series - series.mean()) / std)

        if not standardized_series:
            print("⚠️  Warning: Could not build 'food_index_trend' (food components lacked variance)")
            return

        composite = pd.concat(standardized_series, axis=1).mean(axis=1).rename('food_index_trend')
        self.trends_data['food_index_trend'] = composite

        if drop_components:
            self.trends_data.drop(columns=component_columns, inplace=True, errors='ignore')

        print(f"Built 'food_index_trend' from {len(standardized_series)} components")
    
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
    
    def build_feature_matrix(self, horizon_months: int = 3) -> pd.DataFrame:
        """
        Add extra columns for modeling:
        - future CPI targets horizon_months ahead
        - lags of CPI_MoM and CPI_YoY
        - lead versions of each *_trend column (so trends can act as early indicators)
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before building feature matrix")

        df = self.merged_data.copy()

        # Future CPI targets (what we eventually want to predict)
        df[f"CPI_MoM_future_{horizon_months}m"] = df["CPI_MoM"].shift(-horizon_months)
        df[f"CPI_YoY_future_{horizon_months}m"] = df["CPI_YoY"].shift(-horizon_months)

        # Lags of CPI_MoM and CPI_YoY
        for k in [1, 2, 3, 6, 12]:
            df[f"CPI_MoM_lag{k}"] = df["CPI_MoM"].shift(k)
            df[f"CPI_YoY_lag{k}"] = df["CPI_YoY"].shift(k)

        # Leads of each trend term: trend at t, t+1, t+2, t+3
        trend_cols = [c for c in df.columns if c.endswith("_trend")]
        for col in trend_cols:
            base_name = col.replace("_trend", "")
            for k in [0, 1, 2, 3]:
                df[f"{base_name}_trend_lead{k}"] = df[col].shift(-k)

        # Drop rows without future CPI targets (edge at end of sample)
        df = df.dropna(subset=[f"CPI_MoM_future_{horizon_months}m",
                               f"CPI_YoY_future_{horizon_months}m"])

        self.merged_data = df
        print(f"\n✓ Added derived CPI and trend features (horizon = {horizon_months} months)")
        print(f"  Rows available for modeling after dropping edge NaNs: {len(df)}")

        return df

    def build_pca_dataset(
        self,
        target_col: str,
        variance_threshold: float = 0.95,
        save_explained_path: str = 'results/pca_explained_variance.csv'
    ) -> pd.DataFrame:
        """
        Build a PCA-transformed dataset for modeling.

        Steps:
        - Take all numeric feature columns (lags/leads/trends/etc.) except the target
          and any 'future_' columns (to avoid leaking future info).
        - Standardize features (mean 0, std 1).
        - Run PCA to keep enough components to explain `variance_threshold`
          of the variance (e.g., 0.95 = 95%).
        - Save an explained-variance table and return a DataFrame with PC1..PCk + target.
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged and feature matrix built before PCA")

        df = self.merged_data.copy()

        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in merged_data. "
                f"Did you call build_feature_matrix() so that '{target_col}' exists?"
            )

        # Keep only rows where target is known
        df = df.dropna(subset=[target_col])

        # Candidate feature columns: numeric, not the target, and not "future_" (to avoid leakage)
        feature_cols = [
            c for c in df.columns
            if c != target_col
            and df[c].dtype != 'O'
            and 'future_' not in c
        ]

        # Drop any rows with NaNs in the features
        df_model = df.dropna(subset=feature_cols)
        X = df_model[feature_cols].values
        y = df_model[target_col].values

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA to explain given fraction of variance
        pca = PCA(n_components=variance_threshold)
        X_pca = pca.fit_transform(X_scaled)

        # Save explained variance table
        explained = pd.DataFrame({
            "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_)
        })
        results_path = Path(save_explained_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        explained.to_csv(results_path, index=False)

        print(f"\n✓ PCA complete for target '{target_col}'")
        print(f"  Original numeric features: {len(feature_cols)}")
        print(f"  PCA components kept:       {X_pca.shape[1]}")
        print(f"  Explained-variance table saved to: {results_path}")

        # Build DataFrame with PCs + target (index preserved for time alignment)
        pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, index=df_model.index, columns=pc_cols)
        df_pca[target_col] = y

        return df_pca
    
    def calculate_correlations(self):
        """
        Calculate correlation coefficients between Google Trends and CPI changes.
        
        Returns:
            DataFrame with correlation results
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before calculating correlations")
        
        print("\nCalculating correlations...")
        
        # Get trend columns (restricted to key series)
        trend_cols = [
            col for col in ["rent_trend", "gas_prices_trend", "food_index_trend"]
            if col in self.merged_data.columns
        ]
        
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
    
    def visualize_correlations(self, save_path='results/fig_corr_heatmap_scatter.png'):
        """
        Produce correlation heatmap (CPI changes vs trend columns) and best-pair scatter plot.
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before visualization")

        trend_cols = [col for col in ["rent_trend", "gas_prices_trend", "food_index_trend"] if col in self.merged_data.columns]
        if not trend_cols:
            print("⚠️  Warning: No trend columns available for visualization")
            return

        results_path = Path(save_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        ax_hm = axes[0]
        corr_data = self.merged_data[['CPI_MoM', 'CPI_YoY'] + trend_cols].corr()
        corr_subset = corr_data.loc[['CPI_MoM', 'CPI_YoY'], trend_cols]

        sns.heatmap(
            corr_subset,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax_hm,
            cbar_kws={'label': 'Correlation'}
        )
        ax_hm.set_title('Correlation: CPI Changes vs Google Trends', fontsize=12, fontweight='bold')
        ax_hm.set_ylabel('CPI Metrics')
        ax_hm.set_xlabel('Google Trends Search Terms')

        ax_sc = axes[1]
        corr_with_yoy = self.merged_data[trend_cols].corrwith(self.merged_data['CPI_YoY'])
        best_trend = corr_with_yoy.abs().idxmax()

        ax_sc.scatter(
            self.merged_data[best_trend],
            self.merged_data['CPI_YoY'],
            alpha=0.6,
            s=50
        )
        ax_sc.set_xlabel(f'{best_trend.replace("_trend", "").replace("_", " ")} (search index, rel. units)')
        ax_sc.set_ylabel('CPI YoY %')
        ax_sc.set_title(
            f'Scatter: {best_trend.replace("_trend", "").replace("_", " ")} vs CPI YoY  |  r = {corr_with_yoy[best_trend]:.3f}',
            fontsize=12,
            fontweight='bold'
        )

        z = np.polyfit(self.merged_data[best_trend], self.merged_data['CPI_YoY'], 1)
        p = np.poly1d(z)
        xs = self.merged_data[best_trend]
        ax_sc.plot(xs, p(xs), "r--", alpha=0.8, linewidth=2)
        ax_sc.grid(True, alpha=0.3)

        fig.tight_layout()
        plt.savefig(results_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {results_path}")
        plt.close(fig)
    
    def plot_small_multiples(self, save_path='results/fig_cpi_vs_trends_small_multiples.png'):
        """
        Create small multiples plot: CPI YoY % vs each trend series in separate panels.
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before plotting")

        trend_cols = [col for col in ["rent_trend", "gas_prices_trend", "food_index_trend"] if col in self.merged_data.columns]
        if not trend_cols:
            print("⚠️  Warning: No trend columns available for small multiples")
            return

        results_path = Path(save_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            nrows=len(trend_cols), ncols=1,
            figsize=(12, 3.5 * len(trend_cols)),
            sharex=True, constrained_layout=True
        )
        if len(trend_cols) == 1:
            axes = [axes]

        cpi_series = self.merged_data['CPI_YoY']
        target_y = 8.5
        y_min = min(cpi_series.min(), target_y - 1.0)
        y_max = max(cpi_series.max(), target_y + 1.0)
        y_lower = y_min - 0.3
        y_upper = y_max + 0.3

        cpi_handle = None

        for ax, col in zip(axes, trend_cols):
            cpi_line, = ax.plot(
                self.merged_data.index, self.merged_data['CPI_YoY'],
                linewidth=2, color='tab:blue'
            )
            if cpi_handle is None:
                cpi_handle = cpi_line
            ax.set_ylabel("CPI YoY %")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(y_lower, y_upper)

            s = self.merged_data[col]
            norm = (s - s.min()) / (s.max() - s.min() + 1e-9) * 10.0

            ax2 = ax.twinx()
            kw_label = col.replace('_trend', '').replace('_', ' ')
            kw_line, = ax2.plot(
                self.merged_data.index, norm,
                linewidth=1.8, linestyle='--', alpha=0.9, color='tab:gray'
            )
            ax2.set_ylabel("Search index (0–10)")
            ax2.legend(
                [kw_line],
                [kw_label],
                loc='upper right',
                frameon=False,
                handlelength=2.6
            )

        axes[-1].set_xlabel("Date")
        fig.suptitle("CPI vs Google Trends (Small Multiples)", y=0.97)

        if cpi_handle:
            fig.legend(
                [cpi_handle], ["CPI YoY %"],
                loc='upper left',
                bbox_to_anchor=(0.01, 0.995),
                frameon=False
            )

        fig.savefig(results_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {results_path}")
        plt.close(fig)


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

    def lead_lag_correlations(
        self,
        ks=(0, 1, 2, 3),
        targets=('CPI_MoM', 'CPI_YoY'),
        save_path='results/lead_corr.csv'
    ):
        """
        Compute correlations r(X_t, Y_{t+k}) for each trend column and lead k.
        Saves the detailed results to a CSV and prints the best lead per term.
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before computing lead/lag correlations")

        results_path = Path(save_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.merged_data.copy()
        trend_cols = [
            col for col in ["rent_trend", "gas_prices_trend", "food_index_trend"]
            if col in df.columns
        ]
        if not trend_cols:
            print("⚠️  Warning: No trend columns available for lead/lag correlations")
            return pd.DataFrame()

        rows = []
        for target in targets:
            if target not in df.columns:
                print(f"⚠️  Warning: Target '{target}' not found in merged data; skipping.")
                continue
            for col in trend_cols:
                x = df[col]
                for k in ks:
                    y = df[target].shift(-k)  # compare X_t with Y_{t+k}
                    aligned = pd.concat([x, y], axis=1).dropna()
                    if aligned.empty:
                        r = np.nan
                    else:
                        r = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    rows.append({'target': target, 'term': col, 'k': k, 'r': r})

        out = pd.DataFrame(rows)
        out.to_csv(results_path, index=False)

        if out.empty:
            print("⚠️  Warning: No valid lead/lag correlations computed.")
            return out

        best = (
            out.dropna(subset=['r'])
               .assign(abs_r=lambda d: d['r'].abs())
               .sort_values(['target', 'term', 'abs_r'], ascending=[True, True, False])
               .groupby(['target', 'term'], as_index=False)
               .first()[['target', 'term', 'k', 'r']]
        )
        print("\nBest lead per term:")
        print(best.to_string(index=False))
        print(f"✓ Saved lead correlations to {results_path}")
        return out

    def nowcast_yoy_custom(
        self,
        lag_map,
        test_start='2019-01-01',
        save_path='results/nowcast_custom.csv'
    ):
        """
        One-step-ahead CPI YoY nowcast using specific lags per term.
        Example lag_map: {'rent_trend': 3, 'gas_prices_trend': 0}
        """
        if self.merged_data is None:
            raise ValueError("Data must be merged before nowcasting")

        results_path = Path(save_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.merged_data.copy()

        feature_columns = []
        present_terms = []
        for term, lag in lag_map.items():
            if term in df.columns:
                shifted = df[term].shift(lag).rename(f'{term}_lag{lag}')
                feature_columns.append(shifted)
                present_terms.append(term)

        if not feature_columns:
            raise ValueError("None of the requested terms exist in merged data")

        X = pd.concat(feature_columns, axis=1)
        y = df['CPI_YoY']
        data = pd.concat([y.rename('CPI_YoY'), X], axis=1).dropna()

        train = data[data.index < test_start]
        test = data[data.index >= test_start]
        if len(train) < 24 or len(test) < 6:
            raise ValueError("Not enough data for split; choose earlier test_start")

        Xtr = np.c_[np.ones(len(train)), train.drop(columns=['CPI_YoY']).values]
        beta = np.linalg.lstsq(Xtr, train['CPI_YoY'].values, rcond=None)[0]

        Xte = np.c_[np.ones(len(test)), test.drop(columns=['CPI_YoY']).values]
        yhat = Xte @ beta
        yhat = pd.Series(yhat, index=test.index, name='model')

        baseline_last_month = test['CPI_YoY'].shift(1).rename('baseline_last_month')
        baseline_seasonal = test['CPI_YoY'].shift(12).rename('baseline_seasonal')

        comparison = pd.concat(
            [
                test['CPI_YoY'].rename('actual'),
                yhat,
                baseline_last_month,
                baseline_seasonal,
            ],
            axis=1,
        ).dropna()

        def rmse(a, b):
            return float(np.sqrt(((a - b) ** 2).mean()))

        rmse_model = rmse(comparison['actual'], comparison['model'])
        rmse_last_month = rmse(comparison['actual'], comparison['baseline_last_month'])
        rmse_seasonal = rmse(comparison['actual'], comparison['baseline_seasonal'])

        def directional_accuracy(pred, actual):
            return float((np.sign(pred.diff()) == np.sign(actual.diff())).mean())

        diracc_model = directional_accuracy(comparison['model'], comparison['actual'])
        diracc_last_month = directional_accuracy(
            comparison['baseline_last_month'], comparison['actual']
        )
        diracc_seasonal = directional_accuracy(
            comparison['baseline_seasonal'], comparison['actual']
        )

        comparison.to_csv(results_path)

        print(f"\nNowcast (lags={lag_map}, test_start={test_start})")
        print(
            f"RMSE  model={rmse_model:.3f} | last_month={rmse_last_month:.3f} | seasonal={rmse_seasonal:.3f}"
        )
        print(
            f"DirAcc model={diracc_model:.2%} | last_month={diracc_last_month:.2%} | seasonal={diracc_seasonal:.2%}"
        )
        print(f"✓ Saved: {results_path}")

        return {
        'lags': lag_map,
        'rmse_model': rmse_model,
        'rmse_last_month': rmse_last_month,
        'rmse_seasonal': rmse_seasonal,
        'diracc_model': diracc_model,
        'diracc_last_month': diracc_last_month,
        'diracc_seasonal': diracc_seasonal,
    }

def run_pycaret_on_pca(pca_csv_path: str = "results/pca_dataset_yoy_1m.csv"):
    """
    Use PyCaret to train and compare regression models
    on the PCA feature dataset for CPI_YoY_future_1m.
    """
    print("\n============================================================")
    print("Running PyCaret regression on PCA features...")
    print("============================================================")

    df = pd.read_csv(pca_csv_path)

    for col in ["date", "CPI_MoM", "CPI_YoY"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    target_col = "CPI_YoY_future_1m"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {pca_csv_path}")

    exp = setup(
        data=df,
        target=target_col,
        train_size=0.8,
        fold=5,
        fold_shuffle=True,
        session_id=42,
        verbose=False,   # no 'silent' arg
    )
# Try a bunch of models and pick the best by default metric (RMSE)
    best_model = compare_models()

    print("\nTop models leaderboard:")
    leaderboard = pull()
    print(leaderboard.head(10))

    Path("results").mkdir(parents=True, exist_ok=True)
    save_model(best_model, "results/best_pycaret_model_yoy_1m")
    print("\n✓ Saved best PyCaret model to 'results/best_pycaret_model_yoy_1m.pkl'")


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
    
    FOOD_TERMS = [
        "grocery prices", "grocery bill", "grocery inflation",
        "egg prices", "milk price", "beef prices", "bread price",
        "restaurant prices", "menu prices", "fast food prices", "takeout prices"
    ]
    search_terms = ["rent", "gas prices"] + FOOD_TERMS

    
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
            model.build_food_index(FOOD_TERMS, drop_components=True)
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

            # Add more columns (future CPI targets, CPI lags, trend leads)
            # AFTER: 1-month ahead targets instead of 3-month
            model.build_feature_matrix(horizon_months=1)

            # Calculate correlations (still just using the base *_trend columns)
            correlations = model.calculate_correlations()
            
            # Create visualizations
            model.visualize_correlations(save_path='results/fig_corr_heatmap_scatter.png')
            model.plot_small_multiples(save_path='results/fig_timeseries_small_multiples.png')
            
            # Save merged dataset (now includes all the new feature columns)
            model.save_data(filepath='merged_inflation_data.csv')

            # Save correlation results
            correlations.to_csv('correlation_results.csv')
            print(f"✓ Correlation results saved to: correlation_results.csv")

            # --- PCA step: build dataset for modeling (e.g., for PyCaret) ---
            Path('results').mkdir(parents=True, exist_ok=True)

            # Forecast target: CPI YoY 1 month ahead
            pca_target = "CPI_YoY_future_1m"

            df_pca = model.build_pca_dataset(
                target_col=pca_target,
                variance_threshold=0.95,
                save_explained_path='results/pca_explained_variance_yoy_1m.csv'
            )
            df_pca.to_csv('results/pca_dataset_yoy_1m.csv')
            print("✓ PCA dataset saved to: results/pca_dataset_yoy_1m.csv")

            model.lead_lag_correlations(ks=(0,1,2,3), targets=('CPI_MoM','CPI_YoY'),
                            save_path='results/lead_corr.csv')

            # Simple nowcasts with rent_lag3 and gas at 0/1/3; pick a reasonable backtest start
            results = []
            configs = [
                {'rent_trend': 3, 'gas_prices_trend': 0},
                {'rent_trend': 3, 'gas_prices_trend': 1},
                {'rent_trend': 3, 'gas_prices_trend': 3},
            ]
            for conf in configs:
                path = f"results/nowcast_r3_g{conf['gas_prices_trend']}.csv"
                res = model.nowcast_yoy_custom(lag_map=conf, test_start='2019-01-01', save_path=path)
                results.append(res)

            # Run PyCaret on the PCA dataset
            run_pycaret_on_pca("results/pca_dataset_yoy_1m.csv")


            print("\n" + "=" * 60)
            print("Analysis Complete!")
            print("=" * 60)
            print("\nKey Findings:")
            print(f"  • Strongest predictor (by absolute correlation with CPI YoY):")
            best_predictor = correlations['CPI_YoY'].abs().idxmax()
            best_corr = correlations.loc[best_predictor, 'CPI_YoY']
            print(f"    {best_predictor.replace('_trend', '')}: {best_corr:.3f}")
            print(f"\n  • All correlation results saved to 'correlation_results.csv'")
            print(f"  • Correlation heatmap & scatter saved to 'results/fig_corr_heatmap_scatter.png'")
            print(f"  • Small multiples chart saved to 'results/fig_cpi_vs_trends_small_multiples.png'")
            print(f"  • Merged data saved to 'merged_inflation_data.csv'")
        

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()