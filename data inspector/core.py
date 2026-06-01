import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats

class DataInspector:
    """Automated framework for CSV data ingestion, sanitization, and analysis."""
    
    def __init__(self):
        self.df: pd.DataFrame = None

    def upload_data(self) -> None:
        try:
            from google.colab import files
        except ImportError:
            print("❌ This method is designed specifically for Google Colab environments.")
            return

        uploaded = files.upload()
        if not uploaded:
            print("❌ Upload cancelled.")
            return
        
        file_name = list(uploaded.keys())[0]
        garbage_strings = ['?', 'n/a', 'N/A', 'NULL', 'null', ' ', '']
        self.df = pd.read_csv(io.BytesIO(uploaded[file_name]), na_values=garbage_strings)
        print(f"✅ Successfully loaded '{file_name}' ({self.df.shape[0]} rows, {self.df.shape[1]} columns).")
        self._auto_type_correction()

    def _auto_type_correction(self) -> None:
        if self.df is None: return
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                converted = pd.to_numeric(self.df[col], errors='coerce')
                if not converted.isna().all():
                    self.df[col] = converted

    def data_summary(self) -> None:
        if self.df is None: return
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        print("="*60)
        print(f" STRUCTURAL SUMMARY: {self.df.shape[0]} Rows | {self.df.shape[1]} Columns")
        print(f" Numeric Features ({len(num_cols)}): {num_cols}")
        print(f" Categorical Features ({len(cat_cols)}): {cat_cols}")
        print("="*60)
        display(self.df.head(5))

    def handle_missing_values(self, strategy: str = 'median', fill_value=None) -> None:
        if self.df is None: return
        for col in self.df.columns:
            if self.df[col].isna().sum() == 0: continue
                
            if strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)
            elif self.df[col].dtype in [np.float64, np.int64]:
                if strategy == 'mean': self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median': self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode': self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "Missing")
        print(f"✅ Missing value resolution achieved via strategy: '{strategy}'.")

    def handle_outliers(self, columns: list, action: str = 'flag') -> pd.DataFrame:
        if self.df is None: return None
        outlier_mask = pd.Series(False, index=self.df.index)
        
        for col in columns:
            if col in self.df.columns and self.df[col].dtype in [np.float64, np.int64]:
                Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask |= (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)
                
        if action == 'delete':
            initial_len = len(self.df)
            self.df = self.df[~outlier_mask]
            print(f"🗑️ Dropped {initial_len - len(self.df)} rows containing IQR outliers.")
        else:
            self.df['is_outlier'] = outlier_mask.astype(int)
            print("🚩 Outliers flagged.")
        return self.df

    def extract_normalized_numeric_data(self, strategy: str = 'standard') -> pd.DataFrame:
        if self.df is None: return pd.DataFrame()
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if 'is_outlier' in num_df.columns: num_df.drop(columns=['is_outlier'], inplace=True)
        
        for col in num_df.columns:
            if strategy == 'minmax':
                min_val, max_val = num_df[col].min(), num_df[col].max()
                num_df[col] = (num_df[col] - min_val) / (max_val - min_val + 1e-9)
            elif strategy == 'standard':
                num_df[col] = (num_df[col] - num_df[col].mean()) / (num_df[col].std() + 1e-9)
            elif strategy == 'robust':
                q25, q50, q75 = num_df[col].quantile(0.25), num_df[col].median(), num_df[col].quantile(0.75)
                num_df[col] = (num_df[col] - q50) / (q75 - q25 + 1e-9)
        return num_df

    def extract_normalized_categorical_data(self, strategy: str = 'onehot') -> pd.DataFrame:
        if self.df is None: return pd.DataFrame()
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if cat_df.empty: return pd.DataFrame()
        
        if strategy == 'onehot': return pd.get_dummies(cat_df, dtype=float)
        elif strategy in ['ordinal', 'uniform']:
            for col in cat_df.columns:
                cat_df[col] = cat_df[col].astype('category').cat.codes
                if strategy == 'uniform' and cat_df[col].max() > 0:
                    cat_df[col] = cat_df[col] / cat_df[col].max()
            return cat_df.astype(float)

    def merge_processed_datasets(self, num_strategy: str = 'standard', cat_strategy: str = 'onehot') -> pd.DataFrame:
        num_part = self.extract_normalized_numeric_data(strategy=num_strategy)
        cat_part = self.extract_normalized_categorical_data(strategy=cat_strategy)
        return pd.concat([num_part, cat_part], axis=1)

    def plot_univariate_subplots(self, column: str) -> None:
        if self.df is None or column not in self.df.columns: return
        fig = make_subplots(rows=3, cols=1, subplot_titles=(f"Violin", f"Scatter", f"Histogram"))
        fig.add_trace(go.Violin(x=self.df[column], box_visible=True, meanline_visible=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode='markers'), row=2, col=1)
        fig.add_trace(go.Histogram(x=self.df[column]), row=3, col=1)
        fig.update_layout(height=800, template="plotly_white", showlegend=False)
        fig.show()

    def plot_relationship(self, col_x: str, col_y: str) -> None:
        if self.df is None or col_x not in self.df.columns or col_y not in self.df.columns: return
        is_x_num = self.df[col_x].dtype in [np.float64, np.int64]
        is_y_num = self.df[col_y].dtype in [np.float64, np.int64]
        
        if is_x_num and is_y_num:
            fig = px.scatter(self.df, x=col_x, y=col_y, trendline="ols")
        elif not is_x_num and not is_y_num:
            counts = self.df.groupby([col_x, col_y]).size().reset_index(name='Counts')
            fig = px.bar(counts, x=col_x, y='Counts', color=col_y, barmode='group')
        else:
            cat_col = col_x if not is_x_num else col_y
            num_col = col_y if is_y_num else col_x
            fig = px.box(self.df, x=cat_col, y=num_col, points="all")
        fig.update_layout(template="plotly_white")
        fig.show()