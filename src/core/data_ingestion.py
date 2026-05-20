import pandas as pd
import os

class DataIngestor:
    @staticmethod
    def load_csv(filepath):
        """
        Loads frequency response data from CSV or TXT. 
        Expects 'frequency' and 'magnitude' columns.
        Handles REW/AutoEQ formatted text files.
        """
        try:
            # Try to load as a simple space/tab delimited file first (common for REW)
            df = pd.read_csv(filepath, sep=None, engine='python', comment='*', header=None)
            if df.shape[1] >= 2:
                df = df.iloc[:, [0, 1]]
                df.columns = ['frequency', 'magnitude']
                return df
        except Exception:
            pass

        # Fallback to standard CSV
        df = pd.read_csv(filepath)
        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        if 'freq' in df.columns:
            df = df.rename(columns={'freq': 'frequency'})
        if 'mag' in df.columns:
            df = df.rename(columns={'mag': 'magnitude'})
            
        return df[['frequency', 'magnitude']]
