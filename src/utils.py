import pandas as pd
import numpy as np


def search_df(df: pd.DataFrame, substring: str, case: bool = False) -> pd.DataFrame:
    mask = np.column_stack([df[col].astype(str).str.contains(substring.lower(), case=case, na=False) for col in df])
    return df.loc[mask.any(axis=1)]
