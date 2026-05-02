import pandas as pd
from sklearn.pipeline import Pipeline

def get_df_from_pipeline(pipeline : Pipeline, df : pd.DataFrame):
    df_ = pipeline.fit_transform(df)
    return pd.DataFrame(df_)


