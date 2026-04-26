from preprocess import cleaning_pipeline
from features import feature_engineering_pipeline

def run_pipeline():

    cleaning_pipeline()

    feature_engineering_pipeline()


if __name__ == "__main__":
    run_pipeline()