from preprocess import cleaning_pipeline
from features import feature_engineering_pipeline
from training import training_pipeling

def run_pipeline():

    cleaning_pipeline()

    feature_engineering_pipeline()

    training_pipeling()


if __name__ == "__main__":
    run_pipeline()