import logging

logging.basicConfig(
    filename='../logs/pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

from preprocess import clean
from features import feature_engineering_pipeline
from training import training_pipeling
from sklearn.pipeline import Pipeline

def run_pipeline():

    cleaning_pipeline : Pipeline = clean()

    feature_engineering_pipeline()

    training_pipeling()


if __name__ == "__main__":
    run_pipeline()