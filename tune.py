from pprint import pprint

from src.text_summarizer.components.data_ingestion import DataIngestion
from src.text_summarizer.components.data_transformation import DataTransformation
from src.text_summarizer.components.hyperparameter_tuning import HyperparameterTuner


def main():
    dataset = DataIngestion().load_data()
    tokenized_dataset = DataTransformation().transform_data(dataset)
    best_run = HyperparameterTuner().tune(tokenized_dataset)

    print("\nBest hyperparameters")
    pprint(best_run)


if __name__ == "__main__":
    main()
