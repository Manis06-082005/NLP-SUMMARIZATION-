from src.text_summarizer.components.data_ingestion import DataIngestion
from src.text_summarizer.components.data_transformation import DataTransformation
from src.text_summarizer.components.model_trainer import ModelTrainer


# ==========================================
# 1. DATA INGESTION
# ==========================================

data_ingestion = DataIngestion()

dataset = data_ingestion.load_data()


# ==========================================
# 2. DATA TRANSFORMATION
# ==========================================

data_transformation = DataTransformation()

tokenized_dataset = data_transformation.transform_data(
    dataset
)


# ==========================================
# 3. MODEL TRAINING
# ==========================================

model_trainer = ModelTrainer()

model = model_trainer.train(
    tokenized_dataset
)


print("\nTraining completed successfully")
print("The trained model is ready for prediction.")
