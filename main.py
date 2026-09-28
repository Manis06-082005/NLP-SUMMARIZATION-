from src.text_summarizer.components.data_ingestion import DataIngestion
from src.text_summarizer.components.data_transformation import DataTransformation
from src.text_summarizer.components.model_trainer import ModelTrainer
from src.text_summarizer.components.model_evaluation import ModelEvaluation


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


# ==========================================
# 4. MODEL EVALUATION
# ==========================================

model_evaluation = ModelEvaluation()

rouge_scores = model_evaluation.evaluate_model(
    dataset
)


# ==========================================
# 5. PRINT RESULT
# ==========================================

print("\nModel Evaluation Results")

print("ROUGE-1 :", rouge_scores["rouge1"])
print("ROUGE-2 :", rouge_scores["rouge2"])
print("ROUGE-L :", rouge_scores["rougeL"])
print("ROUGE-Lsum :", rouge_scores["rougeLsum"])
