import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.exception import Custom_exception_handling
from src.logger import logging
from src.utlis import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Parameters for Linear Regression (empty dict as no hyperparameters are tuned in this case)
            params = {}

            # Evaluate the model
            test_model_score = evaluate_model(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, param=params
            )

            # Save the trained model
            model = LinearRegression()
            model.fit(X_train, y_train)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            # Convert R-squared score to percentage
            accuracy = test_model_score * 100
            logging.info(f"Model accuracy: {accuracy:.2f}%")

            return accuracy

        except Exception as e:
            raise Custom_exception_handling(e, sys)
