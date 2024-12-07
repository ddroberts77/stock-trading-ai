import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

class ModelOptimizer:
    def __init__(self, model_class, param_grid):
        self.model_class = model_class
        self.param_grid = param_grid
        self.best_params = None
        self.best_model = None

    def optimize(self, X_train, y_train, cv=5):
        """Perform grid search to find optimal hyperparameters"""
        model = KerasRegressor(build_fn=self.model_class)
        grid = GridSearchCV(
            estimator=model,
            param_grid=self.param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=2
        )
        grid_result = grid.fit(X_train, y_train)
        self.best_params = grid_result.best_params_
        return grid_result

    def train_best_model(self, X_train, y_train, validation_split=0.2):
        """Train model with best parameters"""
        if not self.best_params:
            raise ValueError('Must run optimize() first')

        self.best_model = self.model_class(**self.best_params)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.best_model.fit(
            X_train, y_train,
            validation_split=validation_split,
            callbacks=[early_stopping],
            epochs=1000,
            verbose=1
        )
        return history