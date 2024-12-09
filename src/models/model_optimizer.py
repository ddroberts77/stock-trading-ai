import torch
import torch.nn as nn
from typing import Dict, Any, Callable
from sklearn.model_selection import KFold
import numpy as np

class ModelOptimizer:
    def __init__(self, model_creator: Callable, param_grid: Dict[str, list]):
        """
        Initialize the model optimizer.
        
        Args:
            model_creator: Function that creates a PyTorch model
            param_grid: Dictionary of parameters to optimize
        """
        self.model_creator = model_creator
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = float('-inf')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _train_evaluate_model(self, model: nn.Module, train_data: torch.Tensor, 
                            train_labels: torch.Tensor, val_data: torch.Tensor, 
                            val_labels: torch.Tensor, params: Dict[str, Any]) -> float:
        """Train and evaluate a model configuration."""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        # Move data to device
        train_data = train_data.to(self.device)
        train_labels = train_labels.to(self.device)
        val_data = val_data.to(self.device)
        val_labels = val_labels.to(self.device)
        
        # Training
        model.train()
        for epoch in range(10):  # Quick training for validation
            optimizer.zero_grad()
            outputs = model(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_labels)
            
        return -val_loss.item()  # Negative because we want to maximize score
        
    def optimize(self, X: torch.Tensor, y: torch.Tensor, cv: int = 3) -> Dict[str, Any]:
        """
        Perform grid search with cross-validation.
        
        Args:
            X: Input data tensor
            y: Target tensor
            cv: Number of cross-validation folds
        """
        kf = KFold(n_splits=cv, shuffle=True)
        
        for params in self._get_param_combinations():
            scores = []
            
            for train_idx, val_idx in kf.split(X):
                # Create new model instance
                model = self.model_creator(**params)
                
                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train and evaluate
                try:
                    score = self._train_evaluate_model(
                        model, X_train, y_train, X_val, y_val, params
                    )
                    scores.append(score)
                except RuntimeError as e:
                    print(f"Error during training with params {params}: {str(e)}")
                    continue
                
            if scores:
                avg_score = np.mean(scores)
                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.best_params = params
        
        return self.best_params
    
    def _get_param_combinations(self):
        """Generate all parameter combinations from param_grid."""
        import itertools
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
    
    def train_best_model(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Train a model with the best parameters."""
        if self.best_params is None:
            raise ValueError("Must run optimize() before training best model")
            
        model = self.model_creator(**self.best_params)
        model = model.to(self.device)
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=self.best_params.get('learning_rate', 0.001))
        criterion = nn.MSELoss()
        
        history = {'loss': []}
        
        model.train()
        for epoch in range(100):  # Full training
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
            
        return history