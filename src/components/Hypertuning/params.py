from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "XGBRegressor": XGBRegressor()
}

params = {
    "DecisionTreeRegressor": {
        "max_depth": [5], 
        "min_samples_split": [10],
        "criterion": [
            "squared_error", 
            "friedman_mse", 
            "absolute_error", 
            "poisson"
        ]
    },
    "RandomForestRegressor": {
        "n_estimators": [100],
        "max_depth": [10]
    },
    "GradientBoostingRegressor": {
        "n_estimators": [100], 
        "learning_rate": [0.1], 
        "subsample": [0.7, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9], 
        "max_depth": [3]
    },
    "AdaBoostRegressor": {
        "n_estimators": [50], 
        "learning_rate": [1.0]
    },
    "LinearRegression": {},
    "KNeighborsRegressor": {
        "n_neighbors": [5]
    },
    "CatBoostRegressor": {
        "iterations": [1000], 
        "learning_rate": [0.1], 
        "depth": [6]
    },
    "XGBRegressor": {
        "n_estimators": [100], 
        "learning_rate": [0.1], 
        "max_depth": [6]
    }
}
