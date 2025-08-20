"""
Configuration file for Serie A Betting Model
Adjust these parameters to tune your model performance
"""

import os

# File paths
DATA_RAW_PATH = "data/raw/serie_a_historical.csv"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/saved_models/"
RESULTS_PATH = "results/"

# Model parameters
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
}

# Betting strategy parameters
BETTING_CONFIG = {
    'confidence_threshold': 0.65,  # Minimum model confidence (0.5-0.8)
    'value_threshold': 1.08,       # Minimum betting value (1.05-1.15)
    'max_odds': 5.0,              # Maximum odds to consider
    'min_odds': 1.3,              # Minimum odds to consider
    'bankroll_management': {
        'kelly_fraction': 0.25,    # Kelly criterion fraction
        'max_bet_size': 0.05,      # Max 5% of bankroll per bet
        'min_bet_size': 0.01       # Min 1% of bankroll per bet
    }
}

# Backtesting parameters
BACKTEST_CONFIG = {
    'test_size_ratio': 0.2,        # Use last 20% of data for testing
    'min_matches_for_training': 500, # Minimum matches needed to train
    'cross_validation_folds': 5,   # Number of CV folds
    'walk_forward_analysis': True  # Use walk-forward validation
}

# Feature engineering parameters
FEATURE_CONFIG = {
    'rolling_window_games': 5,     # Games for rolling averages
    'min_games_played': 3,         # Min games before making predictions
    'home_advantage_factor': True, # Include home advantage features
    'referee_features': True,      # Include referee statistics
    'weather_features': False     # Include weather (if available)
}

# Model evaluation metrics
EVALUATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss'
]

# Betting markets to predict
BETTING_MARKETS = {
    'match_outcome': True,         # 1X2 betting
    'over_under_2_5': True,        # Over/Under 2.5 goals
    'both_teams_score': True,      # BTTS
    'over_under_1_5': False,       # Over/Under 1.5 goals
    'over_under_3_5': False,       # Over/Under 3.5 goals
    'asian_handicap': False        # Asian handicap (if odds available)
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': 'results/betting_model.log'
}

# Create directories if they don't exist
def create_directories():
    directories = [
        "data/raw", "data/processed", "data/predictions",
        "models/saved_models", "results/backtest_results", 
        "results/predictions", "notebooks"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Project directories created successfully!")

if __name__ == "__main__":
    create_directories()