"""
Serie A Betting Model - Main Execution Script
Fixed version for pandas compatibility
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/raw', 'data/processed', 'data/predictions',
        'models/saved_models', 'results/backtest_results', 
        'results/predictions', 'src'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Directories created successfully!")

def load_data():
    """Load and validate data"""
    data_file = 'data/raw/serie_a_historical.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please add your Serie A CSV file to data/raw/serie_a_historical.csv")
        return None
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(data_file, encoding=encoding)
                print(f"✅ Data loaded successfully with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try without specifying encoding
        df = pd.read_csv(data_file)
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def main():
    print("🏆 Serie A Betting Model")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Load data
    print("📊 Loading data...")
    df = load_data()
    
    if df is None:
        return
    
    print(f"✅ Loaded {len(df)} matches")
    print(f"📅 Date range: {df['date_GMT'].min()} to {df['date_GMT'].max()}")
    print(f"📋 Columns: {len(df.columns)}")
    print("\n🔍 First few column names:")
    for i, col in enumerate(df.columns[:10]):
        print(f"   {i+1}. {col}")
    
    # Check for required columns
    required_columns = [
        'date_GMT', 'home_team', 'away_team', 
        'home_team_goal_count', 'away_team_goal_count', 'total_goal_count'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"\n⚠️  Missing required columns: {missing_columns}")
        print("Please check your CSV file format.")
        return
    
    print("\n✅ All required columns found!")
    
    # Try to import and run the model
    try:
        from src.betting_model import SerieABettingModel
        
        print("🤖 Initializing betting model...")
        model = SerieABettingModel()
        
        print("🧹 Processing data...")
        df_clean = model.load_and_clean_data(df)
        
        print("🎯 Training models...")
        model.train_outcome_models(df_clean)
        model.train_goals_models(df_clean)
        
        print("💰 Running backtest...")
        results = model.backtest_strategy(df_clean, confidence_threshold=0.65, value_threshold=1.08)
        
        print("\n🎉 Analysis complete!")
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure src/betting_model.py exists and is properly formatted.")
    except Exception as e:
        print(f"❌ Error running model: {str(e)}")

if __name__ == "__main__":
    main()