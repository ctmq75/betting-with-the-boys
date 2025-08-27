"""
Test the improved betting model
Run this after updating betting_model.py
"""

import pandas as pd
from src.betting_model import SerieABettingModel

def test_improved_model():
    print("🧪 Testing improved betting model...")
    
    # Load data
    try:
        df = pd.read_csv('Desktop/series-a-betting/data/raw/serie_a_historical.csv')
        print(f"✅ Loaded {len(df)} historical matches")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize model
    model = SerieABettingModel()
    
    # Clean data
    df_clean = model.load_and_clean_data(df)
    print(f"✅ Cleaned data: {len(df_clean)} matches")
    
    # Train models
    print("\n🎯 Training models...")
    model.train_outcome_models(df_clean)
    model.train_goals_models(df_clean)
    
    # Show trained models
    print(f"\n🤖 Trained models: {list(model.models.keys())}")
    
    # Run backtest
    print("\n📊 Running backtest...")
    results = model.backtest_strategy(
        df_clean, 
        confidence_threshold=0.65,  # Slightly lower to get more bets
        value_threshold=1.10       # Slightly lower to get more bets
    )
    
    print(f"\n📈 Backtest generated {len(results)} individual bets")
    
    # Show feature importance
    if 'outcome_rf' in model.models:
        model.show_feature_importance('outcome_rf')
    
    print("\n✅ Model test complete!")

if __name__ == "__main__":
    test_improved_model()