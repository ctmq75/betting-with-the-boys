"""
Serie A Betting Model - Simplified Version
Uses only scikit-learn (no XGBoost/LightGBM dependencies)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class SerieABettingModel:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        
    def load_and_clean_data(self, df):
        """Clean and prepare the dataset"""
        print("🧹 Cleaning data...")
        
        # Convert date
        df['date_GMT'] = pd.to_datetime(df['date_GMT'])
        
        # Create outcome variables
        df['home_win'] = (df['home_team_goal_count'] > df['away_team_goal_count']).astype(int)
        df['draw'] = (df['home_team_goal_count'] == df['away_team_goal_count']).astype(int)
        df['away_win'] = (df['home_team_goal_count'] < df['away_team_goal_count']).astype(int)
        
        # Create betting targets
        df['over_1.5'] = (df['total_goal_count'] > 1.5).astype(int)
        df['over_2.5'] = (df['total_goal_count'] > 2.5).astype(int)
        df['over_3.5'] = (df['total_goal_count'] > 3.5).astype(int)
        df['btts'] = ((df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)).astype(int)
        
        # Goal difference
        df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
        
        # Sort by date
        df = df.sort_values('date_GMT')
        
        # PPG difference
        df['ppg_difference'] = df.get('home_ppg', 0) - df.get('away_ppg', 0)
        df['pre_match_ppg_difference'] = df.get('Pre-Match PPG (Home)', 0) - df.get('Pre-Match PPG (Away)', 0)
        
        # xG features
        if 'team_a_xg' in df.columns and 'team_b_xg' in df.columns:
            df['xg_difference'] = df['team_a_xg'] - df['team_b_xg']
        else:
            df['xg_difference'] = 0
            
        if 'Home Team Pre-Match xG' in df.columns and 'Away Team Pre-Match xG' in df.columns:
            df['pre_match_xg_difference'] = df['Home Team Pre-Match xG'] - df['Away Team Pre-Match xG']
        else:
            df['pre_match_xg_difference'] = 0
        
        # Odds-based features
        if 'odds_ft_home_team_win' in df.columns:
            df['home_win_probability'] = 1 / df['odds_ft_home_team_win']
            df['draw_probability'] = 1 / df['odds_ft_draw']
            df['away_win_probability'] = 1 / df['odds_ft_away_team_win']
            
            # Market efficiency
            df['total_probability'] = df['home_win_probability'] + df['draw_probability'] + df['away_win_probability']
            df['bookmaker_margin'] = (df['total_probability'] - 1) * 100
            
            # Value betting features
            df['home_win_value'] = df['home_win_probability'] * df['odds_ft_home_team_win']
            df['draw_value'] = df['draw_probability'] * df['odds_ft_draw']
            df['away_win_value'] = df['away_win_probability'] * df['odds_ft_away_team_win']
        
        if 'odds_ft_over25' in df.columns:
            df['over_2.5_probability'] = 1 / df['odds_ft_over25']
        
        if 'odds_btts_yes' in df.columns:
            df['btts_probability'] = 1 / df['odds_btts_yes']
        
        return df
    
    def create_features(self, df):
        """Create feature matrix for modeling"""
        print("🔧 Creating features...")
        
        # Define feature columns
        feature_columns = []
        
        # Basic team stats
        basic_features = [
            'home_ppg', 'away_ppg', 'ppg_difference',
            'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'pre_match_ppg_difference'
        ]
        
        for col in basic_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # xG features
        xg_features = [
            'team_a_xg', 'team_b_xg', 'xg_difference',
            'Home Team Pre-Match xG', 'Away Team Pre-Match xG', 'pre_match_xg_difference'
        ]
        
        for col in xg_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Match stats
        match_features = [
            'home_team_shots', 'away_team_shots',
            'home_team_shots_on_target', 'away_team_shots_on_target',
            'home_team_possession', 'away_team_possession',
            'home_team_corner_count', 'away_team_corner_count',
            'home_team_yellow_cards', 'away_team_yellow_cards',
            'home_team_red_cards', 'away_team_red_cards',
            'home_team_fouls', 'away_team_fouls'
        ]
        
        for col in match_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Pre-match percentages
        percentage_features = [
            'average_goals_per_match_pre_match', 'btts_percentage_pre_match',
            'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
            'over_35_percentage_pre_match'
        ]
        
        for col in percentage_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Odds features
        odds_features = [
            'home_win_probability', 'draw_probability', 'away_win_probability',
            'over_2.5_probability', 'btts_probability', 'bookmaker_margin'
        ]
        
        for col in odds_features:
            if col in df.columns:
                feature_columns.append(col)
        
        # Game week
        if 'Game Week' in df.columns:
            feature_columns.append('Game Week')
        
        # Handle missing values and create features dataframe
        if not feature_columns:
            print("⚠️  No valid feature columns found. Using basic features.")
            feature_columns = ['home_ppg', 'away_ppg', 'ppg_difference']
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
        
        features_df = df[feature_columns].copy()
        features_df = features_df.fillna(features_df.median())
        
        print(f"✅ Created {len(feature_columns)} features")
        return features_df
    
    def train_outcome_models(self, df):
        """Train models for match outcome prediction (1X2)"""
        print("🎯 Training match outcome models...")
        
        features = self.create_features(df)
        
        # Create outcome target (0=home, 1=draw, 2=away)
        y_outcome = np.where(df['home_win'] == 1, 0, 
                            np.where(df['draw'] == 1, 1, 2))
        
        # Time series split for proper validation
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for speed
        
        # Use only scikit-learn models
        models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        for name, model in models.items():
            accuracies = []
            try:
                for train_idx, val_idx in tscv.split(features):
                    X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                    y_train, y_val = y_outcome[train_idx], y_outcome[val_idx]
                    
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    accuracies.append(accuracy_score(y_val, pred))
                
                avg_accuracy = np.mean(accuracies)
                print(f"   {name.upper()} Accuracy: {avg_accuracy:.3f}")
                
                # Train on full dataset
                model.fit(features, y_outcome)
                self.models[f'outcome_{name}'] = model
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f'outcome_{name}'] = dict(zip(features.columns, model.feature_importances_))
                    
            except Exception as e:
                print(f"   ⚠️  Error training {name}: {str(e)}")
    
    def train_goals_models(self, df):
        """Train models for goals betting (Over/Under, BTTS)"""
        print("⚽ Training goals betting models...")
        
        features = self.create_features(df)
        targets = ['over_2.5', 'btts']
        
        for target in targets:
            if target not in df.columns:
                print(f"   ⚠️  Target {target} not found, skipping...")
                continue
                
            y = df[target]
            
            models = {
                'rf': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            for name, model in models.items():
                accuracies = []
                try:
                    for train_idx, val_idx in tscv.split(features):
                        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        model.fit(X_train, y_train)
                        pred = model.predict(X_val)
                        accuracies.append(accuracy_score(y_val, pred))
                    
                    avg_accuracy = np.mean(accuracies)
                    print(f"   {name.upper()} {target}: {avg_accuracy:.3f}")
                    
                    # Train on full dataset
                    model.fit(features, y)
                    self.models[f'{target}_{name}'] = model
                    
                    # Store feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[f'{target}_{name}'] = dict(zip(features.columns, model.feature_importances_))
                        
                except Exception as e:
                    print(f"   ⚠️  Error training {name} for {target}: {str(e)}")
    
    def calculate_betting_value(self, df, confidence_threshold=0.6, value_threshold=1.05):
        """Calculate betting value and generate recommendations"""
        print("💰 Calculating betting opportunities...")
        
        features = self.create_features(df)
        betting_recommendations = []
        
        for idx in range(len(df)):
            match_features = features.iloc[idx:idx+1]
            match_data = df.iloc[idx]
            
            recommendations = {
                'date': match_data['date_GMT'],
                'home_team': match_data['home_team'],
                'away_team': match_data['away_team'],
                'bets': []
            }
            
            # Check outcome betting using Random Forest (most reliable)
            if 'outcome_rf' in self.models and 'odds_ft_home_team_win' in match_data:
                try:
                    outcome_probs = self.models['outcome_rf'].predict_proba(match_features)[0]
                    outcome_confidence = max(outcome_probs)
                    
                    if outcome_confidence >= confidence_threshold:
                        predicted_outcome = np.argmax(outcome_probs)
                        
                        if predicted_outcome == 0:  # Home win
                            odds = match_data['odds_ft_home_team_win']
                            if pd.notna(odds) and odds > 0:
                                value = outcome_probs[0] * odds
                                if value >= value_threshold:
                                    recommendations['bets'].append({
                                        'bet_type': 'Home Win',
                                        'odds': odds,
                                        'probability': outcome_probs[0],
                                        'value': value,
                                        'confidence': outcome_confidence
                                    })
                        
                        elif predicted_outcome == 2:  # Away win
                            odds = match_data['odds_ft_away_team_win']
                            if pd.notna(odds) and odds > 0:
                                value = outcome_probs[2] * odds
                                if value >= value_threshold:
                                    recommendations['bets'].append({
                                        'bet_type': 'Away Win',
                                        'odds': odds,
                                        'probability': outcome_probs[2],
                                        'value': value,
                                        'confidence': outcome_confidence
                                    })
                except Exception as e:
                    pass
            
            # Check over 2.5 goals
            if 'over_2.5_rf' in self.models and 'odds_ft_over25' in match_data:
                try:
                    over25_prob = self.models['over_2.5_rf'].predict_proba(match_features)[0][1]
                    if over25_prob >= confidence_threshold:
                        odds = match_data['odds_ft_over25']
                        if pd.notna(odds) and odds > 0:
                            value = over25_prob * odds
                            if value >= value_threshold:
                                recommendations['bets'].append({
                                    'bet_type': 'Over 2.5 Goals',
                                    'odds': odds,
                                    'probability': over25_prob,
                                    'value': value,
                                    'confidence': over25_prob
                                })
                except Exception as e:
                    pass
            
            # Check BTTS
            if 'btts_rf' in self.models and 'odds_btts_yes' in match_data:
                try:
                    btts_prob = self.models['btts_rf'].predict_proba(match_features)[0][1]
                    if btts_prob >= confidence_threshold:
                        odds = match_data['odds_btts_yes']
                        if pd.notna(odds) and odds > 0:
                            value = btts_prob * odds
                            if value >= value_threshold:
                                recommendations['bets'].append({
                                    'bet_type': 'Both Teams To Score',
                                    'odds': odds,
                                    'probability': btts_prob,
                                    'value': value,
                                    'confidence': btts_prob
                                })
                except Exception as e:
                    pass
            
            if recommendations['bets']:
                betting_recommendations.append(recommendations)
        
        return betting_recommendations
    
    def backtest_strategy(self, df, start_date=None, confidence_threshold=0.6, value_threshold=1.05):
        """Backtest the betting strategy"""
        print("📊 Running backtest...")
        
        if start_date:
            df_test = df[df['date_GMT'] >= start_date].copy()
        else:
            # Use last 20% of data for backtesting
            split_idx = int(len(df) * 0.8)
            df_test = df.iloc[split_idx:].copy()
        
        print(f"   Backtesting on {len(df_test)} matches")
        
        recommendations = self.calculate_betting_value(df_test, confidence_threshold, value_threshold)
        
        total_bets = 0
        total_stake = 0
        total_return = 0
        winning_bets = 0
        
        bet_results = []
        
        for rec in recommendations:
            match_data = df_test[
                (df_test['date_GMT'] == rec['date']) & 
                (df_test['home_team'] == rec['home_team']) & 
                (df_test['away_team'] == rec['away_team'])
            ]
            
            if len(match_data) == 0:
                continue
                
            match_data = match_data.iloc[0]
            
            for bet in rec['bets']:
                total_bets += 1
                stake = 1  # Unit stake
                total_stake += stake
                
                # Check if bet won
                won = False
                if bet['bet_type'] == 'Home Win' and match_data['home_win'] == 1:
                    won = True
                elif bet['bet_type'] == 'Away Win' and match_data['away_win'] == 1:
                    won = True
                elif bet['bet_type'] == 'Over 2.5 Goals' and match_data['over_2.5'] == 1:
                    won = True
                elif bet['bet_type'] == 'Both Teams To Score' and match_data['btts'] == 1:
                    won = True
                
                if won:
                    winning_bets += 1
                    total_return += bet['odds'] * stake
                
                bet_results.append({
                    'date': rec['date'],
                    'match': f"{rec['home_team']} vs {rec['away_team']}",
                    'bet_type': bet['bet_type'],
                    'odds': bet['odds'],
                    'value': bet['value'],
                    'won': won,
                    'profit': (bet['odds'] * stake - stake) if won else -stake
                })
        
        # Calculate and display results
        if total_bets > 0:
            profit = total_return - total_stake
            roi = (profit / total_stake) * 100
            win_rate = (winning_bets / total_bets) * 100
            
            print("\n" + "="*50)
            print("🏆 BACKTESTING RESULTS")
            print("="*50)
            print(f"📊 Total Bets: {total_bets}")
            print(f"✅ Winning Bets: {winning_bets}")
            print(f"📈 Win Rate: {win_rate:.1f}%")
            print(f"💰 Total Stake: {total_stake:.2f} units")
            print(f"💵 Total Return: {total_return:.2f} units")
            print(f"📊 Profit: {profit:.2f} units")
            print(f"📈 ROI: {roi:.1f}%")
            print("="*50)
        else:
            print("⚠️  No betting opportunities found in backtest period")
        
        return bet_results
    
    def show_feature_importance(self, model_name='outcome_rf'):
        """Display feature importance"""
        if model_name in self.feature_importance:
            importance = self.feature_importance[model_name]
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            print(f"\n📈 Top 10 Most Important Features for {model_name}:")
            for i, (feature, score) in enumerate(sorted_importance[:10], 1):
                print(f"   {i:2d}. {feature}: {score:.4f}")
        else:
            print(f"⚠️  Model {model_name} not found or no feature importance available")
            print(f"Available models: {list(self.models.keys())}")