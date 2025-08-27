"""
Serie A Betting Model - Updated Version
Replace your existing src/betting_model.py with this file
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
        self.team_name_mapping = {
            # Add team name variations here if needed
            'AC Milan': 'Milan',
            'Inter Milan': 'Inter',
            'Hellas Verona': 'Verona',
            # Add more as needed based on check_teams.py output
        }
        
    def normalize_team_name(self, team_name):
        """Normalize team names to handle variations"""
        return self.team_name_mapping.get(team_name, team_name)
        
    def load_and_clean_data(self, df):
        """Clean and prepare the dataset - IMPROVED VERSION"""
        print("🧹 Cleaning data...")
        
        # Normalize team names
        df['home_team'] = df['home_team'].apply(self.normalize_team_name)
        df['away_team'] = df['away_team'].apply(self.normalize_team_name)
        
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
        
        # PPG difference - with better handling of missing values
        df['home_ppg'] = df['home_ppg'].fillna(1.0)
        df['away_ppg'] = df['away_ppg'].fillna(1.0)
        df['ppg_difference'] = df['home_ppg'] - df['away_ppg']
        
        if 'Pre-Match PPG (Home)' in df.columns:
            df['Pre-Match PPG (Home)'] = df['Pre-Match PPG (Home)'].fillna(df['home_ppg'])
            df['Pre-Match PPG (Away)'] = df['Pre-Match PPG (Away)'].fillna(df['away_ppg'])
            df['pre_match_ppg_difference'] = df['Pre-Match PPG (Home)'] - df['Pre-Match PPG (Away)']
        else:
            df['pre_match_ppg_difference'] = df['ppg_difference']
        
        # xG features - improved handling
        if 'team_a_xg' in df.columns and 'team_b_xg' in df.columns:
            df['team_a_xg'] = df['team_a_xg'].fillna(1.0)
            df['team_b_xg'] = df['team_b_xg'].fillna(1.0)
            df['xg_difference'] = df['team_a_xg'] - df['team_b_xg']
        else:
            df['team_a_xg'] = 1.0
            df['team_b_xg'] = 1.0
            df['xg_difference'] = 0
            
        if 'Home Team Pre-Match xG' in df.columns and 'Away Team Pre-Match xG' in df.columns:
            df['Home Team Pre-Match xG'] = df['Home Team Pre-Match xG'].fillna(df['team_a_xg'])
            df['Away Team Pre-Match xG'] = df['Away Team Pre-Match xG'].fillna(df['team_b_xg'])
            df['pre_match_xg_difference'] = df['Home Team Pre-Match xG'] - df['Away Team Pre-Match xG']
        else:
            df['pre_match_xg_difference'] = df['xg_difference']
        
        # Improved odds handling
        if 'odds_ft_home_team_win' in df.columns:
            # Remove invalid odds
            df = df[df['odds_ft_home_team_win'] > 1.0]
            df = df[df['odds_ft_draw'] > 1.0]
            df = df[df['odds_ft_away_team_win'] > 1.0]
            
            df['home_win_probability'] = 1 / df['odds_ft_home_team_win']
            df['draw_probability'] = 1 / df['odds_ft_draw']
            df['away_win_probability'] = 1 / df['odds_ft_away_team_win']
            
            # Market efficiency
            df['total_probability'] = df['home_win_probability'] + df['draw_probability'] + df['away_win_probability']
            df['bookmaker_margin'] = (df['total_probability'] - 1) * 100
            
            # Only keep matches with reasonable margins (2-15%)
            df = df[(df['bookmaker_margin'] >= 2) & (df['bookmaker_margin'] <= 15)]
        
        # Fill remaining missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        print(f"✅ Data cleaned. {len(df)} matches remaining.")
        return df
    
    def create_features(self, df):
        """Create feature matrix for modeling - IMPROVED VERSION"""
        print("🔧 Creating features...")
        
        # Core features that should always exist
        feature_columns = []
        
        # Team strength (most important)
        strength_features = ['home_ppg', 'away_ppg', 'ppg_difference']
        feature_columns.extend(strength_features)
        
        # Pre-match features
        if 'pre_match_ppg_difference' in df.columns:
            feature_columns.append('pre_match_ppg_difference')
        
        # Expected goals
        xg_features = ['team_a_xg', 'team_b_xg', 'xg_difference']
        feature_columns.extend(xg_features)
        
        if 'pre_match_xg_difference' in df.columns:
            feature_columns.append('pre_match_xg_difference')
        
        # Match stats (if available)
        match_stats = [
            'home_team_shots', 'away_team_shots',
            'home_team_shots_on_target', 'away_team_shots_on_target',
            'home_team_possession', 'away_team_possession',
            'home_team_corner_count', 'away_team_corner_count'
        ]
        
        for stat in match_stats:
            if stat in df.columns:
                feature_columns.append(stat)
        
        # Betting market features
        market_features = ['home_win_probability', 'draw_probability', 'away_win_probability']
        for feature in market_features:
            if feature in df.columns:
                feature_columns.append(feature)
        
        # Create the features dataframe
        features_df = df[feature_columns].copy()
        
        # Fill any remaining NaN values
        features_df = features_df.fillna(features_df.median())
        
        print(f"✅ Created {len(feature_columns)} features")
        return features_df
    
    def train_outcome_models(self, df):
        """Train models for match outcome prediction (1X2) - IMPROVED"""
        print("🎯 Training match outcome models...")
        
        features = self.create_features(df)
        
        # Create outcome target (0=home, 1=draw, 2=away)
        y_outcome = np.where(df['home_win'] == 1, 0, 
                            np.where(df['draw'] == 1, 1, 2))
        
        # Use only the best models with better parameters
        models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced outcomes
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Time series validation
        tscv = TimeSeriesSplit(n_splits=3)
        
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
                    importance = dict(zip(features.columns, model.feature_importances_))
                    self.feature_importance[f'outcome_{name}'] = importance
                    
            except Exception as e:
                print(f"   ⚠️  Error training {name}: {str(e)}")
    
    def train_goals_models(self, df):
        """Train models for goals betting (Over/Under, BTTS) - IMPROVED"""
        print("⚽ Training goals betting models...")
        
        features = self.create_features(df)
        targets = ['over_2.5', 'btts']
        
        for target in targets:
            if target not in df.columns:
                print(f"   ⚠️  Target {target} not found, skipping...")
                continue
                
            y = df[target]
            
            # Use best performing models
            models = {
                'rf': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                ),
                'gb': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
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
                    
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(features.columns, model.feature_importances_))
                        self.feature_importance[f'{target}_{name}'] = importance
                        
                except Exception as e:
                    print(f"   ⚠️  Error training {name} for {target}: {str(e)}")
    
    def calculate_betting_value(self, df, confidence_threshold=0.68, value_threshold=1.12):
        """Calculate betting value - IMPROVED VERSION"""
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
            
            # Use Random Forest as primary model (usually most reliable)
            if 'outcome_rf' in self.models and 'odds_ft_home_team_win' in match_data:
                try:
                    outcome_probs = self.models['outcome_rf'].predict_proba(match_features)[0]
                    outcome_confidence = max(outcome_probs)
                    
                    if outcome_confidence >= confidence_threshold:
                        predicted_outcome = np.argmax(outcome_probs)
                        
                        if predicted_outcome == 0 and outcome_probs[0] >= confidence_threshold:  # Home win
                            odds = match_data['odds_ft_home_team_win']
                            if pd.notna(odds) and odds > 1.0:
                                value = outcome_probs[0] * odds
                                if value >= value_threshold:
                                    recommendations['bets'].append({
                                        'bet_type': 'Home Win',
                                        'odds': odds,
                                        'probability': outcome_probs[0],
                                        'value': value,
                                        'confidence': outcome_confidence
                                    })
                        
                        elif predicted_outcome == 2 and outcome_probs[2] >= confidence_threshold:  # Away win
                            odds = match_data['odds_ft_away_team_win']
                            if pd.notna(odds) and odds > 1.0:
                                value = outcome_probs[2] * odds
                                if value >= value_threshold:
                                    recommendations['bets'].append({
                                        'bet_type': 'Away Win',
                                        'odds': odds,
                                        'probability': outcome_probs[2],
                                        'value': value,
                                        'confidence': outcome_confidence
                                    })
                        
                        # Only bet on draws if very confident (draws are harder to predict)
                        elif predicted_outcome == 1 and outcome_probs[1] >= (confidence_threshold + 0.05):
                            odds = match_data['odds_ft_draw']
                            if pd.notna(odds) and odds > 1.0:
                                value = outcome_probs[1] * odds
                                if value >= (value_threshold + 0.05):  # Higher threshold for draws
                                    recommendations['bets'].append({
                                        'bet_type': 'Draw',
                                        'odds': odds,
                                        'probability': outcome_probs[1],
                                        'value': value,
                                        'confidence': outcome_confidence
                                    })
                except Exception as e:
                    pass
            
            # Goals betting with better logic
            if 'over_2.5_rf' in self.models and 'odds_ft_over25' in match_data:
                try:
                    over25_prob = self.models['over_2.5_rf'].predict_proba(match_features)[0][1]
                    if over25_prob >= confidence_threshold:
                        odds = match_data['odds_ft_over25']
                        if pd.notna(odds) and odds > 1.0:
                            value = over25_prob * odds
                            if value >= value_threshold:
                                recommendations['bets'].append({
                                    'bet_type': 'Over 2.5 Goals',
                                    'odds': odds,
                                    'probability': over25_prob,
                                    'value': value,
                                    'confidence': over25_prob
                                })
                    
                    # Also consider under 2.5 if probability is low enough
                    elif over25_prob <= (1 - confidence_threshold):
                        under25_prob = 1 - over25_prob
                        # Estimate under 2.5 odds (usually around 1/(1-over_prob))
                        estimated_under_odds = 1 / under25_prob * 0.95  # Adjust for margin
                        value = under25_prob * estimated_under_odds
                        if value >= value_threshold and estimated_under_odds >= 1.3:
                            recommendations['bets'].append({
                                'bet_type': 'Under 2.5 Goals',
                                'odds': estimated_under_odds,
                                'probability': under25_prob,
                                'value': value,
                                'confidence': under25_prob
                            })
                            
                except Exception as e:
                    pass
            
            # BTTS betting
            if 'btts_rf' in self.models and 'odds_btts_yes' in match_data:
                try:
                    btts_prob = self.models['btts_rf'].predict_proba(match_features)[0][1]
                    if btts_prob >= confidence_threshold:
                        odds = match_data['odds_btts_yes']
                        if pd.notna(odds) and odds > 1.0:
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
    
    def backtest_strategy(self, df, start_date=None, confidence_threshold=0.68, value_threshold=1.12):
        """Backtest the betting strategy - IMPROVED"""
        print("📊 Running backtest...")
        
        if start_date:
            df_test = df[df['date_GMT'] >= start_date].copy()
        else:
            # Use last 25% of data for backtesting (more recent data)
            split_idx = int(len(df) * 0.75)
            df_test = df.iloc[split_idx:].copy()
        
        print(f"   Backtesting on {len(df_test)} matches from {df_test['date_GMT'].min()} to {df_test['date_GMT'].max()}")
        
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
                elif bet['bet_type'] == 'Draw' and match_data['draw'] == 1:
                    won = True
                elif bet['bet_type'] == 'Over 2.5 Goals' and match_data['over_2.5'] == 1:
                    won = True
                elif bet['bet_type'] == 'Under 2.5 Goals' and match_data['over_2.5'] == 0:
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
        
        # Calculate results
        if total_bets > 0:
            profit = total_return - total_stake
            roi = (profit / total_stake) * 100
            win_rate = (winning_bets / total_bets) * 100
            
            print("\n" + "="*50)
            print("🏆 BACKTESTING RESULTS")
            print("="*50)
            print(f"📊 Total Bets: {total_bets}")
            print(f"✅ Winning Bets: {winning_bets}")
            print(f"❌ Losing Bets: {total_bets - winning_bets}")
            print(f"📈 Win Rate: {win_rate:.1f}%")
            print(f"💰 Total Stake: {total_stake:.2f} units")
            print(f"💵 Total Return: {total_return:.2f} units")
            print(f"📊 Profit/Loss: {profit:+.2f} units")
            print(f"📈 ROI: {roi:+.1f}%")
            
            if win_rate >= 50 and roi > 0:
                print("🎉 PROFITABLE STRATEGY!")
            elif win_rate >= 50:
                print("⚠️  Good win rate but low ROI - consider higher value threshold")
            else:
                print("❌ Strategy needs improvement - consider higher confidence threshold")
            
            print("="*50)
        else:
            print("⚠️  No betting opportunities found in backtest period")
            print("   Try lowering confidence_threshold or value_threshold")
        
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
            print(f"⚠️  Model {model_name} not found")
            print(f"Available models: {list(self.models.keys())}")
            