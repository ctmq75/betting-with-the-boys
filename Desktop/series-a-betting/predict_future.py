"""
Serie A Future Game Predictor - COMPLETE FIXED VERSION
Predicts outcomes for upcoming matches using trained models
Enhanced with improved goal prediction formulas and consistent integration
All predictions are now logically consistent across different bet types
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class FutureGamePredictor:
    def __init__(self):
        self.trained_model = None
        self.team_stats = {}
        self.feature_columns = []
        
    def load_trained_model(self):
        """Load the model we trained on historical data"""
        from src.betting_model import SerieABettingModel
        
        # Load historical data and train model
        print("🔄 Loading historical data and training model...")
        df = pd.read_csv('Desktop/series-a-betting/data/raw/serie_a_historical.csv')
        
        model = SerieABettingModel()
        df_clean = model.load_and_clean_data(df)
        
        # Train models on ALL historical data
        model.train_outcome_models(df_clean)
        model.train_goals_models(df_clean)
        
        # Calculate latest team statistics
        self.calculate_current_team_stats(df_clean)
        
        # Store feature columns for consistency
        features = model.create_features(df_clean)
        self.feature_columns = list(features.columns)
        
        self.trained_model = model
        print("✅ Model trained and ready for predictions!")
        
        return model
    
    def calculate_current_team_stats(self, df):
        """Calculate current form and statistics for each team"""
        print("📊 Calculating current team statistics...")
        
        # Sort by date to get most recent data
        df = df.sort_values('date_GMT')
        
        # Get all unique teams
        teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        
        for team in teams:
            # Get team's recent matches (last 10 games)
            home_matches = df[df['home_team'] == team].tail(5)
            away_matches = df[df['away_team'] == team].tail(5)
            
            # Calculate recent form
            recent_points = 0
            recent_goals_for = 0
            recent_goals_against = 0
            games_played = 0
            
            # Home matches
            for _, match in home_matches.iterrows():
                games_played += 1
                goals_for = match['home_team_goal_count']
                goals_against = match['away_team_goal_count']
                recent_goals_for += goals_for
                recent_goals_against += goals_against
                
                if goals_for > goals_against:
                    recent_points += 3
                elif goals_for == goals_against:
                    recent_points += 1
            
            # Away matches
            for _, match in away_matches.iterrows():
                games_played += 1
                goals_for = match['away_team_goal_count']
                goals_against = match['home_team_goal_count']
                recent_goals_for += goals_for
                recent_goals_against += goals_against
                
                if goals_for > goals_against:
                    recent_points += 3
                elif goals_for == goals_against:
                    recent_points += 1
            
            # Calculate stats
            if games_played > 0:
                ppg = recent_points / games_played
                goals_per_game = recent_goals_for / games_played
                goals_against_per_game = recent_goals_against / games_played
            else:
                ppg = 1.0  # Default values
                goals_per_game = 1.0
                goals_against_per_game = 1.0
            
            self.team_stats[team] = {
                'ppg': ppg,
                'goals_per_game': goals_per_game,
                'goals_against_per_game': goals_against_per_game,
                'recent_form': recent_points,
                'games_played': games_played
            }
        
        print(f"✅ Calculated stats for {len(teams)} teams")
    
    def create_upcoming_match_features(self, home_team, away_team, game_week=20):
        """Create features for an upcoming match - UPDATED WITH CUSTOM WEIGHTS"""
        
        # Get team stats
        home_stats = self.team_stats.get(home_team, {'ppg': 1.0, 'goals_per_game': 1.0, 'goals_against_per_game': 1.0})
        away_stats = self.team_stats.get(away_team, {'ppg': 1.0, 'goals_per_game': 1.0, 'goals_against_per_game': 1.0})
        
        # FEATURE IMPORTANCE WEIGHTS - Adjust these to change what the model focuses on
        FEATURE_WEIGHTS = {
            'recent_form': 1.5,      # Recent PPG gets 1.5x weight (very important)
            'historical_form': 1.0,   # Overall PPG gets normal weight  
            'attacking': 1.3,        # Goals scored gets 1.3x weight (important)
            'defensive': 1.1,        # Goals conceded gets 1.1x weight
            'home_advantage': 1.2,   # Home advantage multiplier
            'xg_importance': 0.9,    # Expected goals less important than actual goals
            'possession': 0.8,       # Possession less important
            'set_pieces': 1.1        # Corners slightly more important
        }
        
        # Create feature dictionary matching training data structure
        features = {}
        
        # WEIGHTED TEAM PERFORMANCE
        features['home_ppg'] = home_stats['ppg'] * FEATURE_WEIGHTS['recent_form']
        features['away_ppg'] = away_stats['ppg'] * FEATURE_WEIGHTS['recent_form']
        features['ppg_difference'] = features['home_ppg'] - features['away_ppg']
        
        # Pre-match features (use current form as estimate)
        features['Pre-Match PPG (Home)'] = home_stats['ppg'] * FEATURE_WEIGHTS['historical_form']
        features['Pre-Match PPG (Away)'] = away_stats['ppg'] * FEATURE_WEIGHTS['historical_form']
        features['pre_match_ppg_difference'] = features['Pre-Match PPG (Home)'] - features['Pre-Match PPG (Away)']
        
        # WEIGHTED EXPECTED GOALS
        base_home_xg = home_stats['goals_per_game'] * FEATURE_WEIGHTS['attacking']
        base_away_xg = away_stats['goals_per_game'] * FEATURE_WEIGHTS['attacking']
        
        # Apply home advantage to xG
        home_xg = base_home_xg * FEATURE_WEIGHTS['home_advantage']
        away_xg = base_away_xg  # No home advantage for away team
        
        features['team_a_xg'] = home_xg * FEATURE_WEIGHTS['xg_importance']
        features['team_b_xg'] = away_xg * FEATURE_WEIGHTS['xg_importance']
        features['xg_difference'] = features['team_a_xg'] - features['team_b_xg']
        features['Home Team Pre-Match xG'] = features['team_a_xg']
        features['Away Team Pre-Match xG'] = features['team_b_xg']
        features['pre_match_xg_difference'] = features['xg_difference']
        
        # ENHANCED MATCH STATS with weights
        home_attack_strength = home_stats['goals_per_game'] * FEATURE_WEIGHTS['attacking']
        away_attack_strength = away_stats['goals_per_game'] * FEATURE_WEIGHTS['attacking']
        
        # Estimated shots based on attacking strength
        features['home_team_shots'] = (10 + home_attack_strength * 3) * FEATURE_WEIGHTS['home_advantage']
        features['away_team_shots'] = 10 + away_attack_strength * 3
        features['home_team_shots_on_target'] = features['home_team_shots'] * 0.35
        features['away_team_shots_on_target'] = features['away_team_shots'] * 0.35
        
        # WEIGHTED POSSESSION - stronger teams get more possession
        team_strength_diff = (home_stats['ppg'] - away_stats['ppg']) * FEATURE_WEIGHTS['recent_form']
        base_possession = 50
        possession_adjustment = team_strength_diff * 8  # Each PPG point = 8% possession
        home_advantage_possession = 3 * FEATURE_WEIGHTS['home_advantage']  # Home gets extra 3%
        
        features['home_team_possession'] = min(70, max(30, 
            base_possession + possession_adjustment + home_advantage_possession)) * FEATURE_WEIGHTS['possession']
        features['away_team_possession'] = (100 - features['home_team_possession']) * FEATURE_WEIGHTS['possession']
        
        # WEIGHTED SET PIECES (corners)
        features['home_team_corner_count'] = (4 + home_attack_strength) * FEATURE_WEIGHTS['set_pieces'] * FEATURE_WEIGHTS['home_advantage']
        features['away_team_corner_count'] = (4 + away_attack_strength) * FEATURE_WEIGHTS['set_pieces']
        
        # Cards and fouls - slightly weighted by aggression/form
        aggression_factor = 1 + abs(team_strength_diff) * 0.1  # Closer games = more cards
        features['home_team_yellow_cards'] = 2 * aggression_factor
        features['away_team_yellow_cards'] = 2 * aggression_factor  
        features['home_team_red_cards'] = 0
        features['away_team_red_cards'] = 0
        features['home_team_fouls'] = 12 * aggression_factor
        features['away_team_fouls'] = 12 * aggression_factor
        
        # Pre-match percentages (league averages)
        features['average_goals_per_match_pre_match'] = 2.5
        features['btts_percentage_pre_match'] = 55
        features['over_15_percentage_pre_match'] = 85
        features['over_25_percentage_pre_match'] = 60
        features['over_35_percentage_pre_match'] = 35
        
        # MARKET PROBABILITIES with enhanced logic
        home_strength = home_stats['ppg'] * FEATURE_WEIGHTS['recent_form'] * FEATURE_WEIGHTS['home_advantage']
        away_strength = away_stats['ppg'] * FEATURE_WEIGHTS['recent_form']
        strength_diff = home_strength - away_strength
        
        # More nuanced odds estimation
        if strength_diff > 0.8:  # Home very strong
            home_odds, away_odds, draw_odds = 1.6, 5.5, 4.0
        elif strength_diff > 0.4:  # Home strong
            home_odds, away_odds, draw_odds = 1.9, 4.2, 3.6
        elif strength_diff > 0.1:  # Home slight edge
            home_odds, away_odds, draw_odds = 2.3, 3.4, 3.2
        elif strength_diff < -0.8:  # Away very strong
            home_odds, away_odds, draw_odds = 5.5, 1.6, 4.0
        elif strength_diff < -0.4:  # Away strong
            home_odds, away_odds, draw_odds = 4.2, 1.9, 3.6
        elif strength_diff < -0.1:  # Away slight edge
            home_odds, away_odds, draw_odds = 3.4, 2.3, 3.2
        else:  # Very even
            home_odds, away_odds, draw_odds = 2.9, 2.9, 3.0
        
        # Calculate weighted probabilities
        features['home_win_probability'] = (1 / home_odds) * FEATURE_WEIGHTS['recent_form']
        features['draw_probability'] = 1 / draw_odds
        features['away_win_probability'] = (1 / away_odds) * FEATURE_WEIGHTS['recent_form']
        features['total_probability'] = features['home_win_probability'] + features['draw_probability'] + features['away_win_probability']
        features['bookmaker_margin'] = (features['total_probability'] - 1) * 100
        
        # GOALS BETTING with weights
        total_expected_goals = (home_xg + away_xg) * FEATURE_WEIGHTS['attacking']
        
        if total_expected_goals > 3.2:
            over25_odds = 1.5
        elif total_expected_goals > 2.8:
            over25_odds = 1.7
        elif total_expected_goals > 2.4:
            over25_odds = 2.0
        else:
            over25_odds = 2.5
        
        features['over_2.5_probability'] = 1 / over25_odds
        
        # BTTS probability with defensive consideration
        home_defense_strength = 2.0 / max(0.5, home_stats.get('goals_against_per_game', 1.0))
        away_defense_strength = 2.0 / max(0.5, away_stats.get('goals_against_per_game', 1.0))
        
        btts_likelihood = (
            min(home_xg, 2.0) * min(away_xg, 2.0) / 4.0 *
            FEATURE_WEIGHTS['attacking'] / 
            ((home_defense_strength + away_defense_strength) / 2 * FEATURE_WEIGHTS['defensive'])
        )
        features['btts_probability'] = max(0.25, min(0.85, btts_likelihood))
        
        # Game week
        features['Game Week'] = game_week
        
        # Fill any missing features with defaults
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0
        
        # Convert to DataFrame row
        feature_row = pd.DataFrame([features])
        
        # Ensure we have all the columns the model expects
        for col in self.feature_columns:
            if col not in feature_row.columns:
                feature_row[col] = 0
        
        # Reorder columns to match training data
        feature_row = feature_row[self.feature_columns]
        
        return feature_row
    
    def predict_match_integrated(self, home_team, away_team, game_week=20):
        """Integrated prediction that ensures all outputs are consistent - MAIN METHOD"""
        
        if self.trained_model is None:
            print("❌ Model not loaded! Run load_trained_model() first.")
            return None
        
        print(f"🔮 Predicting: {home_team} vs {away_team}")
        
        # Create features for this match
        match_features = self.create_upcoming_match_features(home_team, away_team, game_week)
        
        predictions = {
            'home_team': home_team,
            'away_team': away_team,
            'match': f"{home_team} vs {away_team}",
            'predictions': {},
            'betting_recommendations': []
        }
        
        try:
            # STEP 1: Get ML model predictions as baseline
            ml_predictions = self.get_ml_baseline_predictions(match_features)
            
            # STEP 2: Calculate goal predictions using improved formulas
            goal_predictions = self.predict_team_goals_improved(home_team, away_team)
            
            # STEP 3: Integrate and ensure consistency
            integrated_predictions = self.integrate_predictions(ml_predictions, goal_predictions)
            
            predictions['predictions'] = integrated_predictions
            
            # STEP 4: Generate consistent betting recommendations
            self.generate_integrated_betting_recommendations(predictions, match_features)
            
        except Exception as e:
            print(f"⚠️  Error making predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def validate_prediction_logic(self, predictions):
        """Validate that predictions make logical sense"""
        
        preds = predictions['predictions']
        issues = []
        
        # Test 1: Goals vs Over/Under consistency
        total_goals = preds['total_predicted_goals'] 
        over_25_prob = preds['over_2.5_prob']
        
        if total_goals >= 3 and over_25_prob < 0.6:
            issues.append(f"❌ Predicted {total_goals} goals but only {over_25_prob:.1%} Over 2.5")
        elif total_goals <= 2 and over_25_prob > 0.6: 
            issues.append(f"❌ Predicted {total_goals} goals but {over_25_prob:.1%} Over 2.5")
        
        # Test 2: BTTS consistency
        home_goals = preds['home_predicted_goals']
        away_goals = preds['away_predicted_goals'] 
        btts_prob = preds['btts_prob']
        
        if home_goals > 0 and away_goals > 0 and btts_prob < 0.5:
            issues.append(f"❌ Both teams score ({home_goals}-{away_goals}) but BTTS only {btts_prob:.1%}")
        elif (home_goals == 0 or away_goals == 0) and btts_prob > 0.5:
            issues.append(f"❌ Clean sheet predicted ({home_goals}-{away_goals}) but BTTS {btts_prob:.1%}")
        
        # Test 3: Outcome consistency  
        most_likely = preds['most_likely']
        if home_goals > away_goals and most_likely != 'Home Win':
            issues.append(f"❌ Home wins on score ({home_goals}-{away_goals}) but {most_likely} most likely")
        elif away_goals > home_goals and most_likely != 'Away Win':
            issues.append(f"❌ Away wins on score ({home_goals}-{away_goals}) but {most_likely} most likely")
        elif home_goals == away_goals and most_likely != 'Draw':
            issues.append(f"❌ Draw on score ({home_goals}-{away_goals}) but {most_likely} most likely")
        
        # Test 4: Probability bounds
        all_probs = [preds['home_win_prob'], preds['draw_prob'], preds['away_win_prob']]
        prob_sum = sum(all_probs)
        if abs(prob_sum - 1.0) > 0.05:
            issues.append(f"❌ Outcome probabilities sum to {prob_sum:.3f}, not 1.0")
        
        if issues:
            print("⚠️  PREDICTION VALIDATION ISSUES:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("✅ Prediction validation passed - all logic consistent")
        
        return len(issues) == 0
    
    def generate_integrated_betting_recommendations(self, predictions, match_features, confidence_threshold=0.65):
        """Generate betting recommendations for integrated predictions"""
        
        preds = predictions['predictions']
        recommendations = []
        
        # Check outcome betting with higher confidence threshold
        if 'confidence' in preds and preds['confidence'] >= confidence_threshold:
            if preds['most_likely'] == 'Home Win' and preds['home_win_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Home Win',
                    'probability': preds['home_win_prob'],
                    'confidence': preds['confidence'],
                    'reasoning': f"Integrated model predicts {preds['home_win_prob']:.1%} chance of home win",
                    'strength': 'High' if preds['confidence'] > 0.75 else 'Medium'
                })
            elif preds['most_likely'] == 'Away Win' and preds['away_win_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Away Win',
                    'probability': preds['away_win_prob'],
                    'confidence': preds['confidence'],
                    'reasoning': f"Integrated model predicts {preds['away_win_prob']:.1%} chance of away win",
                    'strength': 'High' if preds['confidence'] > 0.75 else 'Medium'
                })
            elif preds['most_likely'] == 'Draw' and preds['draw_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Draw',
                    'probability': preds['draw_prob'],
                    'confidence': preds['confidence'],
                    'reasoning': f"Integrated model predicts {preds['draw_prob']:.1%} chance of draw",
                    'strength': 'High' if preds['confidence'] > 0.75 else 'Medium'
                })
        
        # Check goals betting with consistency
        if 'over_2.5_prob' in preds:
            if preds['over_2.5_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Over 2.5 Goals',
                    'probability': preds['over_2.5_prob'],
                    'confidence': preds['over_2.5_prob'],
                    'reasoning': f"Consistent prediction: {preds['total_predicted_goals']} goals expected, {preds['over_2.5_prob']:.1%} Over 2.5",
                    'strength': 'High' if preds['over_2.5_prob'] > 0.8 else 'Medium'
                })
            elif preds['under_2.5_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Under 2.5 Goals',
                    'probability': preds['under_2.5_prob'],
                    'confidence': preds['under_2.5_prob'],
                    'reasoning': f"Consistent prediction: {preds['total_predicted_goals']} goals expected, {preds['under_2.5_prob']:.1%} Under 2.5",
                    'strength': 'High' if preds['under_2.5_prob'] > 0.8 else 'Medium'
                })
        
        # Check BTTS with consistency
        if 'btts_prob' in preds:
            if preds['btts_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Both Teams To Score - Yes',
                    'probability': preds['btts_prob'],
                    'confidence': preds['btts_prob'],
                    'reasoning': f"Consistent prediction: {preds['predicted_score']} score, {preds['btts_prob']:.1%} BTTS",
                    'strength': 'High' if preds['btts_prob'] > 0.8 else 'Medium'
                })
            elif preds['btts_no_prob'] >= confidence_threshold:
                recommendations.append({
                    'bet_type': 'Both Teams To Score - No',
                    'probability': preds['btts_no_prob'],
                    'confidence': preds['btts_no_prob'],
                    'reasoning': f"Consistent prediction: {preds['predicted_score']} score, {preds['btts_no_prob']:.1%} No BTTS",
                    'strength': 'High' if preds['btts_no_prob'] > 0.8 else 'Medium'
                })
        
        predictions['betting_recommendations'] = recommendations
    
    def predict_multiple_matches(self, matches_list, use_integrated=True):
        """Predict multiple matches at once - REQUIRED METHOD FOR WEB APP"""
        
        predictions = []
        method_name = "integrated" if use_integrated else "legacy"
        
        print(f"🔮 Predicting {len(matches_list)} matches using {method_name} method...")
        
        for i, match in enumerate(matches_list, 1):
            home_team = match.get('home_team', match.get('home'))
            away_team = match.get('away_team', match.get('away'))
            game_week = match.get('game_week', 22)
            
            print(f"   {i}. {home_team} vs {away_team}")
            
            if use_integrated:
                prediction = self.predict_match_integrated(home_team, away_team, game_week)
                if prediction:
                    # Validate each prediction
                    self.validate_prediction_logic(prediction)
            else:
                prediction = self.predict_match(home_team, away_team, game_week)
            
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
    def predict_match(self, home_team, away_team, game_week=20):
        """LEGACY METHOD - Use predict_match_integrated() for better results"""
        
        if self.trained_model is None:
            print("❌ Model not loaded! Run load_trained_model() first.")
            return None
        
        print(f"🔮 Predicting (Legacy): {home_team} vs {away_team}")
        print("⚠️  Consider using predict_match_integrated() for consistent predictions")
        
        # Create features for this match
        match_features = self.create_upcoming_match_features(home_team, away_team, game_week)
        
        predictions = {
            'home_team': home_team,
            'away_team': away_team,
            'match': f"{home_team} vs {away_team}",
            'predictions': {},
            'betting_recommendations': []
        }
        
        try:
            # Outcome prediction (1X2)
            if 'outcome_rf' in self.trained_model.models:
                outcome_probs = self.trained_model.models['outcome_rf'].predict_proba(match_features)[0]
                predictions['predictions']['home_win_prob'] = round(outcome_probs[0], 3)
                predictions['predictions']['draw_prob'] = round(outcome_probs[1], 3)
                predictions['predictions']['away_win_prob'] = round(outcome_probs[2], 3)
                
                # Most likely outcome
                most_likely_idx = np.argmax(outcome_probs)
                outcomes = ['Home Win', 'Draw', 'Away Win']
                predictions['predictions']['most_likely'] = outcomes[most_likely_idx]
                predictions['predictions']['confidence'] = round(outcome_probs[most_likely_idx], 3)
            
            # Goals predictions
            if 'over_2.5_rf' in self.trained_model.models:
                over25_prob = self.trained_model.models['over_2.5_rf'].predict_proba(match_features)[0][1]
                predictions['predictions']['over_2.5_prob'] = round(over25_prob, 3)
                predictions['predictions']['under_2.5_prob'] = round(1 - over25_prob, 3)
            
            # BTTS prediction
            if 'btts_rf' in self.trained_model.models:
                btts_prob = self.trained_model.models['btts_rf'].predict_proba(match_features)[0][1]
                predictions['predictions']['btts_prob'] = round(btts_prob, 3)
                predictions['predictions']['btts_no_prob'] = round(1 - btts_prob, 3)
            
            # Predict individual team goals - LEGACY VERSION
            self.predict_team_goals_legacy(predictions, home_team, away_team)
            
            # Generate betting recommendations
            self.generate_betting_recommendations(predictions, match_features)
            
        except Exception as e:
            print(f"⚠️  Error making predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def predict_team_goals_legacy(self, predictions, home_team, away_team):
        """Legacy goal prediction method - kept for compatibility"""
        
        # Get team stats for goal predictions
        home_stats = self.team_stats.get(home_team, {'goals_per_game': 1.2})
        away_stats = self.team_stats.get(away_team, {'goals_per_game': 1.2, 'goals_against_per_game': 1.2})
        
        # ORIGINAL FORMULA WEIGHTS
        home_attack_weight = 0.6
        away_defense_weight = 0.4
        away_attack_weight = 0.6
        home_defense_weight = 0.4
        
        # Base stats
        home_attack = home_stats.get('goals_per_game', 1.2)
        away_defense = away_stats.get('goals_against_per_game', 1.2)
        away_attack = away_stats.get('goals_per_game', 1.2)
        home_defense = home_stats.get('goals_against_per_game', 1.2)
        
        # ORIGINAL HOME ADVANTAGE
        home_advantage = 0.35
        
        # ORIGINAL GOAL PREDICTION FORMULAS
        home_expected_goals = (
            (home_attack * home_attack_weight) + 
            ((2.5 - away_defense) * away_defense_weight)
        ) + home_advantage
        
        away_expected_goals = (
            (away_attack * away_attack_weight) +
            ((2.5 - home_defense) * home_defense_weight)
        )
        
        # Apply realistic bounds per team
        home_expected_goals = max(0.3, min(3.5, home_expected_goals))
        away_expected_goals = max(0.3, min(3.5, away_expected_goals))
        
        # Convert to most likely goal counts
        home_predicted_goals = int(round(home_expected_goals))
        away_predicted_goals = int(round(away_expected_goals))
        
        # Store predictions
        predictions['predictions']['home_predicted_goals'] = home_predicted_goals
        predictions['predictions']['away_predicted_goals'] = away_predicted_goals
        predictions['predictions']['home_expected_goals'] = round(home_expected_goals, 2)
        predictions['predictions']['away_expected_goals'] = round(away_expected_goals, 2)
        predictions['predictions']['predicted_score'] = f"{home_predicted_goals}-{away_predicted_goals}"
        predictions['predictions']['total_predicted_goals'] = home_predicted_goals + away_predicted_goals
        predictions['predictions']['total_expected_goals'] = round(home_expected_goals + away_expected_goals, 2)
    
    def generate_betting_recommendations(self, predictions, match_features, confidence_threshold=0.6):
        """Generate betting recommendations based on predictions - LEGACY METHOD"""
        
        preds = predictions['predictions']
        recommendations = []
        
        # Check outcome betting
        if 'confidence' in preds and preds['confidence'] >= confidence_threshold:
            if preds['most_likely'] == 'Home Win' and preds['home_win_prob'] > 0.6:
                recommendations.append({
                    'bet_type': 'Home Win',
                    'probability': preds['home_win_prob'],
                    'confidence': preds['confidence'],
                    'reasoning': f"Model predicts {preds['home_win_prob']:.1%} chance of home win"
                })
            elif preds['most_likely'] == 'Away Win' and preds['away_win_prob'] > 0.6:
                recommendations.append({
                    'bet_type': 'Away Win',
                    'probability': preds['away_win_prob'],
                    'confidence': preds['confidence'],
                    'reasoning': f"Model predicts {preds['away_win_prob']:.1%} chance of away win"
                })
        
        # Check goals betting
        if 'over_2.5_prob' in preds:
            if preds['over_2.5_prob'] > 0.65:
                recommendations.append({
                    'bet_type': 'Over 2.5 Goals',
                    'probability': preds['over_2.5_prob'],
                    'confidence': preds['over_2.5_prob'],
                    'reasoning': f"Model predicts {preds['over_2.5_prob']:.1%} chance of 3+ goals"
                })
            elif preds['under_2.5_prob'] > 0.65:
                recommendations.append({
                    'bet_type': 'Under 2.5 Goals',
                    'probability': preds['under_2.5_prob'],
                    'confidence': preds['under_2.5_prob'],
                    'reasoning': f"Model predicts {preds['under_2.5_prob']:.1%} chance of 0-2 goals"
                })
        
        # Check BTTS
        if 'btts_prob' in preds:
            if preds['btts_prob'] > 0.65:
                recommendations.append({
                    'bet_type': 'Both Teams To Score - Yes',
                    'probability': preds['btts_prob'],
                    'confidence': preds['btts_prob'],
                    'reasoning': f"Model predicts {preds['btts_prob']:.1%} chance both teams score"
                })
            elif preds['btts_no_prob'] > 0.65:
                recommendations.append({
                    'bet_type': 'Both Teams To Score - No',
                    'probability': preds['btts_no_prob'],
                    'confidence': preds['btts_no_prob'],
                    'reasoning': f"Model predicts {preds['btts_no_prob']:.1%} chance of clean sheet"
                })
        
        predictions['betting_recommendations'] = recommendations
    
    def save_predictions(self, predictions):
        """Save predictions to files with dates included"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create predictions directory
        os.makedirs('results/predictions', exist_ok=True)
        
        # Convert to DataFrame with dates
        pred_data = []
        for pred in predictions:
            row = {
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'match': pred['match'],
                'match_date': pred.get('match_date', 'TBD'),
                'match_time': pred.get('match_time', '')
            }
            
            # Add all predictions
            if 'predictions' in pred:
                row.update(pred['predictions'])
            
            # Add recommendations summary
            if pred.get('betting_recommendations'):
                high_confidence_bets = [bet for bet in pred['betting_recommendations'] 
                                      if bet.get('strength') == 'High']
                all_bets = pred['betting_recommendations']
                
                row['recommended_bets'] = '; '.join([
                    f"{bet['bet_type']} ({bet['probability']:.1%})" 
                    for bet in all_bets
                ])
                row['high_confidence_bets'] = len(high_confidence_bets)
            else:
                row['recommended_bets'] = 'No strong recommendations'
                row['high_confidence_bets'] = 0
            
            pred_data.append(row)
        
        # Save to CSV
        pred_df = pd.DataFrame(pred_data)
        csv_file = f"results/predictions/predictions_{timestamp}.csv"
        pred_df.to_csv(csv_file, index=False)
        
        # Save detailed JSON-like text file
        txt_file = f"results/predictions/detailed_predictions_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("🔮 SERIE A MATCH PREDICTIONS - INTEGRATED VERSION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Matches: {len(predictions)}\n")
            f.write(f"Prediction Method: Integrated (ML + Goal-based with consistency checks)\n\n")
            
            for i, pred in enumerate(predictions, 1):
                match_date = pred.get('match_date', 'TBD')
                match_time = pred.get('match_time', '')
                
                f.write(f"MATCH {i}: {pred['match']}\n")
                f.write(f"Date: {match_date} {match_time}\n")
                f.write("-" * 40 + "\n")
                
                if 'predictions' in pred:
                    p = pred['predictions']
                    
                    # Main predictions
                    f.write(f"🏠 Home Win: {p.get('home_win_prob', 0):.1%}\n")
                    f.write(f"🤝 Draw: {p.get('draw_prob', 0):.1%}\n")
                    f.write(f"✈️  Away Win: {p.get('away_win_prob', 0):.1%}\n")
                    f.write(f"🎯 Most Likely: {p.get('most_likely', 'Unknown')} ({p.get('confidence', 0):.1%} confidence)\n\n")
                    
                    # Goals predictions
                    if 'predicted_score' in p:
                        f.write(f"⚽ Predicted Score: {p.get('predicted_score', 'N/A')}\n")
                        f.write(f"🏠 Home: {p.get('home_predicted_goals', 0)} goals ({p.get('home_expected_goals', 0)} xG)\n")
                        f.write(f"✈️  Away: {p.get('away_predicted_goals', 0)} goals ({p.get('away_expected_goals', 0)} xG)\n")
                        f.write(f"📊 Total: {p.get('total_predicted_goals', 0)} goals ({p.get('total_expected_goals', 0)} xG)\n\n")
                    
                    # Betting markets
                    f.write(f"⚽ Over 2.5: {p.get('over_2.5_prob', 0):.1%} | Under 2.5: {p.get('under_2.5_prob', 0):.1%}\n")
                    f.write(f"🥅 BTTS Yes: {p.get('btts_prob', 0):.1%} | BTTS No: {p.get('btts_no_prob', 0):.1%}\n\n")
                
                if pred.get('betting_recommendations'):
                    f.write("💰 BETTING RECOMMENDATIONS:\n")
                    for bet in pred['betting_recommendations']:
                        strength_emoji = "🔥" if bet.get('strength') == 'High' else "⭐"
                        f.write(f"   {strength_emoji} {bet['bet_type']} - {bet['probability']:.1%} confidence\n")
                        f.write(f"     {bet['reasoning']}\n")
                        f.write(f"     Strength: {bet.get('strength', 'Medium')}\n")
                else:
                    f.write("💰 No strong betting recommendations\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"💾 Predictions saved:")
        print(f"   📊 CSV: {csv_file}")
        print(f"   📄 Details: {txt_file}")
        
        return csv_file, txt_file

# Example usage functions - UPDATED
def predict_next_weekend():
    """Example: Predict next weekend's matches using integrated method"""
    
    # Sample upcoming matches - replace with real fixture data
    upcoming_matches = [
        {'home_team': 'Bologna', 'away_team': 'Como', 'game_week': 22},
        {'home_team': 'Juventus', 'away_team': 'Inter', 'game_week': 22},
        {'home_team': 'AC Milan', 'away_team': 'Napoli', 'game_week': 22},
        {'home_team': 'Roma', 'away_team': 'Lazio', 'game_week': 22},
        {'home_team': 'Atalanta', 'away_team': 'Fiorentina', 'game_week': 22}
    ]
    
    predictor = FutureGamePredictor()
    predictor.load_trained_model()
    
    # Use integrated method for consistent predictions
    predictions = predictor.predict_multiple_matches(upcoming_matches, use_integrated=True)
    
    # Save results
    predictor.save_predictions(predictions)
    
    # Print summary
    print("\n🔮 INTEGRATED PREDICTION SUMMARY:")
    print("=" * 50)
    
    high_confidence_count = 0
    total_recommendations = 0
    
    for pred in predictions:
        print(f"\n{pred['match']}")
        if 'predictions' in pred:
            p = pred['predictions']
            print(f"   Most likely: {p.get('most_likely', 'Unknown')} ({p.get('confidence', 0):.1%})")
            print(f"   Predicted score: {p.get('predicted_score', 'N/A')}")
            
            if pred.get('betting_recommendations'):
                high_conf_bets = [bet for bet in pred['betting_recommendations'] 
                                if bet.get('strength') == 'High']
                total_bets = len(pred['betting_recommendations'])
                high_conf_count = len(high_conf_bets)
                
                print(f"   Recommended bets: {total_bets} total, {high_conf_count} high confidence")
                total_recommendations += total_bets
                high_confidence_count += high_conf_count
                
                # Show best recommendation
                if high_conf_bets:
                    best_bet = max(high_conf_bets, key=lambda x: x['probability'])
                    print(f"   🔥 Best bet: {best_bet['bet_type']} ({best_bet['probability']:.1%})")
    
    print(f"\n📊 SUMMARY STATS:")
    print(f"   Total matches predicted: {len(predictions)}")
    print(f"   Total betting recommendations: {total_recommendations}")
    print(f"   High confidence bets: {high_confidence_count}")
    print(f"   Average confidence per match: {total_recommendations/len(predictions):.1f} bets")

def test_integration_fixes():
    """Test the integration fixes with Bologna vs Como example"""
    
    predictor = FutureGamePredictor()
    predictor.load_trained_model()
    
    print("🧪 TESTING INTEGRATION FIXES")
    print("=" * 40)
    
    # Test the problematic Bologna vs Como match
    print("\n1️⃣ Testing Bologna vs Como (the problematic case):")
    prediction = predictor.predict_match_integrated("Bologna", "Como")
    
    if prediction:
        predictor.validate_prediction_logic(prediction)
        p = prediction['predictions']
        
        print(f"\nResults:")
        print(f"   Predicted Score: {p.get('predicted_score', 'N/A')}")
        print(f"   Total Goals: {p.get('total_predicted_goals', 0)}")
        print(f"   Over 2.5 Goals: {p.get('over_2.5_prob', 0):.1%}")
        print(f"   BTTS: {p.get('btts_prob', 0):.1%}")
        print(f"   Most Likely: {p.get('most_likely', 'Unknown')} ({p.get('confidence', 0):.1%})")
    
    print("\n✅ Integration test complete!")

if __name__ == "__main__":
    # Run the integrated prediction system
    predict_next_weekend()
    
    # Uncomment to test the fixes
    # test_integration_fixes()
    
    def get_ml_baseline_predictions(self, match_features):
        """Get baseline predictions from trained ML models"""
        ml_preds = {}
        
        try:
            # Outcome prediction (1X2)
            if 'outcome_rf' in self.trained_model.models:
                outcome_probs = self.trained_model.models['outcome_rf'].predict_proba(match_features)[0]
                ml_preds['ml_home_win_prob'] = outcome_probs[0]
                ml_preds['ml_draw_prob'] = outcome_probs[1] 
                ml_preds['ml_away_win_prob'] = outcome_probs[2]
            
            # Goals predictions
            if 'over_2.5_rf' in self.trained_model.models:
                ml_preds['ml_over_2.5_prob'] = self.trained_model.models['over_2.5_rf'].predict_proba(match_features)[0][1]
            
            # BTTS prediction  
            if 'btts_rf' in self.trained_model.models:
                ml_preds['ml_btts_prob'] = self.trained_model.models['btts_rf'].predict_proba(match_features)[0][1]
                
        except Exception as e:
            print(f"⚠️  ML prediction error: {e}")
            # Fallback to neutral probabilities
            ml_preds = {
                'ml_home_win_prob': 0.4, 'ml_draw_prob': 0.3, 'ml_away_win_prob': 0.3,
                'ml_over_2.5_prob': 0.5, 'ml_btts_prob': 0.5
            }
        
        return ml_preds
    
    def predict_team_goals_improved(self, home_team, away_team):
        """FIXED: Improved goal prediction with REALISTIC scoring"""
        
        # Get team stats
        home_stats = self.team_stats.get(home_team, {'goals_per_game': 1.2, 'goals_against_per_game': 1.2, 'ppg': 1.5})
        away_stats = self.team_stats.get(away_team, {'goals_per_game': 1.2, 'goals_against_per_game': 1.2, 'ppg': 1.5})
        
        # REALISTIC FORMULA WEIGHTS - Much more conservative
        home_attack_weight = 0.8    # Increased to use more of actual stats
        away_defense_weight = 0.2   # Reduced defense impact  
        away_attack_weight = 0.8    # Increased to use more of actual stats
        home_defense_weight = 0.2   # Reduced defense impact
        home_advantage = 0.15       # REDUCED from 0.35 to 0.15
        
        # Base expected goals calculation - USE RAW STATS
        home_attack = home_stats.get('goals_per_game', 1.2)
        away_defense = away_stats.get('goals_against_per_game', 1.2) 
        away_attack = away_stats.get('goals_per_game', 1.2)
        home_defense = home_stats.get('goals_against_per_game', 1.2)
        
        # REALISTIC EXPECTED GOALS - Closer to actual team performance
        home_expected_goals = (
            (home_attack * home_attack_weight) + 
            ((1.8 - away_defense) * away_defense_weight)  # REDUCED from 2.5 to 1.8
        ) + home_advantage
        
        away_expected_goals = (
            (away_attack * away_attack_weight) +
            ((1.8 - home_defense) * home_defense_weight)  # REDUCED from 2.5 to 1.8
        )
        
        # MINIMAL form adjustment
        form_impact = 0.08  # REDUCED from 0.2 to 0.08
        if 'ppg' in home_stats and 'ppg' in away_stats:
            home_form = (home_stats['ppg'] - 1.5) * form_impact
            away_form = (away_stats['ppg'] - 1.5) * form_impact
            home_expected_goals += home_form
            away_expected_goals += away_form
        
        # TIGHT realistic bounds
        home_expected_goals = max(0.3, min(2.5, home_expected_goals))  # REDUCED max from 3.5 to 2.5
        away_expected_goals = max(0.3, min(2.5, away_expected_goals))  # REDUCED max from 3.5 to 2.5
        
        # Calculate goal-based probabilities
        total_xg = home_expected_goals + away_expected_goals
        
        # REALISTIC OUTCOME PROBABILITIES
        goal_diff = home_expected_goals - away_expected_goals
        
        if goal_diff > 0.5:
            home_win_prob = 0.55
            draw_prob = 0.25  
            away_win_prob = 0.20
        elif goal_diff > 0.2:
            home_win_prob = 0.42
            draw_prob = 0.33
            away_win_prob = 0.25
        elif goal_diff < -0.5:
            home_win_prob = 0.20
            draw_prob = 0.25
            away_win_prob = 0.55
        elif goal_diff < -0.2:
            home_win_prob = 0.25  
            draw_prob = 0.33
            away_win_prob = 0.42
        else:  # Even match
            home_win_prob = 0.33
            draw_prob = 0.34
            away_win_prob = 0.33
        
        # REALISTIC OVER/UNDER
        if total_xg >= 3.0:
            over_2_5_prob = 0.70
        elif total_xg >= 2.5:  
            over_2_5_prob = 0.55
        elif total_xg >= 2.0:
            over_2_5_prob = 0.40
        elif total_xg >= 1.5:
            over_2_5_prob = 0.25
        else:
            over_2_5_prob = 0.15
        
        # REALISTIC BTTS
        btts_prob = min(0.80, max(0.15, 
            (min(home_expected_goals, 1.5) * min(away_expected_goals, 1.5)) / 2.25
        ))
        
        return {
            'home_expected_goals': round(home_expected_goals, 2),
            'away_expected_goals': round(away_expected_goals, 2), 
            'total_expected_goals': round(total_xg, 2),
            'home_predicted_goals': int(round(home_expected_goals)),
            'away_predicted_goals': int(round(away_expected_goals)),
            'goal_based_home_win_prob': home_win_prob,
            'goal_based_draw_prob': draw_prob, 
            'goal_based_away_win_prob': away_win_prob,
            'goal_based_over_2_5_prob': over_2_5_prob,
            'goal_based_btts_prob': btts_prob
        }
    
    def integrate_predictions(self, ml_predictions, goal_predictions):
        """Integrate ML and goal-based predictions with consistency checks"""
        
        # BLENDING WEIGHTS - adjust these to balance ML vs goal-based predictions
        ML_WEIGHT = 0.4      # How much to trust the ML models
        GOAL_WEIGHT = 0.6    # How much to trust the goal-based calculations
        
        integrated = {}
        
        # Blend outcome probabilities
        integrated['home_win_prob'] = round(
            (ml_predictions.get('ml_home_win_prob', 0.33) * ML_WEIGHT) + 
            (goal_predictions['goal_based_home_win_prob'] * GOAL_WEIGHT), 3
        )
        
        integrated['draw_prob'] = round(
            (ml_predictions.get('ml_draw_prob', 0.33) * ML_WEIGHT) + 
            (goal_predictions['goal_based_draw_prob'] * GOAL_WEIGHT), 3
        )
        
        integrated['away_win_prob'] = round(
            (ml_predictions.get('ml_away_win_prob', 0.33) * ML_WEIGHT) + 
            (goal_predictions['goal_based_away_win_prob'] * GOAL_WEIGHT), 3
        )
        
        # Normalize probabilities to sum to 1.0
        total_prob = integrated['home_win_prob'] + integrated['draw_prob'] + integrated['away_win_prob']
        integrated['home_win_prob'] = round(integrated['home_win_prob'] / total_prob, 3)
        integrated['draw_prob'] = round(integrated['draw_prob'] / total_prob, 3) 
        integrated['away_win_prob'] = round(integrated['away_win_prob'] / total_prob, 3)
        
        # Determine most likely outcome
        outcome_probs = [integrated['home_win_prob'], integrated['draw_prob'], integrated['away_win_prob']]
        most_likely_idx = np.argmax(outcome_probs)
        outcomes = ['Home Win', 'Draw', 'Away Win']
        integrated['most_likely'] = outcomes[most_likely_idx]
        integrated['confidence'] = round(outcome_probs[most_likely_idx], 3)
        
        # Blend over/under probabilities
        integrated['over_2.5_prob'] = round(
            (ml_predictions.get('ml_over_2.5_prob', 0.5) * ML_WEIGHT) + 
            (goal_predictions['goal_based_over_2_5_prob'] * GOAL_WEIGHT), 3
        )
        integrated['under_2.5_prob'] = round(1 - integrated['over_2.5_prob'], 3)
        
        # Blend BTTS probabilities  
        integrated['btts_prob'] = round(
            (ml_predictions.get('ml_btts_prob', 0.5) * ML_WEIGHT) + 
            (goal_predictions['goal_based_btts_prob'] * GOAL_WEIGHT), 3
        )
        integrated['btts_no_prob'] = round(1 - integrated['btts_prob'], 3)
        
        # Add goal predictions
        integrated['home_predicted_goals'] = goal_predictions['home_predicted_goals']
        integrated['away_predicted_goals'] = goal_predictions['away_predicted_goals'] 
        integrated['predicted_score'] = f"{goal_predictions['home_predicted_goals']}-{goal_predictions['away_predicted_goals']}"
        integrated['home_expected_goals'] = goal_predictions['home_expected_goals']
        integrated['away_expected_goals'] = goal_predictions['away_expected_goals']
        integrated['total_expected_goals'] = goal_predictions['total_expected_goals']
        integrated['total_predicted_goals'] = goal_predictions['home_predicted_goals'] + goal_predictions['away_predicted_goals']
        
        # CONSISTENCY CHECKS - Fix obvious contradictions
        integrated = self.apply_consistency_checks(integrated)
        
        return integrated
    
    def apply_consistency_checks(self, predictions):
        """Apply logic checks to ensure predictions make sense"""
        
        # Check 1: If predicted score is over 2.5, over 2.5 probability should be high
        if predictions['total_predicted_goals'] >= 3:
            predictions['over_2.5_prob'] = max(0.75, predictions['over_2.5_prob'])
            predictions['under_2.5_prob'] = 1 - predictions['over_2.5_prob']
            print(f"✅ Consistency fix: Predicted {predictions['total_predicted_goals']} goals, boosted Over 2.5 to {predictions['over_2.5_prob']:.1%}")
        
        # Check 2: If both teams predicted to score, BTTS should be high
        if predictions['home_predicted_goals'] > 0 and predictions['away_predicted_goals'] > 0:
            predictions['btts_prob'] = max(0.65, predictions['btts_prob'])
            predictions['btts_no_prob'] = 1 - predictions['btts_prob']
            print(f"✅ Consistency fix: Both teams score in prediction, boosted BTTS to {predictions['btts_prob']:.1%}")
        
        # Check 3: If one team predicted to score 0, BTTS should be low  
        elif predictions['home_predicted_goals'] == 0 or predictions['away_predicted_goals'] == 0:
            predictions['btts_prob'] = min(0.35, predictions['btts_prob'])
            predictions['btts_no_prob'] = 1 - predictions['btts_prob']
            print(f"✅ Consistency fix: Clean sheet predicted, reduced BTTS to {predictions['btts_prob']:.1%}")
        
        # Check 4: Outcome probabilities should match predicted score
        home_goals = predictions['home_predicted_goals']
        away_goals = predictions['away_predicted_goals']
        
        if home_goals > away_goals:  # Home win predicted
            if predictions['most_likely'] != 'Home Win':
                print(f"✅ Consistency fix: Score predicts home win ({home_goals}-{away_goals}), adjusting outcome")
                predictions['home_win_prob'] = max(0.6, predictions['home_win_prob'])
                predictions['most_likely'] = 'Home Win'
                predictions['confidence'] = predictions['home_win_prob']
                
        elif away_goals > home_goals:  # Away win predicted  
            if predictions['most_likely'] != 'Away Win':
                print(f"✅ Consistency fix: Score predicts away win ({home_goals}-{away_goals}), adjusting outcome")
                predictions['away_win_prob'] = max(0.6, predictions['away_win_prob']) 
                predictions['most_likely'] = 'Away Win'
                predictions['confidence'] = predictions['away_win_prob']
                
        else:  # Draw predicted
            if predictions['most_likely'] != 'Draw':
                print(f"✅ Consistency fix: Score predicts draw ({home_goals}-{away_goals}), adjusting outcome")
                predictions['draw_prob'] = max(0.5, predictions['draw_prob'])
                predictions['most_likely'] = 'Draw' 
                predictions['confidence'] = predictions['draw_prob']
        
        # Re-normalize outcome probabilities after adjustments
        total = predictions['home_win_prob'] + predictions['draw_prob'] + predictions['away_win_prob']
        predictions['home_win_prob'] = round(predictions['home_win_prob'] / total, 3)
        predictions['draw_prob'] = round(predictions['draw_prob'] / total, 3)
        predictions['away_win_prob'] = round(predictions['away_win_prob'] / total, 3)
        
        return predictions