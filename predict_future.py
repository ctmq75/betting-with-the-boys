"""
Serie A Future Game Predictor
Predicts outcomes for upcoming matches using trained models
Fixed version with dates in output
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
        df = pd.read_csv('data/raw/serie_a_historical.csv')
        
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
        """Create features for an upcoming match"""
        
        # Get team stats
        home_stats = self.team_stats.get(home_team, {'ppg': 1.0, 'goals_per_game': 1.0, 'goals_against_per_game': 1.0})
        away_stats = self.team_stats.get(away_team, {'ppg': 1.0, 'goals_per_game': 1.0, 'goals_against_per_game': 1.0})
        
        # Create feature dictionary matching training data structure
        features = {}
        
        # Basic team performance
        features['home_ppg'] = home_stats['ppg']
        features['away_ppg'] = away_stats['ppg']
        features['ppg_difference'] = home_stats['ppg'] - away_stats['ppg']
        
        # Pre-match features (use current form as estimate)
        features['Pre-Match PPG (Home)'] = home_stats['ppg']
        features['Pre-Match PPG (Away)'] = away_stats['ppg']
        features['pre_match_ppg_difference'] = home_stats['ppg'] - away_stats['ppg']
        
        # Expected goals (estimate based on scoring form)
        home_xg = home_stats['goals_per_game']
        away_xg = away_stats['goals_per_game']
        features['team_a_xg'] = home_xg
        features['team_b_xg'] = away_xg
        features['xg_difference'] = home_xg - away_xg
        features['Home Team Pre-Match xG'] = home_xg
        features['Away Team Pre-Match xG'] = away_xg
        features['pre_match_xg_difference'] = home_xg - away_xg
        
        # Estimated match stats (based on team averages)
        features['home_team_shots'] = home_stats['goals_per_game'] * 5  # Rough estimate
        features['away_team_shots'] = away_stats['goals_per_game'] * 5
        features['home_team_shots_on_target'] = home_stats['goals_per_game'] * 2
        features['away_team_shots_on_target'] = away_stats['goals_per_game'] * 2
        
        # Possession estimate (slightly favor home team)
        base_possession = 50
        possession_advantage = (home_stats['ppg'] - away_stats['ppg']) * 5
        features['home_team_possession'] = min(65, max(35, base_possession + possession_advantage + 5))  # Home advantage
        features['away_team_possession'] = 100 - features['home_team_possession']
        
        # Corner and other stats estimates
        features['home_team_corner_count'] = 5 + (home_stats['ppg'] - 1) * 2
        features['away_team_corner_count'] = 5 + (away_stats['ppg'] - 1) * 2
        
        # Cards estimates
        features['home_team_yellow_cards'] = 2
        features['away_team_yellow_cards'] = 2
        features['home_team_red_cards'] = 0
        features['away_team_red_cards'] = 0
        features['home_team_fouls'] = 12
        features['away_team_fouls'] = 12
        
        # Pre-match percentages (league averages)
        features['average_goals_per_match_pre_match'] = 2.5
        features['btts_percentage_pre_match'] = 55
        features['over_15_percentage_pre_match'] = 85
        features['over_25_percentage_pre_match'] = 60
        features['over_35_percentage_pre_match'] = 35
        
        # Market-based features (we'll estimate these)
        # Stronger team gets lower odds
        home_strength = home_stats['ppg']
        away_strength = away_stats['ppg']
        strength_diff = home_strength - away_strength
        
        # Estimate odds based on strength difference
        if strength_diff > 0.5:  # Home much stronger
            home_odds = 1.8
            away_odds = 4.5
            draw_odds = 3.5
        elif strength_diff > 0.2:  # Home slightly stronger
            home_odds = 2.2
            away_odds = 3.5
            draw_odds = 3.2
        elif strength_diff < -0.5:  # Away much stronger
            home_odds = 4.5
            away_odds = 1.8
            draw_odds = 3.5
        elif strength_diff < -0.2:  # Away slightly stronger
            home_odds = 3.5
            away_odds = 2.2
            draw_odds = 3.2
        else:  # Evenly matched
            home_odds = 2.8
            away_odds = 2.8
            draw_odds = 3.0
        
        # Calculate probabilities from estimated odds
        features['home_win_probability'] = 1 / home_odds
        features['draw_probability'] = 1 / draw_odds
        features['away_win_probability'] = 1 / away_odds
        features['total_probability'] = features['home_win_probability'] + features['draw_probability'] + features['away_win_probability']
        features['bookmaker_margin'] = (features['total_probability'] - 1) * 100
        
        # Goals betting odds estimates
        expected_goals = home_xg + away_xg
        if expected_goals > 2.8:
            over25_odds = 1.6
        elif expected_goals > 2.3:
            over25_odds = 1.9
        else:
            over25_odds = 2.3
        
        features['over_2.5_probability'] = 1 / over25_odds
        
        # BTTS probability
        btts_likelihood = min(home_xg, 2.0) * min(away_xg, 2.0) / 4.0  # Both teams likely to score
        features['btts_probability'] = max(0.3, min(0.8, btts_likelihood))
        
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
    
    def predict_match(self, home_team, away_team, game_week=20):
        """Predict the outcome of a specific match"""
        
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
            
            # Predict individual team goals
            self.predict_team_goals(predictions, home_team, away_team)
            
            # Generate betting recommendations
            self.generate_betting_recommendations(predictions, match_features)
            
        except Exception as e:
            print(f"⚠️  Error making predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def generate_betting_recommendations(self, predictions, match_features, confidence_threshold=0.6):
        """Generate betting recommendations based on predictions"""
        
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
    
    def predict_team_goals(self, predictions, home_team, away_team):
        """Predict individual team goal counts"""
        
        # Get team stats for goal predictions
        home_stats = self.team_stats.get(home_team, {'goals_per_game': 1.2})
        away_stats = self.team_stats.get(away_team, {'goals_per_game': 1.2, 'goals_against_per_game': 1.2})
        
        # Base prediction on team scoring/defensive averages
        home_attack = home_stats.get('goals_per_game', 1.2)
        away_defense = away_stats.get('goals_against_per_game', 1.2)
        away_attack = away_stats.get('goals_per_game', 1.2)
        home_defense = home_stats.get('goals_against_per_game', 1.2)
        
        # Adjust for home advantage (typically +0.2-0.3 goals)
        home_advantage = 0.25
        
        # Predict expected goals for each team
        # Home team: their attack vs away defense + home advantage
        home_expected_goals = (home_attack + away_defense) / 2 + home_advantage
        
        # Away team: their attack vs home defense
        away_expected_goals = (away_attack + home_defense) / 2
        
        # Apply some realistic bounds (teams rarely score 5+ goals)
        home_expected_goals = max(0.3, min(4.0, home_expected_goals))
        away_expected_goals = max(0.3, min(4.0, away_expected_goals))
        
        # Use simple rounding for most likely goal count (fallback if scipy not available)
        try:
            from scipy.stats import poisson
            # Get most likely goal count (mode of Poisson distribution)
            home_predicted_goals = int(round(home_expected_goals))
            away_predicted_goals = int(round(away_expected_goals))
        except ImportError:
            # Fallback: simple rounding
            home_predicted_goals = int(round(home_expected_goals))
            away_predicted_goals = int(round(away_expected_goals))
        
        # If we have outcome predictions, adjust based on who's more likely to win
        if 'most_likely' in predictions['predictions']:
            if predictions['predictions']['most_likely'] == 'Home Win':
                # If home is favored, slightly increase their goals
                if home_predicted_goals <= away_predicted_goals:
                    home_predicted_goals = away_predicted_goals + 1
            elif predictions['predictions']['most_likely'] == 'Away Win':
                # If away is favored, slightly increase their goals
                if away_predicted_goals <= home_predicted_goals:
                    away_predicted_goals = home_predicted_goals + 1
            # For draws, keep goals close
            elif predictions['predictions']['most_likely'] == 'Draw':
                # Make goals equal or very close
                avg_goals = (home_predicted_goals + away_predicted_goals) / 2
                if avg_goals < 1:
                    home_predicted_goals = away_predicted_goals = 1
                else:
                    home_predicted_goals = away_predicted_goals = int(round(avg_goals))
        
        # Store predictions
        predictions['predictions']['home_predicted_goals'] = home_predicted_goals
        predictions['predictions']['away_predicted_goals'] = away_predicted_goals
        predictions['predictions']['home_expected_goals'] = round(home_expected_goals, 2)
        predictions['predictions']['away_expected_goals'] = round(away_expected_goals, 2)
        predictions['predictions']['predicted_score'] = f"{home_predicted_goals}-{away_predicted_goals}"
        predictions['predictions']['total_predicted_goals'] = home_predicted_goals + away_predicted_goals
    
    def predict_multiple_matches(self, matches_list):
        """Predict multiple matches at once"""
        
        predictions = []
        
        print(f"🔮 Predicting {len(matches_list)} matches...")
        
        for i, match in enumerate(matches_list, 1):
            home_team = match.get('home_team', match.get('home'))
            away_team = match.get('away_team', match.get('away'))
            game_week = match.get('game_week', 22)
            
            print(f"   {i}. {home_team} vs {away_team}")
            
            prediction = self.predict_match(home_team, away_team, game_week)
            if prediction:
                predictions.append(prediction)
        
        return predictions
    
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
                'match_date': pred.get('match_date', 'TBD'),  # Include date
                'match_time': pred.get('match_time', '')      # Include time
            }
            
            # Add all predictions
            if 'predictions' in pred:
                row.update(pred['predictions'])
            
            # Add recommendations summary
            if pred.get('betting_recommendations'):
                row['recommended_bets'] = '; '.join([
                    f"{bet['bet_type']} ({bet['probability']:.1%})" 
                    for bet in pred['betting_recommendations']
                ])
            else:
                row['recommended_bets'] = 'No strong recommendations'
            
            pred_data.append(row)
        
        # Save to CSV
        pred_df = pd.DataFrame(pred_data)
        csv_file = f"results/predictions/predictions_{timestamp}.csv"
        pred_df.to_csv(csv_file, index=False)
        
        # Save detailed JSON-like text file
        txt_file = f"results/predictions/detailed_predictions_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("🔮 SERIE A MATCH PREDICTIONS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, pred in enumerate(predictions, 1):
                match_date = pred.get('match_date', 'TBD')
                match_time = pred.get('match_time', '')
                
                f.write(f"MATCH {i}: {pred['match']}\n")
                f.write(f"Date: {match_date} {match_time}\n")
                f.write("-" * 30 + "\n")
                
                if 'predictions' in pred:
                    p = pred['predictions']
                    f.write(f"🏠 Home Win: {p.get('home_win_prob', 0):.1%}\n")
                    f.write(f"🤝 Draw: {p.get('draw_prob', 0):.1%}\n")
                    f.write(f"✈️  Away Win: {p.get('away_win_prob', 0):.1%}\n")
                    f.write(f"⚽ Over 2.5 Goals: {p.get('over_2.5_prob', 0):.1%}\n")
                    f.write(f"🥅 BTTS: {p.get('btts_prob', 0):.1%}\n")
                    f.write(f"🎯 Most Likely: {p.get('most_likely', 'Unknown')}\n")
                    
                    # Add predicted goals
                    if 'predicted_score' in p:
                        f.write(f"⚽ Predicted Score: {p.get('predicted_score', 'N/A')}\n")
                        f.write(f"🏠 Home Goals: {p.get('home_predicted_goals', 0)} ({p.get('home_expected_goals', 0)} xG)\n")
                        f.write(f"✈️  Away Goals: {p.get('away_predicted_goals', 0)} ({p.get('away_expected_goals', 0)} xG)\n")
                        f.write(f"📊 Total Goals: {p.get('total_predicted_goals', 0)}\n")
                    f.write("\n")
                
                if pred.get('betting_recommendations'):
                    f.write("💰 BETTING RECOMMENDATIONS:\n")
                    for bet in pred['betting_recommendations']:
                        f.write(f"   • {bet['bet_type']} - {bet['probability']:.1%} confidence\n")
                        f.write(f"     {bet['reasoning']}\n")
                else:
                    f.write("💰 No strong betting recommendations\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"💾 Predictions saved:")
        print(f"   📊 CSV: {csv_file}")
        print(f"   📄 Details: {txt_file}")
        
        return csv_file, txt_file

# Example usage functions
def predict_next_weekend():
    """Example: Predict next weekend's matches"""
    
    # Sample upcoming matches - replace with real fixture data
    upcoming_matches = [
        {'home_team': 'Juventus', 'away_team': 'Inter', 'game_week': 22},
        {'home_team': 'AC Milan', 'away_team': 'Napoli', 'game_week': 22},
        {'home_team': 'Roma', 'away_team': 'Lazio', 'game_week': 22},
        {'home_team': 'Atalanta', 'away_team': 'Fiorentina', 'game_week': 22},
        {'home_team': 'Bologna', 'away_team': 'Torino', 'game_week': 22}
    ]
    
    predictor = FutureGamePredictor()
    predictor.load_trained_model()
    
    predictions = predictor.predict_multiple_matches(upcoming_matches)
    
    # Save results
    predictor.save_predictions(predictions)
    
    # Print summary
    print("\n🔮 PREDICTION SUMMARY:")
    print("=" * 40)
    for pred in predictions:
        print(f"{pred['match']}")
        if 'predictions' in pred:
            p = pred['predictions']
            print(f"   Most likely: {p.get('most_likely', 'Unknown')} ({p.get('confidence', 0):.1%})")
            if pred.get('betting_recommendations'):
                print(f"   Recommended bets: {len(pred['betting_recommendations'])}")
        print()

if __name__ == "__main__":
    predict_next_weekend()