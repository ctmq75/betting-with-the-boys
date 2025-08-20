"""
Complete Serie A Future Match Predictor
Fixed version with dates in output
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta
from predict_future import FutureGamePredictor

def get_real_teams():
    """Get actual team names from your historical data"""
    try:
        df = pd.read_csv('data/raw/serie_a_historical.csv')
        
        # Get all unique team names
        home_teams = set(df['home_team'].unique())
        away_teams = set(df['away_team'].unique())
        all_teams = sorted(list(home_teams | away_teams))
        
        print(f"📋 Found {len(all_teams)} teams in your data:")
        for i, team in enumerate(all_teams, 1):
            print(f"   {i:2d}. {team}")
        
        return all_teams
    
    except Exception as e:
        print(f"❌ Error reading team data: {e}")
        # Fallback to common Serie A teams
        return [
            'Atalanta', 'Bologna', 'Fiorentina', 'Inter', 'Juventus',
            'Lazio', 'AC Milan', 'Napoli', 'Roma', 'Torino'
        ]

def load_fixtures_from_csv(csv_file_path):
    """Load real future fixtures from CSV file"""
    try:
        print(f"📁 Loading fixtures from: {csv_file_path}")
        
        # Read the CSV
        fixtures_df = pd.read_csv(csv_file_path)
        
        print(f"✅ Found {len(fixtures_df)} matches in CSV")
        print("📋 CSV columns:", list(fixtures_df.columns))
        
        # Your CSV has: date, home_team, away_team
        home_col = 'home_team'
        away_col = 'away_team' 
        date_col = 'date'
        
        print(f"\n🔍 Using columns:")
        print(f"   Home Team: {home_col}")
        print(f"   Away Team: {away_col}")
        print(f"   Date: {date_col}")
        
        # Convert to our format
        fixtures = []
        
        for idx, row in fixtures_df.iterrows():
            try:
                home_team = str(row[home_col]).strip()
                away_team = str(row[away_col]).strip()
                match_date = str(row[date_col]).strip()
                
                fixture = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'match_date': match_date,
                    'match_time': '15:00',  # Default time
                    'game_week': int(row.get('game_week', 22))
                }
                
                fixtures.append(fixture)
                
            except Exception as e:
                print(f"⚠️  Error processing row {idx}: {e}")
                continue
        
        print(f"\n✅ Successfully loaded {len(fixtures)} fixtures from CSV")
        
        # Show first few fixtures
        print(f"\n📋 First 3 fixtures loaded:")
        for i, fixture in enumerate(fixtures[:3], 1):
            print(f"   {i}. {fixture['home_team']} vs {fixture['away_team']} ({fixture['match_date']} {fixture['match_time']})")
        
        return fixtures
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_team_recent_form(team_name):
    """Get recent form for a specific team"""
    try:
        df = pd.read_csv('data/raw/serie_a_historical.csv')
        df['date_GMT'] = pd.to_datetime(df['date_GMT'])
        df = df.sort_values('date_GMT')
        
        # Get last 5 home and away matches
        home_matches = df[df['home_team'] == team_name].tail(5)
        away_matches = df[df['away_team'] == team_name].tail(5)
        
        points = 0
        games = 0
        goals_for = 0
        goals_against = 0
        
        # Count home matches
        for _, match in home_matches.iterrows():
            games += 1
            gf = match['home_team_goal_count']
            ga = match['away_team_goal_count']
            goals_for += gf
            goals_against += ga
            
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        
        # Count away matches
        for _, match in away_matches.iterrows():
            games += 1
            gf = match['away_team_goal_count']
            ga = match['home_team_goal_count']
            goals_for += gf
            goals_against += ga
            
            if gf > ga:
                points += 3
            elif gf == ga:
                points += 1
        
        if games > 0:
            ppg = points / games
            gpg = goals_for / games
            gapg = goals_against / games
            form = "Excellent" if ppg > 2.5 else "Good" if ppg > 1.5 else "Average" if ppg > 1.0 else "Poor"
            
            return {
                'ppg': ppg,
                'goals_per_game': gpg,
                'goals_against_per_game': gapg,
                'form': form,
                'games': games
            }
    except:
        pass
    
    return {'ppg': 1.5, 'goals_per_game': 1.2, 'goals_against_per_game': 1.2, 'form': 'Unknown', 'games': 0}

def show_team_analysis(fixtures):
    """Show team form analysis before predictions"""
    print("\n📊 TEAM FORM ANALYSIS")
    print("="*50)
    
    teams_analyzed = set()
    
    for fixture in fixtures:
        for team in [fixture['home_team'], fixture['away_team']]:
            if team not in teams_analyzed:
                form = get_team_recent_form(team)
                print(f"{team}: {form['form']} form ({form['ppg']:.1f} PPG, {form['goals_per_game']:.1f} GPG)")
                teams_analyzed.add(team)

def main():
    print("🔮 Serie A Future Match Predictor")
    print("=" * 50)
    
    # Get real teams from data
    teams = get_real_teams()
    
    # FORCE load the CSV - no options, no fallbacks
    csv_file = 'data/upcoming_matches.csv'
    print(f"\n📁 FORCING load of: {csv_file}")
    
    fixtures = load_fixtures_from_csv(csv_file)
    
    if not fixtures:
        print("❌ FAILED to load CSV. STOPPING.")
        return
    
    print(f"\n🎯 Loaded {len(fixtures)} REAL fixtures:")
    for i, fixture in enumerate(fixtures, 1):
        match_date = fixture.get('match_date', 'TBD')
        match_time = fixture.get('match_time', '')
        print(f"   {i}. {fixture['home_team']} vs {fixture['away_team']} ({match_date} {match_time})")
    
    # Show team form analysis
    show_team_analysis(fixtures)
    
    # Initialize predictor
    print(f"\n📚 Training model on historical data...")
    predictor = FutureGamePredictor()
    predictor.load_trained_model()
    
    # Make predictions
    print(f"\n🔮 Making predictions...")
    predictions = predictor.predict_multiple_matches(fixtures)
    
    # Add dates back to predictions (IMPORTANT!)
    for i, pred in enumerate(predictions):
        if i < len(fixtures):
            pred['match_date'] = fixtures[i].get('match_date', 'TBD')
            pred['match_time'] = fixtures[i].get('match_time', '')
    
    # Save to files
    csv_file_output, txt_file = predictor.save_predictions(predictions)
    
    # Print detailed summary with dates
    print(f"\n🔮 DETAILED PREDICTION RESULTS")
    print("=" * 70)
    
    for i, pred in enumerate(predictions, 1):
        # Get match details including dates
        match_date = pred.get('match_date', 'TBD')
        match_time = pred.get('match_time', '')
        
        print(f"\n🏟️  MATCH {i}: {pred['match']}")
        print(f"📅 Date: {match_date} {match_time}")
        print("-" * 50)
        
        if 'predictions' in pred:
            p = pred['predictions']
            print(f"   🏠 Home Win:     {p.get('home_win_prob', 0)*100:5.1f}%")
            print(f"   🤝 Draw:         {p.get('draw_prob', 0)*100:5.1f}%") 
            print(f"   ✈️  Away Win:     {p.get('away_win_prob', 0)*100:5.1f}%")
            print(f"   ⚽ Over 2.5:     {p.get('over_2.5_prob', 0)*100:5.1f}%")
            print(f"   🔒 Under 2.5:    {p.get('under_2.5_prob', 0)*100:5.1f}%")
            print(f"   🥅 BTTS Yes:     {p.get('btts_prob', 0)*100:5.1f}%")
            print(f"   🛡️  BTTS No:      {p.get('btts_no_prob', 0)*100:5.1f}%")
            print(f"   🎯 Most Likely:  {p.get('most_likely', 'Unknown')} ({p.get('confidence', 0)*100:.1f}%)")
            
            # Add predicted goals
            if 'predicted_score' in p:
                print(f"   ⚽ Predicted Score: {p.get('predicted_score', 'N/A')}")
                print(f"   🏠 Home Goals: {p.get('home_predicted_goals', 0)} ({p.get('home_expected_goals', 0)} xG)")
                print(f"   ✈️  Away Goals: {p.get('away_predicted_goals', 0)} ({p.get('away_expected_goals', 0)} xG)")
                print(f"   📊 Total Goals: {p.get('total_predicted_goals', 0)}")
        
        if pred.get('betting_recommendations'):
            print(f"\n   💰 BETTING RECOMMENDATIONS:")
            for j, bet in enumerate(pred['betting_recommendations'], 1):
                print(f"      {j}. {bet['bet_type']}")
                print(f"         Confidence: {bet['probability']*100:.1f}%")
                print(f"         Reason: {bet['reasoning']}")
        else:
            print(f"\n   💰 No strong betting recommendations")
            print(f"       (All predictions below 60% confidence threshold)")
    
    # Summary statistics
    total_recommendations = sum(len(pred.get('betting_recommendations', [])) for pred in predictions)
    high_confidence_matches = sum(1 for pred in predictions 
                                 if pred.get('predictions', {}).get('confidence', 0) > 0.6)
    
    print(f"\n📊 PREDICTION SUMMARY")
    print("=" * 30)
    print(f"Total Matches Analyzed: {len(predictions)}")
    print(f"High Confidence Predictions: {high_confidence_matches}")
    print(f"Total Betting Recommendations: {total_recommendations}")
    print(f"Average Recommendations per Match: {total_recommendations/len(predictions):.1f}")
    
    print(f"\n📁 RESULTS SAVED TO:")
    print(f"   📊 CSV File: {csv_file_output}")
    print(f"   📄 Detailed Report: {txt_file}")
    
    print(f"\n✅ Prediction analysis complete!")

if __name__ == "__main__":
    main()