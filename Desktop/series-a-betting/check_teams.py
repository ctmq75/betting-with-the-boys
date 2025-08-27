"""
Check for team name mismatches between historical data and upcoming matches
Run this first to see what needs fixing
"""

import pandas as pd
import os

def check_team_consistency():
    print("🔍 Checking team consistency between historical and upcoming data...")
    
    # Load historical data
    if os.path.exists('data/raw/serie_a_historical.csv'):
        historical_df = pd.read_csv('data/raw/serie_a_historical.csv')
        historical_teams = set(historical_df['home_team'].unique()) | set(historical_df['away_team'].unique())
        print(f"📊 Found {len(historical_teams)} teams in historical data:")
        for team in sorted(historical_teams):
            print(f"   - {team}")
    else:
        print("❌ Historical data file not found!")
        return
    
    print("\n" + "="*50)
    
    # Load upcoming matches
    if os.path.exists('data/upcoming_matches.csv'):
        upcoming_df = pd.read_csv('data/upcoming_matches.csv')
        upcoming_teams = set(upcoming_df['home_team'].unique()) | set(upcoming_df['away_team'].unique())
        print(f"🔮 Found {len(upcoming_teams)} teams in upcoming matches:")
        for team in sorted(upcoming_teams):
            print(f"   - {team}")
    else:
        print("❌ Upcoming matches file not found!")
        return
    
    print("\n" + "="*50)
    
    # Find mismatches
    missing_from_historical = upcoming_teams - historical_teams
    missing_from_upcoming = historical_teams - upcoming_teams
    
    if missing_from_historical:
        print("⚠️  Teams in upcoming matches but NOT in historical data:")
        for team in sorted(missing_from_historical):
            print(f"   ❌ {team}")
    
    if missing_from_upcoming:
        print("\n📋 Teams in historical data but NOT in upcoming matches:")
        for team in sorted(missing_from_upcoming):
            print(f"   📝 {team}")
    
    if not missing_from_historical and not missing_from_upcoming:
        print("✅ All teams match perfectly!")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Historical teams: {len(historical_teams)}")
    print(f"   Upcoming teams: {len(upcoming_teams)}")
    print(f"   Teams that match: {len(historical_teams & upcoming_teams)}")
    print(f"   Teams missing from historical: {len(missing_from_historical)}")

if __name__ == "__main__":
    check_team_consistency()