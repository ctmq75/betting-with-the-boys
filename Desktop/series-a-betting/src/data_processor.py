"""
Data processing utilities for Serie A betting model
Handles data cleaning, feature engineering, and preprocessing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.team_encoders = {}
        self.referee_encoders = {}
        
    def process_data(self, df):
        """Main data processing pipeline"""
        df = df.copy()
        
        # Basic cleaning
        df = self._clean_basic_data(df)
        
        # Create target variables
        df = self._create_targets(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df
    
    def _clean_basic_data(self, df):
        """Basic data cleaning operations"""
        print("Cleaning basic data...")
        
        # Convert date
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], errors='coerce')
        
        # Remove matches with missing essential data
        essential_cols = ['home_team_goal_count', 'away_team_goal_count', 'home_team', 'away_team']
        df = df.dropna(subset=essential_cols)
        
        # Sort by date
        df = df.sort_values('date_GMT').reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date_GMT', 'home_team', 'away_team'])
        
        print(f"Data cleaned. {len(df)} matches remaining.")
        return df
    
    def _create_targets(self, df):
        """Create target variables for betting"""
        print("Creating target variables...")
        
        # Match outcomes
        df['home_win'] = (df['home_team_goal_count'] > df['away_team_goal_count']).astype(int)
        df['draw'] = (df['home_team_goal_count'] == df['away_team_goal_count']).astype(int)
        df['away_win'] = (df['home_team_goal_count'] < df['away_team_goal_count']).astype(int)
        
        # Goals betting targets
        df['over_0.5'] = (df['total_goal_count'] > 0.5).astype(int)
        df['over_1.5'] = (df['total_goal_count'] > 1.5).astype(int)
        df['over_2.5'] = (df['total_goal_count'] > 2.5).astype(int)
        df['over_3.5'] = (df['total_goal_count'] > 3.5).astype(int)
        df['over_4.5'] = (df['total_goal_count'] > 4.5).astype(int)
        
        # Both teams to score
        df['btts'] = ((df['home_team_goal_count'] > 0) & (df['away_team_goal_count'] > 0)).astype(int)
        
        # Half-time targets
        df['over_0.5_ht'] = (df['total_goals_at_half_time'] > 0.5).astype(int)
        df['over_1.5_ht'] = (df['total_goals_at_half_time'] > 1.5).astype(int)
        
        # Goal difference
        df['goal_difference'] = df['home_team_goal_count'] - df['away_team_goal_count']
        
        return df
    
    def _engineer_features(self, df):
        """Engineer features for modeling"""
        print("Engineering features...")
        
        # Performance differences
        df['ppg_difference'] = df['home_ppg'] - df['away_ppg']
        df['pre_match_ppg_difference'] = df['Pre-Match PPG (Home)'] - df['Pre-Match PPG (Away)']
        
        # Expected goals features
        if 'team_a_xg' in df.columns and 'team_b_xg' in df.columns:
            df['xg_difference'] = df['team_a_xg'] - df['team_b_xg']
        
        if 'Home Team Pre-Match xG' in df.columns and 'Away Team Pre-Match xG' in df.columns:
            df['pre_match_xg_difference'] = df['Home Team Pre-Match xG'] - df['Away Team Pre-Match xG']
        
        # Odds-based features
        df = self._create_odds_features(df)
        
        # Shot efficiency
        df['home_shot_accuracy'] = np.where(df['home_team_shots'] > 0, 
                                           df['home_team_shots_on_target'] / df['home_team_shots'], 0)
        df['away_shot_accuracy'] = np.where(df['away_team_shots'] > 0, 
                                           df['away_team_shots_on_target'] / df['away_team_shots'], 0)
        
        # Possession difference
        if 'home_team_possession' in df.columns:
            df['possession_difference'] = df['home_team_possession'] - df['away_team_possession']
        
        # Card and discipline features
        df['home_total_cards'] = df['home_team_yellow_cards'] + df['home_team_red_cards'] * 2
        df['away_total_cards'] = df['away_team_yellow_cards'] + df['away_team_red_cards'] * 2
        df['cards_difference'] = df['home_total_cards'] - df['away_total_cards']
        
        # Corner difference
        df['corner_difference'] = df['home_team_corner_count'] - df['away_team_corner_count']
        
        # Fouls difference
        df['fouls_difference'] = df['home_team_fouls'] - df['away_team_fouls']
        
        # Time-based features
        df['month'] = df['date_GMT'].dt.month
        df['day_of_week'] = df['date_GMT'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Season progress (Game Week normalized)
        df['season_progress'] = df['Game Week'] / df['Game Week'].max()
        
        # Rolling averages (team form)
        df = self._create_rolling_features(df)
        
        return df
    
    def _create_odds_features(self, df):
        """Create features based on betting odds"""
        print("Creating odds-based features...")
        
        # Implied probabilities
        df['home_win_probability'] = 1 / df['odds_ft_home_team_win']
        df['draw_probability'] = 1 / df['odds_ft_draw']
        df['away_win_probability'] = 1 / df['odds_ft_away_team_win']
        
        # Market efficiency features
        df['total_probability'] = (df['home_win_probability'] + 
                                  df['draw_probability'] + 
                                  df['away_win_probability'])
        df['bookmaker_margin'] = (df['total_probability'] - 1) * 100
        
        # Favorite/underdog features
        df['home_favorite'] = (df['odds_ft_home_team_win'] < df['odds_ft_away_team_win']).astype(int)
        df['odds_ratio'] = df['odds_ft_home_team_win'] / df['odds_ft_away_team_win']
        
        # Value features (for later betting decisions)
        df['home_win_value'] = df['home_win_probability'] * df['odds_ft_home_team_win']
        df['draw_value'] = df['draw_probability'] * df['odds_ft_draw']
        df['away_win_value'] = df['away_win_probability'] * df['odds_ft_away_team_win']
        
        # Goals betting odds
        if 'odds_ft_over25' in df.columns:
            df['over_2.5_probability'] = 1 / df['odds_ft_over25']
            df['under_2.5_probability'] = 1 - df['over_2.5_probability']
        
        if 'odds_btts_yes' in df.columns:
            df['btts_probability'] = 1 / df['odds_btts_yes']
            df['btts_no_probability'] = 1 / df['odds_btts_no']
        
        return df
    
    def _create_rolling_features(self, df, window=5):
        """Create rolling average features for team form"""
        print(f"Creating rolling features with {window}-game window...")
        
        df = df.sort_values(['date_GMT']).reset_index(drop=True)
        
        # Team-specific rolling features
        for team_col in ['home_team', 'away_team']:
            for stat in ['goal_count', 'shots', 'shots_on_target', 'possession', 'corner_count']:
                col_name = f'{team_col.replace("_team", "")}_team_{stat}'
                if col_name in df.columns:
                    # Create rolling average for each team
                    df[f'{col_name}_form'] = df.groupby(team_col)[col_name].rolling(window=window, min_periods=1).mean().values
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        print("Handling missing values...")
        
        # Numeric columns - fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical columns - fill with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # Percentage columns - ensure they're between 0 and 100
        percentage_cols = [col for col in df.columns if 'percentage' in col.lower()]
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].clip(0, 100)
        
        print(f"Missing values handled. Final dataset shape: {df.shape}")
        return df
    
    def create_feature_matrix(self, df):
        """Create the final feature matrix for modeling"""
        feature_columns = [
            # Team strength indicators
            'home_ppg', 'away_ppg', 'ppg_difference',
            'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'pre_match_ppg_difference',
            
            # Expected goals
            'team_a_xg', 'team_b_xg', 'xg_difference',
            'Home Team Pre-Match xG', 'Away Team Pre-Match xG', 'pre_match_xg_difference',
            
            # Match performance
            'home_team_shots', 'away_team_shots',
            'home_team_shots_on_target', 'away_team_shots_on_target',
            'home_shot_accuracy', 'away_shot_accuracy',
            
            # Possession and territory
            'home_team_possession', 'away_team_possession', 'possession_difference',
            'home_team_corner_count', 'away_team_corner_count', 'corner_difference',
            
            # Discipline
            'home_total_cards', 'away_total_cards', 'cards_difference',
            'home_team_fouls', 'away_team_fouls'
        ]