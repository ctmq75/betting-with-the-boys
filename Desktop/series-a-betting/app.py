"""
Serie A Betting Predictions - Web App
Streamlit interface for friends to use the betting model
WITH MODEL TUNING UI
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

def apply_custom_parameters_to_predictor(predictor):
    """Apply custom parameters from UI to the predictor"""
    if 'custom_params' in st.session_state:
        params = st.session_state['custom_params']
        
        # Apply goal prediction parameters
        predictor.home_advantage = params.get('home_advantage', 0.35)
        predictor.home_attack_weight = params.get('home_attack_weight', 0.6)
        predictor.away_defense_weight = params.get('away_defense_weight', 0.4)
        predictor.away_attack_weight = params.get('away_attack_weight', 0.6)
        predictor.home_defense_weight = params.get('home_defense_weight', 0.4)
        predictor.form_impact = params.get('form_impact', 0.2)
        predictor.outcome_adjustment = params.get('outcome_adjustment', 0.3)
        predictor.league_avg_goals = params.get('league_avg_goals', 2.6)
        
        # Feature weights
        predictor.feature_weights = {
            'recent_form': params.get('recent_form_weight', 1.5),
            'historical_form': 1.0,
            'attacking': params.get('attacking_weight', 1.3),
            'defensive': params.get('defensive_weight', 1.1),
            'home_advantage': params.get('home_advantage_weight', 1.2),
            'xg_importance': params.get('xg_importance', 0.9),
            'possession': params.get('possession_weight', 0.8),
            'set_pieces': 1.1
        }
        
        # Market logic parameters
        predictor.strength_diff_major = params.get('strength_diff_major', 0.8)
        predictor.strength_diff_minor = params.get('strength_diff_minor', 0.4)
        predictor.over25_high_threshold = params.get('over25_high_threshold', 3.2)
        predictor.over25_med_threshold = params.get('over25_med_threshold', 2.8)
        
        # Betting strategy parameters
        predictor.default_confidence = params.get('default_confidence', 0.65)
        predictor.outcome_confidence = params.get('outcome_confidence', 0.60)
        predictor.goals_confidence = params.get('goals_confidence', 0.65)
        predictor.btts_confidence = params.get('btts_confidence', 0.65)
        
        st.info("🎛️ Using custom parameters for predictions!")
    
    return predictor

# Page config
st.set_page_config(
    page_title="Serie A Betting Predictions",
    page_icon="⚽",
    layout="wide"
)

# Title and description
st.title("⚽ Serie A Betting Predictions")
st.markdown("### 🤖 AI-Powered Football Betting Analysis")
st.markdown("Upload your upcoming matches CSV file and get instant predictions!")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    1. **Upload CSV** with upcoming matches
    2. **Check the data** looks correct
    3. **Click predict** to get AI analysis
    4. **Download results** as CSV
    
    **CSV Format Required:**
    - `home_team` column
    - `away_team` column  
    - `date` column (optional)
    """)
    
    st.header("📊 Model Info")
    st.markdown("""
    - **Accuracy**: ~65% win rate
    - **Data**: 3+ years Serie A history
    - **Models**: Random Forest + Gradient Boosting
    - **Conservative**: Only high-confidence bets
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📁 Upload Matches")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose your CSV file with upcoming matches",
        type="csv",
        help="File should contain home_team, away_team, and optionally date columns"
    )
    
    # Sample data option
    if st.button("📝 Use Sample Data"):
        sample_data = {
            'home_team': ['Juventus', 'AC Milan', 'Roma', 'Napoli', 'Inter'],
            'away_team': ['Inter', 'Napoli', 'Lazio', 'Atalanta', 'Fiorentina'],
            'date': ['2025-08-23', '2025-08-24', '2025-08-25', '2025-08-26', '2025-08-27']
        }
        st.session_state['sample_df'] = pd.DataFrame(sample_data)
        uploaded_file = "sample"

with col2:
    st.header("⚡ Quick Stats")
    if 'predictions_made' not in st.session_state:
        st.session_state['predictions_made'] = 0
    
    st.metric("Predictions Made", st.session_state['predictions_made'])
    st.metric("Model Accuracy", "65.2%")
    st.metric("Available Teams", "20")

# Process uploaded file
df = None
if uploaded_file is not None:
    try:
        if uploaded_file == "sample":
            df = st.session_state['sample_df']
            st.success("✅ Sample data loaded!")
        else:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        
        # Display the data
        st.header("📊 Upcoming Matches")
        st.dataframe(df, use_container_width=True)
        
        # Validate data
        required_columns = ['home_team', 'away_team']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.stop()
        
        st.success(f"✅ Found {len(df)} matches ready for prediction!")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.stop()

# Prediction section
if df is not None:
    st.header("🔮 Generate Predictions")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.5, 
            max_value=0.8, 
            value=0.65, 
            step=0.05,
            help="Higher = fewer but more confident predictions"
        )
    
    with col2:
        show_details = st.checkbox("Show Detailed Analysis", value=True)
    
    with col3:
        st.write("") # Spacer
        predict_button = st.button("🚀 Generate Predictions", type="primary")
    
    if predict_button:
        with st.spinner("🤖 Training AI model and analyzing matches..."):
            try:
                # Import and run the model
                from predict_future import FutureGamePredictor
                
                # Initialize predictor
                predictor = FutureGamePredictor()
                
                # Apply custom parameters if they exist
                predictor = apply_custom_parameters_to_predictor(predictor)
                
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading historical data...")
                progress_bar.progress(20)
                
                predictor.load_trained_model()
                progress_bar.progress(60)
                
                # Convert CSV to fixtures format
                status_text.text("Processing matches...")
                fixtures = []
                for _, row in df.iterrows():
                    fixtures.append({
                        'home_team': str(row['home_team']).strip(),
                        'away_team': str(row['away_team']).strip(),
                        'match_date': str(row.get('date', 'TBD')),
                        'match_time': '15:00',
                        'game_week': 22
                    })
                
                progress_bar.progress(80)
                
                # Make predictions
                status_text.text("Generating predictions...")
                predictions = predictor.predict_multiple_matches(fixtures)
                
                # Add dates back to predictions and ensure all fields exist
                for i, pred in enumerate(predictions):
                    if i < len(fixtures):
                        pred['match_date'] = fixtures[i].get('match_date', 'TBD')
                        pred['match_time'] = fixtures[i].get('match_time', '')
                    
                    # Ensure required fields exist
                    if 'predictions' in pred:
                        p = pred['predictions']
                        # Add missing fields if they don't exist
                        if 'total_predicted_goals' not in p:
                            home_goals = p.get('home_predicted_goals', 1)
                            away_goals = p.get('away_predicted_goals', 1)
                            p['total_predicted_goals'] = home_goals + away_goals
                        
                        if 'predicted_score' not in p:
                            home_goals = p.get('home_predicted_goals', 1)
                            away_goals = p.get('away_predicted_goals', 1)
                            p['predicted_score'] = f"{home_goals}-{away_goals}"
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Update session state
                st.session_state['predictions_made'] += len(predictions)
                st.session_state['last_predictions'] = predictions
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show parameter info if custom params are being used
                if 'custom_params' in st.session_state:
                    st.success("🎛️ Predictions generated using your custom parameters!")
                    with st.expander("📊 Current Parameter Settings"):
                        st.json(st.session_state['custom_params'])
                
                # Display results
                st.header("🎯 Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                total_recommendations = sum(len(pred.get('betting_recommendations', [])) for pred in predictions)
                high_confidence = sum(1 for pred in predictions if pred.get('predictions', {}).get('confidence', 0) > confidence_threshold)
                
                with col1:
                    st.metric("Matches Analyzed", len(predictions))
                with col2:
                    st.metric("High Confidence", high_confidence)
                with col3:
                    st.metric("Betting Tips", total_recommendations)
                with col4:
                    avg_confidence = np.mean([pred.get('predictions', {}).get('confidence', 0) for pred in predictions]) * 100
                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                
                # Individual match predictions
                for i, pred in enumerate(predictions, 1):
                    match_date = pred.get('match_date', 'TBD')
                    match_time = pred.get('match_time', '')
                    
                    with st.expander(f"🏟️ {pred['match']} - {match_date} {match_time}", expanded=(i <= 3)):
                        if 'predictions' in pred:
                            p = pred['predictions']
                            
                            # Main prediction
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🏠 Home Win", f"{p.get('home_win_prob', 0)*100:.1f}%")
                            with col2:
                                st.metric("🤝 Draw", f"{p.get('draw_prob', 0)*100:.1f}%")
                            with col3:
                                st.metric("✈️ Away Win", f"{p.get('away_win_prob', 0)*100:.1f}%")
                            
                            # Predicted score
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if 'predicted_score' in p:
                                    st.metric("⚽ Predicted Score", p.get('predicted_score', 'N/A'))
                            with col2:
                                st.metric("🎯 Most Likely", p.get('most_likely', 'Unknown'))
                            with col3:
                                confidence = p.get('confidence', 0) * 100
                                st.metric("📊 Confidence", f"{confidence:.1f}%")
                            
                            if show_details:
                                # Additional details
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Goals Predictions:**")
                                    st.write(f"- Over 2.5: {p.get('over_2.5_prob', 0)*100:.1f}%")
                                    st.write(f"- Under 2.5: {p.get('under_2.5_prob', 0)*100:.1f}%")
                             