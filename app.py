import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
import urllib.parse
import logging
import atexit
from backend.inference import MusicEmotionPredictor, predict_multiple_files

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('apollodb.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="ApollodB - Music Emotion Analysis",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://github.com/parikshitkumar/apollodb',
        'Report a bug': 'https://github.com/parikshitkumar/apollodb/issues',
        'About': "# ApollodB\nAI-Powered Music Emotion Analysis & EQ Optimization\n\nMade by Parikshit Kumar"
    }
)

# Advanced CSS for Premium Standalone Web App Experience
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
    @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
    
    /* Main Content Centering */
    .main .block-container {
        max-width: 1200px;
        margin: 0 auto;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Hide Streamlit Branding and UI Elements */
    .stApp > header {
        background-color: transparent;
    }
    
    .stApp > header[data-testid="stHeader"] {
        display: none;
    }
    
    .stToolbar {
        display: none;
    }
    
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    
    #MainMenu {
        visibility: hidden;
        height: 0%;
    }
    
    header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0%;
    }
    
    footer {
        visibility: hidden;
        height: 0%;
    }
    
    .viewerBadge_container__1QSob {
        display: none;
    }
    
    .viewerBadge_link__1S137 {
        display: none;
    }
    
    /* Custom App Styling */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Premium Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%);
        color: #ffffff;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }
    
    /* Animated Background Pattern */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(0, 188, 212, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(0, 188, 212, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(0, 188, 212, 0.03) 0%, transparent 50%);
        animation: backgroundShift 20s ease-in-out infinite;
        z-index: -1;
    }
    
    @keyframes backgroundShift {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 3rem;
        max-width: 1400px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }
    
    /* Premium Navigation Bar */
    .premium-navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(10, 10, 10, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 188, 212, 0.2);
        padding: 1rem 2rem;
        z-index: 1000;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .navbar-content {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .navbar-logo {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 1.5rem;
        color: #00bcd4;
        text-decoration: none;
    }
    
    .navbar-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .navbar-link {
        color: rgba(255, 255, 255, 0.8);
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s ease;
    }
    
    .navbar-link:hover {
        color: #00bcd4;
    }
    
    /* Hero Section */
    .hero-section {
        margin-top: 80px;
        padding: 4rem 2rem;
        text-align: center;
        position: relative;
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00bcd4 0%, #ffffff 50%, #00bcd4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        line-height: 1.2;
        text-align: center;
    }
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        0% { filter: drop-shadow(0 0 20px rgba(0, 188, 212, 0.3)); }
        100% { filter: drop-shadow(0 0 40px rgba(0, 188, 212, 0.6)); }
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 2rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
        text-align: center;
    }
    
    /* Premium Card System */
    .premium-card {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(0, 188, 212, 0.2);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.3),
            0 1px 0 rgba(255, 255, 255, 0.1) inset;
        backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .premium-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 188, 212, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 188, 212, 0.4);
        box-shadow: 
            0 16px 48px rgba(0, 0, 0, 0.4),
            0 0 20px rgba(0, 188, 212, 0.2);
    }
    
    .premium-card:hover::before {
        left: 100%;
    }
    
    /* Advanced Tab System */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, rgba(20, 20, 20, 0.95) 0%, rgba(30, 30, 30, 0.95) 100%);
        border-radius: 12px;
        padding: 0.5rem;
        gap: 4px;
        border: 1px solid rgba(0, 188, 212, 0.2);
        justify-content: center;
        margin: 2rem auto;
        max-width: 800px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border: none;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        position: relative;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 188, 212, 0.1);
        color: #00bcd4;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00bcd4 0%, #00acc1 100%);
        color: #000000;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 188, 212, 0.4);
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00bcd4 0%, #00acc1 100%);
        color: #000000;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(0, 188, 212, 0.3);
        transition: all 0.2s ease;
        min-height: 3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #ffffff 0%, #00bcd4 100%);
        box-shadow: 0 8px 24px rgba(0, 188, 212, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    /* Premium File Uploader */
    .stFileUploader {
        border: 2px dashed rgba(0, 188, 212, 0.5) !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: linear-gradient(135deg, rgba(0, 188, 212, 0.05) 0%, rgba(26, 26, 26, 0.8) 100%) !important;
        margin: 1rem 0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stFileUploader:hover {
        border-color: #00bcd4 !important;
        background: linear-gradient(135deg, rgba(0, 188, 212, 0.1) 0%, rgba(26, 26, 26, 0.9) 100%) !important;
        box-shadow: 0 8px 24px rgba(0, 188, 212, 0.2) !important;
    }
    
    /* Premium Metrics */
    .metric-container {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(0, 188, 212, 0.2);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
    }
    
    .metric-container::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00bcd4, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
        border-color: rgba(0, 188, 212, 0.4);
    }
    
    .metric-container:hover::after {
        opacity: 1;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        color: #00bcd4;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 188, 212, 0.3);
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.875rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Premium Code Display */
    .eq-container {
        background: linear-gradient(135deg, rgba(10, 10, 10, 0.95) 0%, rgba(20, 20, 20, 0.95) 100%);
        border: 1px solid rgba(0, 188, 212, 0.3);
        border-radius: 8px;
        padding: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        color: #00bcd4;
        white-space: pre-wrap;
        overflow-x: auto;
        box-shadow: 
            inset 0 2px 8px rgba(0, 0, 0, 0.3),
            0 4px 16px rgba(0, 0, 0, 0.2);
        min-height: 120px;
        max-height: 250px;
        overflow-y: auto;
        margin: 1rem 0;
        position: relative;
    }
    
    .eq-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 188, 212, 0.5), transparent);
    }
    
    /* Settings Panel */
    .settings-panel {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%);
        border: 1px solid rgba(0, 188, 212, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
    }
    
    /* Form Elements */
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.2);
        height: 4px;
        border-radius: 2px;
    }
    
    .stSlider > div > div > div > div {
        background: #00bcd4;
        border: 2px solid #ffffff;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(0, 188, 212, 0.3);
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%);
        border: 1px solid rgba(0, 188, 212, 0.2);
        border-radius: 8px;
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #00bcd4;
        box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.2);
    }
    
    /* Radio Buttons */
    .stRadio > div > label > div[data-checked="true"] {
        background-color: #00bcd4 !important;
        border-color: #00bcd4 !important;
    }
    
    .stRadio > div > label > div[data-checked="true"]::before {
        background-color: #000000 !important;
    }
    
    /* Alerts and Messages */
    .stAlert {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%);
        border: 1px solid rgba(0, 188, 212, 0.2);
        border-radius: 8px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(20px);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(26, 26, 26, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%) !important;
        border: 1px solid #4caf50 !important;
        border-radius: 8px !important;
        color: #4caf50 !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff;
    }
    
    p, div, span {
        font-family: 'Inter', sans-serif;
        color: rgba(255, 255, 255, 0.87);
        line-height: 1.6;
    }
    
    /* Premium Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .animate-slide {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .premium-navbar {
            padding: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
        
        .premium-card {
            padding: 1rem;
        }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 26, 26, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00bcd4, #00acc1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #ffffff, #00bcd4);
    }
    
    /* Loading Animations */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Premium Footer */
    .premium-footer {
        margin-top: 4rem;
        padding: 2rem 0;
        text-align: center;
        border-top: 1px solid rgba(0, 188, 212, 0.2);
        background: linear-gradient(135deg, rgba(10, 10, 10, 0.9) 0%, rgba(20, 20, 20, 0.9) 100%);
        backdrop-filter: blur(20px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with error handling
if 'predictor' not in st.session_state:
    try:
        with st.spinner("Loading AI model... This may take a moment on first run."):
            st.session_state.predictor = MusicEmotionPredictor()
        logger.info("MusicEmotionPredictor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"""
        **Error loading AI model:** {e}
        
        This could be due to:
        - Missing model files (best_model.h5, scaler_mean.npy, scaler_scale.npy)
        - Insufficient system memory
        - Missing dependencies
        
        Please check the installation and try again.
        """)
        st.stop()

def create_valence_arousal_plot(valence, arousal, emotion=None):
    """Create an interactive valence-arousal plot"""
    fig = go.Figure()
    
    # Add quadrant background
    fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5,
                  fillcolor="rgba(255,0,0,0.1)", line=dict(color="rgba(255,0,0,0.3)"))
    fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5,
                  fillcolor="rgba(0,255,0,0.1)", line=dict(color="rgba(0,255,0,0.3)"))
    fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1,
                  fillcolor="rgba(255,255,0,0.1)", line=dict(color="rgba(255,255,0,0.3)"))
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1,
                  fillcolor="rgba(255,165,0,0.1)", line=dict(color="rgba(255,165,0,0.3)"))
    
    # Add emotion reference points
    emotion_points = {
        "happy": (0.8, 0.8),
        "sad": (0.2, 0.2),
        "calm": (0.8, 0.2),
        "neutral": (0.5, 0.5)
    }
    
    for emo, (v, a) in emotion_points.items():
        color = "#00bcd4" if emo == emotion else "#666666"
        size = 15 if emo == emotion else 8
        fig.add_trace(go.Scatter(
            x=[v], y=[a],
            mode='markers+text',
            text=[emo.capitalize()],
            textposition="top center",
            marker=dict(size=size, color=color),
            name=emo.capitalize(),
            showlegend=False
        ))
    
    # Add user's position
    fig.add_trace(go.Scatter(
        x=[valence], y=[arousal],
        mode='markers',
        marker=dict(size=20, color="#00bcd4", symbol="star"),
        name="Your Music",
        showlegend=True
    ))
    
    # Add quadrant labels
    fig.add_annotation(x=0.25, y=0.75, text="INTENSE<br>Low Valence<br>High Arousal",
                      showarrow=False, font=dict(color="#ffffff", size=12))
    fig.add_annotation(x=0.75, y=0.75, text="HAPPY<br>High Valence<br>High Arousal",
                      showarrow=False, font=dict(color="#ffffff", size=12))
    fig.add_annotation(x=0.25, y=0.25, text="SAD<br>Low Valence<br>Low Arousal",
                      showarrow=False, font=dict(color="#ffffff", size=12))
    fig.add_annotation(x=0.75, y=0.25, text="CALM<br>High Valence<br>Low Arousal",
                      showarrow=False, font=dict(color="#ffffff", size=12))
    
    fig.update_layout(
        title="Valence-Arousal Space",
        xaxis_title="Valence (Negativity ← → Positivity)",
        yaxis_title="Arousal (Low Energy ← → High Energy)",
        xaxis=dict(range=[0, 1], gridcolor="#333333"),
        yaxis=dict(range=[0, 1], gridcolor="#333333"),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#ffffff", family="Roboto", size=12),
        showlegend=True
    )
    
    return fig

def create_eq_visualization(eq_data, emotion):
    """Create EQ curve visualization"""
    # Parse the wavelet EQ data
    eq_string = eq_data["wavelet"]
    eq_parts = eq_string.replace("GraphicEQ: ", "").split("; ")
    
    frequencies = []
    gains = []
    
    for part in eq_parts:
        freq, gain = part.split(" ")
        frequencies.append(float(freq))
        gains.append(float(gain))
    
    fig = go.Figure()
    
    # Add EQ curve
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=gains,
        mode='lines+markers',
        line=dict(color="#00bcd4", width=3),
        marker=dict(size=4, color="#00bcd4"),
        name=f"{emotion.capitalize()} EQ"
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#666666")
    
    fig.update_layout(
        title=f"EQ Curve for {emotion.capitalize()} Music",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Gain (dB)",
        xaxis=dict(type="log", gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333"),
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#ffffff", family="Roboto", size=12)
    )
    
    return fig

def create_emotion_distribution_chart(emotion_dist):
    """Create emotion distribution pie chart"""
    emotions = list(emotion_dist.keys())
    counts = list(emotion_dist.values())
    
    colors = ["#00bcd4", "#ffffff", "#888888", "#444444", "#666666"]
    
    fig = go.Figure(data=[go.Pie(
        labels=emotions,
        values=counts,
        hole=0.4,
        marker=dict(colors=colors[:len(emotions)], line=dict(color="#000000", width=2))
    )])
    
    fig.update_layout(
        title="Emotion Distribution in Your Music",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#ffffff", family="Roboto", size=12)
    )
    
    return fig

# Main app
def main():
    # Add Premium Navigation Bar
    st.markdown("""
    <div class="premium-navbar">
        <div class="navbar-content" style="display: flex; justify-content: center; align-items: center; gap: 1.5rem;">
            <div class="navbar-logo" style="font-size: 1.5rem; font-weight: bold; color: #00bcd4;">ApollodB</div>
            <a href="https://github.com/parikshitkumar1" target="_blank" style="
                background: linear-gradient(135deg, #333333 0%, #666666 100%);
                color: #ffffff;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 500;
                font-size: 0.9rem;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(51, 51, 51, 0.3);
            ">GitHub</a>
            <a href="https://www.linkedin.com/in/parikshitkumar1/" target="_blank" style="
                background: linear-gradient(135deg, #0077B5 0%, #0099D4 100%);
                color: #ffffff;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 500;
                font-size: 0.9rem;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(0, 119, 181, 0.3);
            ">LinkedIn</a>
            <a href="https://coff.ee/parikshitkumar" target="_blank" style="
                background: linear-gradient(135deg, #FFDD44 0%, #FF6B35 100%);
                color: #000000;
                text-decoration: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 500;
                font-size: 0.9rem;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(255, 221, 68, 0.3);
            ">Buy Me Coffee</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium Hero Section
    st.markdown("""
    <div class="hero-section" style="text-align: center; padding: 2rem 0;">
        <h1 class="hero-title" style="margin: 0; font-size: 3rem; font-weight: 700; text-align: center;">ApollodB</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical description with premium styling
    st.markdown("""
    <div class="premium-card animate-in" style="margin-bottom: 2rem; max-width: 1000px; margin-left: auto; margin-right: auto;">
        <h3 style="margin-top: 0; color: #00bcd4; text-align: center;">Foundation</h3>
        <p style="text-align: justify; line-height: 1.6;">
        ApollodB represents a breakthrough in computational musicology, leveraging state-of-the-art deep learning on the <strong>DEAM (Database for Emotional Analysis in Music)</strong> dataset. 
        Our convolutional neural network analyzes audio spectrograms to map music into Russell's circumplex valence-arousal emotional space. 
        The model is trained on professionally annotated songs across <strong>4 core emotional categories</strong> (neutral, happy, sad, calm), employing sophisticated bias handling to ensure accurate emotion detection.
        The best part? It's completely open source.
        </p>
        <p style="text-align: justify; line-height: 1.6;">
        The system employs psychoacoustic research to translate emotional classifications into frequency-domain adjustments, generating personalized EQ curves that enhance the perceived emotional impact of music. 
        <strong>Audiophiles and IEM enthusiasts</strong> benefit from scientifically-grounded recommendations that match their listening preferences with optimal frequency responses, 
        bridging the gap between subjective musical taste and objective acoustic engineering. This revolutionary approach enables unprecedented personalization in audio reproduction, 
        helping users discover their optimal sound signature while understanding the emotional characteristics that drive their musical preferences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analysis", "IEM Database", "FAQ", "References", "About"])
    
    with tab1:
        st.markdown("""
        <div class="premium-card animate-in" style="max-width: 800px; margin: 0 auto 2rem auto;">
            <h3 style="margin-top: 0; color: #00bcd4; text-align: center;">Upload Your Music</h3>
            <p style="margin-bottom: 0; text-align: center;">Upload one or multiple audio files to analyze their emotional characteristics and generate personalized EQ settings. Experience the magic of AI-powered music analysis that actually understands your taste!</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose audio files",
            type=['mp3', 'wav', 'm4a', 'flac'],
            accept_multiple_files=True,
            help="Select one or multiple audio files (MP3, WAV, M4A, FLAC)"
        )
        
        # Show upload status
        if uploaded_files:
            st.success(f"Successfully uploaded {len(uploaded_files)} file(s)")
            with st.expander("View uploaded files"):
                for file in uploaded_files:
                    st.write(f"• **{file.name}** ({file.size:,} bytes)")
        
        if uploaded_files:
            # Analysis type selection
            st.markdown("""
            <div class="premium-card animate-slide">
                <h4 style="margin-top: 0; color: #00bcd4;">Analysis Type</h4>
            </div>
            """, unsafe_allow_html=True)
            
            analysis_type = st.radio(
                "Choose analysis type:",
                ["Individual Song Analysis", "Batch Analysis (All Songs)"],
                index=1 if len(uploaded_files) > 1 else 0
            )
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("""
                <div class="settings-panel animate-slide">
                    <h4 style="margin-top: 0; color: #00bcd4;">Settings</h4>
                </div>
                """, unsafe_allow_html=True)
                
                aggression = st.slider(
                    "EQ Aggression",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="Controls how aggressive the EQ adjustments will be - higher values mean more dramatic frequency changes to enhance your music's emotional impact"
                )
                
                eq_style = st.selectbox(
                    "EQ Style",
                    ["Wavelet", "Parametric", "Graphic"],
                    help="Choose your preferred EQ format"
                )
            
            with col2:
                if st.button("Analyze Music", use_container_width=True):
                    with st.spinner("Analyzing your music..."):
                        # Save uploaded files temporarily
                        temp_files = []
                        file_names = []
                        for uploaded_file in uploaded_files:
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                            temp_file.write(uploaded_file.read())
                            temp_file.close()
                            temp_files.append(temp_file.name)
                            file_names.append(uploaded_file.name)
                        
                        try:
                            if analysis_type == "Individual Song Analysis":
                                # Individual analysis
                                individual_results = []
                                for i, (temp_file, filename) in enumerate(zip(temp_files, file_names)):
                                    try:
                                        result = st.session_state.predictor.predict_emotion(temp_file)
                                        result['filename'] = filename
                                        result['temp_path'] = temp_file
                                        individual_results.append(result)
                                        logger.info(f"Successfully analyzed: {filename}")
                                    except Exception as e:
                                        logger.error(f"Error analyzing {filename}: {e}")
                                        st.error(f"Failed to analyze {filename}: {str(e)}")
                                        continue
                                
                                if individual_results:
                                    st.session_state.individual_results = individual_results
                                    st.session_state.analysis_mode = "individual"
                                else:
                                    st.error("No files were successfully analyzed.")
                                    return
                            else:
                                # Batch analysis
                                try:
                                    results = predict_multiple_files(temp_files, st.session_state.predictor)
                                    logger.info(f"Batch analysis completed for {len(temp_files)} files")
                                    
                                    # Also store individual results with filenames for batch mode
                                    individual_results = []
                                    for i, (temp_file, filename) in enumerate(zip(temp_files, file_names)):
                                        if i < len(results['individual_results']):
                                            result = results['individual_results'][i].copy()
                                            result['filename'] = filename
                                            result['temp_path'] = temp_file
                                            individual_results.append(result)
                                    
                                    st.session_state.analysis_results = results
                                    st.session_state.individual_results = individual_results
                                    st.session_state.analysis_mode = "batch"
                                except Exception as e:
                                    logger.error(f"Batch analysis failed: {e}")
                                    st.error(f"Batch analysis failed: {str(e)}")
                                    return
                            
                            st.session_state.aggression = aggression
                            st.session_state.eq_style = eq_style
                            st.session_state.temp_files = temp_files
                            st.session_state.file_names = file_names
                            st.success("Analysis complete! Your music has been analyzed successfully.")
                            logger.info("Analysis completed successfully")
                        
                        except Exception as e:
                            logger.error(f"Critical error during analysis: {e}")
                            st.error(f"""
                            **Critical error during analysis:** {str(e)}
                            
                            Please try:
                            - Using different audio files
                            - Checking file formats (MP3, WAV, M4A, FLAC)
                            - Ensuring files are not corrupted
                            - Refreshing the page and trying again
                            """)
                            # Clean up temp files on error
                            for temp_file in temp_files:
                                try:
                                    os.unlink(temp_file)
                                except:
                                    pass
        
        # Display results based on analysis mode
        if 'analysis_mode' in st.session_state:
            analysis_mode = st.session_state.analysis_mode
            aggression = st.session_state.get('aggression', 0.5)
            eq_style = st.session_state.get('eq_style', 'Wavelet')
            
            if analysis_mode == "individual" and 'individual_results' in st.session_state:
                # Individual Analysis Display
                individual_results = st.session_state.individual_results
                
                st.markdown("---")
                st.markdown("### Individual Song Analysis")
                
                # Song selector
                selected_song = st.selectbox(
                    "Select song to analyze:",
                    range(len(individual_results)),
                    format_func=lambda x: individual_results[x]['filename']
                )
                
                result = individual_results[selected_song]
                
                # Individual song metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{result['primary_emotion'].title()}</div>
                        <div class="metric-label">Primary Emotion</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{result['secondary_emotion'].title()}</div>
                        <div class="metric-label">Secondary Emotion</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{result['valence']:.2f}</div>
                        <div class="metric-label">Valence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{result['confidence']:.2f}</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations for individual song
                col1, col2 = st.columns(2)
                
                with col1:
                    # Valence-Arousal plot
                    va_fig = create_valence_arousal_plot(
                        result['valence'],
                        result['arousal'],
                        result['primary_emotion']
                    )
                    st.plotly_chart(va_fig, use_container_width=True)
                
                with col2:
                    # Emotion probabilities bar chart
                    emotions = list(result['emotion_probabilities'].keys())
                    probs = list(result['emotion_probabilities'].values())
                    
                    prob_fig = go.Figure(data=[
                        go.Bar(x=emotions, y=probs, marker_color='#00bcd4')
                    ])
                    prob_fig.update_layout(
                        title="Emotion Probabilities",
                        xaxis_title="Emotions",
                        yaxis_title="Probability",
                        plot_bgcolor="#1a1a1a",
                        paper_bgcolor="#1a1a1a",
                        font=dict(color="#ffffff", family="Inter", size=12)
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)
                
                # Spectrogram
                st.markdown("---")
                st.markdown("""
                <div class="premium-card animate-slide" style="margin: 2rem 0;">
                    <h3 style="margin-top: 0; color: #00bcd4; display: flex; align-items: center; gap: 0.5rem;">
                        <span style="display: inline-block; width: 8px; height: 8px; background: #00bcd4; border-radius: 50%; animation: pulse 2s infinite;"></span>
                        Mel Spectrogram Analysis
                    </h3>
                    <p style="margin-bottom: 1rem; color: rgba(255, 255, 255, 0.8); font-size: 1.1rem;">
                        Visual representation of your audio's frequency content over time - the digital fingerprint of your music's emotional journey.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create consistent layout for spectrogram
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    try:
                        with st.spinner("Generating mel spectrogram visualization..."):
                            spectrogram_path = st.session_state.predictor.generate_spectrogram(result['temp_path'])
                            if spectrogram_path and os.path.exists(spectrogram_path):
                                st.image(spectrogram_path, 
                                       caption="Mel Spectrogram - Frequency analysis revealing the emotional structure of your music",
                                       use_container_width=True)
                                # Clean up spectrogram file
                                try:
                                    os.unlink(spectrogram_path)
                                except:
                                    pass
                            else:
                                st.warning("Could not generate spectrogram visualization. Audio analysis will continue without visual representation.")
                    except Exception as e:
                        st.error(f"Spectrogram generation error: {str(e)}")
                        st.info("Spectrogram analysis requires matplotlib and librosa. The emotional analysis will continue without visualization.")
                
                with col2:
                    st.markdown("""
                    <div class="premium-card">
                        <h4 style="margin-top: 0; color: #00ffff;">Spectrogram Info</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("The mel spectrogram shows how different frequencies in your music change over time. Brighter colors indicate stronger frequency components, revealing the harmonic structure that contributes to emotional perception.")
                
                # Individual EQ Generation
                st.markdown("---")
                st.markdown("### Personalized EQ for This Song")
                
                eq_data = st.session_state.predictor.generate_eq_curves(
                    result['primary_emotion'],
                    aggression
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    eq_fig = create_eq_visualization(eq_data, result['primary_emotion'])
                    st.plotly_chart(eq_fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div class="premium-card">
                        <h4 style="margin-top: 0; color: #00bcd4;">EQ Description</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(eq_data['description'])
                
                # EQ Export Section - Full width for better visibility
                st.markdown("---")
                st.markdown("### EQ Export")
                
                # Create columns for EQ export
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # EQ Export for individual song
                    if eq_style == "Wavelet":
                        st.markdown("#### Wavelet EQ Export")
                        st.markdown(f'<div class="eq-container">{eq_data["wavelet"]}</div>', unsafe_allow_html=True)
                    elif eq_style == "Parametric":
                        st.markdown("#### Parametric EQ Settings")
                        st.markdown(f'<div class="eq-container">{eq_data["parametric"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown("#### Graphic EQ Export")
                        st.markdown(f'<div class="eq-container">{eq_data["wavelet"]}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="premium-card">
                        <h4 style="margin-top: 0; color: #00bcd4;">Export Instructions</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("Copy the EQ string and paste it into your audio software or equalizer app. The format is compatible with most professional audio tools and many consumer applications.")
                    
                    # Download EQ as text file
                    if eq_style == "Wavelet":
                        eq_filename = f"eq_{result['primary_emotion']}_{result['filename'].split('.')[0]}_wavelet.txt"
                        st.download_button(
                            label="Download Wavelet EQ File",
                            data=eq_data["wavelet"],
                            file_name=eq_filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                    elif eq_style == "Parametric":
                        eq_filename = f"eq_{result['primary_emotion']}_{result['filename'].split('.')[0]}_parametric.txt"
                        st.download_button(
                            label="Download Parametric EQ File",
                            data=eq_data["parametric"],
                            file_name=eq_filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                    else:
                        eq_filename = f"eq_{result['primary_emotion']}_{result['filename'].split('.')[0]}_graphic.txt"
                        st.download_button(
                            label="Download Graphic EQ File",
                            data=eq_data["wavelet"],
                            file_name=eq_filename,
                            mime="text/plain",
                            use_container_width=True
                        )
                
                # EQed Audio Download
                st.markdown("---")
                st.markdown("### Download EQed Audio")
                try:
                    eqed_audio = st.session_state.predictor.apply_eq_to_audio(
                        result['temp_path'], 
                        result['primary_emotion'], 
                        aggression
                    )
                    if eqed_audio:
                        st.download_button(
                            label=f"Download EQed - {result['filename']}",
                            data=eqed_audio,
                            file_name=f"eqed_{result['primary_emotion']}_{result['filename'].split('.')[0]}.wav",
                            mime="audio/wav"
                        )
                except Exception as e:
                    st.error(f"Could not generate EQed audio: {e}")
                
                # Spotify Mix Button for Individual Analysis
                st.markdown("---")
                search_query = f"{result['primary_emotion']} mix"
                spotify_url = f"https://open.spotify.com/search/{urllib.parse.quote(search_query)}"
                
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
                            <div style="
                                background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
                                color: #000000;
                                font-family: 'Roboto', sans-serif;
                                font-weight: 500;
                                border-radius: 8px;
                                padding: 0.6rem 1rem;
                                font-size: 0.9rem;
                                display: inline-block;
                                cursor: pointer;
                                box-shadow: 0 4px 12px rgba(29, 185, 84, 0.25);
                                transition: all 0.2s ease;
                                border: none;
                                min-width: 140px;
                            ">
                                Find {result['primary_emotion'].title()} Mix
                            </div>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif analysis_mode == "batch" and 'analysis_results' in st.session_state:
                # Batch Analysis Display (existing code)
                results = st.session_state.analysis_results
                
                # Main metrics
                st.markdown("---")
                st.markdown("### Batch Analysis Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{results['total_songs']}</div>
                        <div class="metric-label">Songs Analyzed</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{results['dominant_emotion'].title()}</div>
                        <div class="metric-label">Dominant Emotion</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{results['average_valence']:.2f}</div>
                        <div class="metric-label">Average Valence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container animate-in">
                        <div class="metric-value">{results['average_arousal']:.2f}</div>
                        <div class="metric-label">Average Arousal</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Valence-Arousal plot
                    va_fig = create_valence_arousal_plot(
                        results['average_valence'],
                        results['average_arousal'],
                        results['dominant_emotion']
                    )
                    st.plotly_chart(va_fig, use_container_width=True)
                
                with col2:
                    # Emotion distribution
                    if len(results['emotion_distribution']) > 1:
                        dist_fig = create_emotion_distribution_chart(results['emotion_distribution'])
                        st.plotly_chart(dist_fig, use_container_width=True)
                    else:
                        st.info("All songs have the same emotion classification.")
                
                # Individual song breakdown for batch analysis
                st.markdown("### Individual Song Breakdown")
                
                if 'individual_results' in st.session_state:
                    df_data = []
                    for i, result in enumerate(st.session_state.individual_results):
                        df_data.append({
                            'Song': result['filename'],
                            'Primary Emotion': result['primary_emotion'].title(),
                            'Secondary Emotion': result['secondary_emotion'].title(),
                            'Valence': f"{result['valence']:.2f}",
                            'Arousal': f"{result['arousal']:.2f}",
                            'Confidence': f"{result['confidence']:.2f}"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Song selector for detailed analysis
                    st.markdown("---")
                    st.markdown("### Detailed Song Analysis")
                    
                    selected_song = st.selectbox(
                        "Select song for detailed analysis:",
                        range(len(st.session_state.individual_results)),
                        format_func=lambda x: st.session_state.individual_results[x]['filename']
                    )
                    
                    selected_result = st.session_state.individual_results[selected_song]
                    
                    # Spectrogram for selected song
                    st.markdown(f"""
                    <div class="premium-card animate-slide" style="margin: 2rem 0;">
                        <h3 style="margin-top: 0; color: #00bcd4; display: flex; align-items: center; gap: 0.5rem;">
                            <span style="display: inline-block; width: 8px; height: 8px; background: #00bcd4; border-radius: 50%; animation: pulse 2s infinite;"></span>
                            Mel Spectrogram Analysis - {selected_result['filename']}
                        </h3>
                        <p style="margin-bottom: 1rem; color: rgba(255, 255, 255, 0.8); font-size: 1.1rem;">
                            Visual representation of your audio's frequency content over time - the digital fingerprint of your music's emotional journey.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create consistent layout for spectrogram
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        try:
                            with st.spinner("Generating mel spectrogram visualization..."):
                                spectrogram_path = st.session_state.predictor.generate_spectrogram(selected_result['temp_path'])
                                if spectrogram_path and os.path.exists(spectrogram_path):
                                    st.image(spectrogram_path, 
                                           caption=f"Mel Spectrogram - {selected_result['filename']}",
                                           use_container_width=True)
                                    # Clean up spectrogram file
                                    try:
                                        os.unlink(spectrogram_path)
                                    except:
                                        pass
                                else:
                                    st.warning("Could not generate spectrogram visualization. Audio analysis will continue without visual representation.")
                        except Exception as e:
                            st.error(f"Spectrogram generation error: {str(e)}")
                            st.info("Spectrogram analysis requires matplotlib and librosa. The emotional analysis will continue without visualization.")
                    
                    with col2:
                        st.markdown("""
                        <div class="premium-card">
                            <h4 style="margin-top: 0; color: #00bcd4;">Spectrogram Info</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("The mel spectrogram shows how different frequencies in your music change over time. Brighter colors indicate stronger frequency components, revealing the harmonic structure that contributes to emotional perception.")
                        
                        # Show selected song details
                        st.markdown(f"""
                        <div class="premium-card">
                            <h4 style="margin-top: 0; color: #00bcd4;">Song Details</h4>
                            <p><strong>Primary Emotion:</strong> {selected_result['primary_emotion'].title()}</p>
                            <p><strong>Secondary Emotion:</strong> {selected_result['secondary_emotion'].title()}</p>
                            <p><strong>Valence:</strong> {selected_result['valence']:.2f}</p>
                            <p><strong>Arousal:</strong> {selected_result['arousal']:.2f}</p>
                            <p><strong>Confidence:</strong> {selected_result['confidence']:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # EQ Generation for batch
                st.markdown("---")
                st.markdown("### Aggregate EQ Settings")
                
                eq_data = st.session_state.predictor.generate_eq_curves(
                    results['dominant_emotion'],
                    aggression
                )
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    eq_fig = create_eq_visualization(eq_data, results['dominant_emotion'])
                    st.plotly_chart(eq_fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div class="premium-card">
                        <h4 style="margin-top: 0; color: #00bcd4;">EQ Description</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info(eq_data['description'])
                    
                    # EQ Export
                    if eq_style == "Wavelet":
                        st.markdown("#### Wavelet EQ Export")
                        st.markdown(f'<div class="eq-container">{eq_data["wavelet"]}</div>', unsafe_allow_html=True)
                        
                        # Download button for Wavelet EQ
                        eq_filename = f"batch_eq_{results['dominant_emotion']}_wavelet.txt"
                        st.download_button(
                            label="Download Batch Wavelet EQ",
                            data=eq_data["wavelet"],
                            file_name=eq_filename,
                            mime="text/plain",
                            use_container_width=True,
                            key="batch_wavelet_download"
                        )
                    elif eq_style == "Parametric":
                        st.markdown("#### Parametric EQ Settings")
                        st.markdown(f'<div class="eq-container">{eq_data["parametric"]}</div>', unsafe_allow_html=True)
                        
                        # Download button for Parametric EQ
                        eq_filename = f"batch_eq_{results['dominant_emotion']}_parametric.txt"
                        st.download_button(
                            label="Download Batch Parametric EQ",
                            data=eq_data["parametric"],
                            file_name=eq_filename,
                            mime="text/plain",
                            use_container_width=True,
                            key="batch_parametric_download"
                        )
                    else:
                        st.markdown("#### Graphic EQ Export")
                        st.markdown(f'<div class="eq-container">{eq_data["wavelet"]}</div>', unsafe_allow_html=True)
                        
                        # Download button for Graphic EQ
                        eq_filename = f"batch_eq_{results['dominant_emotion']}_graphic.txt"
                        st.download_button(
                            label="Download Batch Graphic EQ",
                            data=eq_data["wavelet"],
                            file_name=eq_filename,
                            mime="text/plain",
                            use_container_width=True,
                            key="batch_graphic_download"
                        )
                
                # Batch EQed Audio Downloads
                st.markdown("---")
                st.markdown("### Download All EQed Songs")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    <div class="premium-card">
                        <h4 style="margin-top: 0; color: #00bcd4;">Individual Downloads</h4>
                        <p>Download each song with its individual emotion-based EQ applied:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for i, result in enumerate(st.session_state.individual_results):
                        try:
                            eqed_audio = st.session_state.predictor.apply_eq_to_audio(
                                result['temp_path'], 
                                result['primary_emotion'], 
                                aggression
                            )
                            if eqed_audio:
                                st.download_button(
                                    label=f"{result['filename']} ({result['primary_emotion'].title()})",
                                    data=eqed_audio,
                                    file_name=f"eqed_{result['primary_emotion']}_{result['filename'].split('.')[0]}.wav",
                                    mime="audio/wav",
                                    key=f"individual_eqed_{i}",
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.warning(f"Could not generate EQ for {result['filename']}: {str(e)}")
                
                with col2:
                    st.markdown("""
                    <div class="premium-card">
                        <h4 style="margin-top: 0; color: #00bcd4;">Batch EQ Summary</h4>
                        <p>Download summary of all EQ settings applied:</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create EQ summary for all songs
                    eq_summary = f"ApollodB Batch EQ Analysis Summary\n"
                    eq_summary += f"=====================================\n\n"
                    eq_summary += f"Batch Analysis Results:\n"
                    eq_summary += f"- Total Songs: {results['total_songs']}\n"
                    eq_summary += f"- Dominant Emotion: {results['dominant_emotion'].title()}\n"
                    eq_summary += f"- Average Valence: {results['average_valence']:.2f}\n"
                    eq_summary += f"- Average Arousal: {results['average_arousal']:.2f}\n"
                    eq_summary += f"- EQ Aggression: {aggression}\n"
                    eq_summary += f"- EQ Style: {eq_style}\n\n"
                    
                    eq_summary += f"Recommended EQ Settings for {results['dominant_emotion'].title()} Music:\n"
                    eq_summary += f"{'='*50}\n"
                    eq_summary += f"{eq_data['wavelet']}\n\n"
                    
                    eq_summary += f"Individual Song Analysis:\n"
                    eq_summary += f"{'='*25}\n"
                    for i, result in enumerate(st.session_state.individual_results):
                        eq_summary += f"{i+1}. {result['filename']}\n"
                        eq_summary += f"   Primary Emotion: {result['primary_emotion'].title()}\n"
                        eq_summary += f"   Secondary Emotion: {result['secondary_emotion'].title()}\n"
                        eq_summary += f"   Valence: {result['valence']:.2f}\n"
                        eq_summary += f"   Arousal: {result['arousal']:.2f}\n"
                        eq_summary += f"   Confidence: {result['confidence']:.2f}\n\n"
                    
                    st.download_button(
                        label="Download Complete Analysis Report",
                        data=eq_summary,
                        file_name=f"apollodb_batch_analysis_{results['dominant_emotion']}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="batch_summary_download"
                    )
                
                # Spotify Mix Button
                st.markdown("---")
                search_query = f"{results['dominant_emotion']} mix"
                spotify_url = f"https://open.spotify.com/search/{urllib.parse.quote(search_query)}"
                
                col1, col2, col3 = st.columns([2, 1, 2])
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 1rem 0;">
                        <a href="{spotify_url}" target="_blank" style="text-decoration: none;">
                            <div style="
                                background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
                                color: #000000;
                                font-family: 'Roboto', sans-serif;
                                font-weight: 500;
                                border-radius: 8px;
                                padding: 0.6rem 1rem;
                                font-size: 0.9rem;
                                display: inline-block;
                                cursor: pointer;
                                box-shadow: 0 4px 12px rgba(29, 185, 84, 0.25);
                                transition: all 0.2s ease;
                                border: none;
                                min-width: 140px;
                            ">
                                Find {results['dominant_emotion'].title()} Mix
                            </div>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <div class="premium-card animate-in" style="max-width: 1000px; margin: 0 auto;">
            <h3 style="margin-top: 0; color: #00bcd4; text-align: center;">IEM Database - Curated Recommendations</h3>
            <p style="margin-bottom: 0; text-align: center;">Understand how your music analysis relates to carefully curated IEM frequency responses. These are hand-picked recommendations from an extensive database of tested IEMs, focusing only on the highest-value options. This is where science meets sound - finding the perfect match between your emotional music preferences and the gear that brings them to life!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Curated IEMs from parikshit's spreadsheet (starred recommendations only)
        iems = {
            # TOTL/Summit-Fi ()
            "LETSHUOER MYSTIC 8": {"price": "$1000", "signature": "Vocal benchmark", "valence_match": ["happy", "calm"], "notes": "Female vocal benchmark under $2000, clean & airy"},
            "Thieaudio Monarch MK4": {"price": "$1150", "signature": "All-rounder", "valence_match": ["neutral", "happy"], "notes": "Smooth, clean, balanced with tuning options"},
            
            # High Value Stars ()
            "XENNS TOP PRO": {"price": "$499", "signature": "Technical all-rounder", "valence_match": ["neutral", "happy"], "notes": "Amazing resolution & detail, direct upgrade from Astrals"},
            "ZIIGAAT Arcanis": {"price": "$399", "signature": "Vocal-centric", "valence_match": ["calm", "sad"], "notes": "Best vocals under $500, genre-specific for slower tracks"},
            "ZIIGAAT Luna": {"price": "$379", "signature": "Warm & dreamy", "valence_match": ["calm", "neutral"], "notes": "Airy, smooth, lush vibes - great for rock/metal"},
            "ZIIGAAT Odyssey": {"price": "$229", "signature": "Musical journey", "valence_match": ["calm", "sad"], "notes": "Mini Subtonic Storm that scales, immersive for indie/ballads"},
            "Softears Volume S": {"price": "$319", "signature": "Vocal scaling", "valence_match": ["happy", "calm"], "notes": "Amazing vocal scaling at high volume"},
            "Kiwi Ears x HBB PUNCH": {"price": "$449", "signature": "Balanced basshead", "valence_match": ["happy", "neutral"], "notes": "Endgame bass with extended vocals & treble"},
            
            # Great Value ()
            "Kiwi Ears Astral": {"price": "$299", "signature": "Balanced fun", "valence_match": ["neutral", "happy"], "notes": "Great all-rounder, airy with good sub-bass"},
            "SIMGOT SUPERMIX4": {"price": "$150", "signature": "Smooth Harman", "valence_match": ["neutral", "calm"], "notes": "Endgame Harman, one of the smoothest IEMs"},
            "Truthear NOVA": {"price": "$150", "signature": "Clean Harman", "valence_match": ["neutral", "happy"], "notes": "Pinnacle of trying not to offend anyone, smooth treble"},
            "ARTTI T10": {"price": "$50", "signature": "Planar value", "valence_match": ["neutral", "happy"], "notes": "Insane value, almost identical to S12"},
            
            # Budget Champions ()
            "TangZu Xuan Wu Gate": {"price": "$650", "signature": "Clean technical", "valence_match": ["neutral", "calm"], "notes": "Neutral tuning done right, very detailed"},
            "HIDIZS MK12": {"price": "$129-209", "signature": "High-volume scaling", "valence_match": ["calm", "sad"], "notes": "Insane scaling, warm & non-fatiguing"},
            "LETSHUOER DX1": {"price": "$159", "signature": "Dynamic vocals", "valence_match": ["happy", "calm"], "notes": "Vibrant vocals with good balance"},
            "ZIIGAAT LUSH": {"price": "$179", "signature": "Clean technical", "valence_match": ["calm", "neutral"], "notes": "Fuller sound with scaling, immersive"},
            "EPZ P50": {"price": "$200", "signature": "Clean balanced", "valence_match": ["neutral", "calm"], "notes": "Better tuned MEGA5EST, dynamic"},
            "Punch Audio Martilo": {"price": "$329", "signature": "Balanced basshead", "valence_match": ["happy", "neutral"], "notes": "Cheaper HBB Punch with slightly less bass"},
            "EPZ K9": {"price": "$300", "signature": "All-rounder v-shape", "valence_match": ["neutral", "happy"], "notes": "Natural vocals, nice mid-bass slam"},
            "CrinEar Meta": {"price": "$249", "signature": "Bright sparkly", "valence_match": ["happy", "neutral"], "notes": "Bright all-rounder with sparkly treble"},
            "Simgot EM6L Phoenix": {"price": "$109", "signature": "Smooth warm", "valence_match": ["calm", "neutral"], "notes": "Great resolution, slightly warm but lively"},
            "Simgot EA500LM": {"price": "$89", "signature": "Warm resolving", "valence_match": ["calm", "happy"], "notes": "Very resolving for price, warmer EA1000"},
            "Hidizs MP145": {"price": "$159", "signature": "Tame Harman", "valence_match": ["neutral", "calm"], "notes": "Less sharp HeyDay, solid all-rounder"},
            "TinHifi P1 MAX 2": {"price": "$139", "signature": "Smooth planar", "valence_match": ["neutral", "calm"], "notes": "Less bright Nova, good for jpop/kpop"},
            "MYER SLA3": {"price": "$100", "signature": "Dynamic engaging", "valence_match": ["neutral", "happy"], "notes": "Dynamic contrast, more engaging than safe picks"},
            "7hertz Sonus": {"price": "$59", "signature": "Neutral balanced", "valence_match": ["neutral", "calm"], "notes": "Hexa with more air, solid neutral set"},
            "EPZ Q1 PRO": {"price": "$30", "signature": "Clean balanced", "valence_match": ["neutral", "calm"], "notes": "OG 7hz Zero with better driver"},
            "ARTTI R2": {"price": "$35", "signature": "Warm balanced", "valence_match": ["calm", "neutral"], "notes": "Budget Tanchjim Origin, very smooth"},
            "Simgot EW200": {"price": "$39", "signature": "Cheaper Aria 2", "valence_match": ["neutral", "calm"], "notes": "Same driver as Aria 2, great value"},
            "KZ EDC PRO": {"price": "$22", "signature": "V-shape smooth", "valence_match": ["happy", "neutral"], "notes": "Nicely tuned v-shape, but it's KZ"},
            "KBear Rosefinch": {"price": "$20", "signature": "Budget basshead", "valence_match": ["happy", "neutral"], "notes": "Best budget basshead, slams like a truck"},
            "Simgot EW100P": {"price": "$20", "signature": "Clean neutral", "valence_match": ["neutral", "calm"], "notes": "Another Harman/DF, great starter"},
            "TangZu Waner 2": {"price": "$20", "signature": "Balanced all-rounder", "valence_match": ["neutral", "calm"], "notes": "More treble air than OG, great accessories"},
            "Moondrop CHU 2": {"price": "$19", "signature": "Harman-ish", "valence_match": ["neutral", "calm"], "notes": "Similar to Tanchjim One but not as smooth"},
            "Tangzu Wan'er": {"price": "$20", "signature": "Vocal forward", "valence_match": ["happy", "calm"], "notes": "Most vocal forward of $20 sets"},
            "Truthear Gate": {"price": "$20", "signature": "All-rounder", "valence_match": ["neutral", "calm"], "notes": "Less uppermids than EW100P"},
            "ZIIGAAT NUO": {"price": "$25", "signature": "Clean Harman", "valence_match": ["neutral", "calm"], "notes": "Similar to G10, heftier note-weight"},
            "TangZu Wan'er S.G": {"price": "$21", "signature": "Clean vocal", "valence_match": ["neutral", "happy"], "notes": "Cleaner Waner with more vocal emphasis"},
            "QKZ HBB": {"price": "$20", "signature": "Warm bassy", "valence_match": ["calm", "happy"], "notes": "Well tuned warm/bassy set"}
        }
        
        if 'analysis_results' in st.session_state:
            dominant_emotion = st.session_state.analysis_results['dominant_emotion']
            
            st.markdown(f"#### Recommended IEMs for {dominant_emotion.title()} Music")
            
            recommended = [name for name, data in iems.items() if dominant_emotion in data['valence_match']]
            
            if recommended:
                # Create cards for recommended IEMs
                num_cols = min(3, len(recommended))
                cols = st.columns(num_cols)
                
                for i, iem in enumerate(recommended[:6]):  # Show max 6 recommendations
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        price = iems[iem].get('price', 'N/A')
                        signature = iems[iem]['signature']
                        notes = iems[iem].get('notes', 'Great choice for your music style')
                        
                        st.markdown(f"""
                        <div class="premium-card" style="margin-bottom: 1rem; min-height: 120px;">
                            <h4 style="color: #00bcd4; margin-top: 0; font-size: 1.1rem;">{iem}</h4>
                            <p style="margin: 0.8rem 0; font-size: 1rem;"><strong>Price:</strong> {price}</p>
                            <p style="margin-top: 1rem; font-size: 0.9rem; color: rgba(255, 255, 255, 0.9);"><strong>Best for:</strong> {', '.join(iems[iem]['valence_match']).title()}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info(f"No specific IEM recommendations for {dominant_emotion} emotion profile in the database.")
        
        # Price category tabs
        st.markdown("### Browse by Price Category")
        
        price_tabs = st.tabs(["Under $100", "$100-500", "Over $500", "All IEMs"])
        
        with price_tabs[0]:  # Budget
            budget_iems = {k: v for k, v in iems.items() 
                          if any(char.isdigit() for char in v.get('price', '')) and 
                          int(''.join(filter(str.isdigit, v.get('price', '0')))) <= 100}
            
            if budget_iems:
                df_budget = pd.DataFrame([
                    {"IEM": name, "Price": data.get("price", "N/A"), 
                     "Best For": ", ".join(data["valence_match"]).title()}
                    for name, data in budget_iems.items()
                ])
                st.dataframe(df_budget, use_container_width=True)
            else:
                st.info("No IEMs under $100 in recommendations.")
        
        with price_tabs[1]:  # Mid-Fi
            midfi_iems = {k: v for k, v in iems.items() 
                         if any(char.isdigit() for char in v.get('price', '')) and 
                         100 < int(''.join(filter(str.isdigit, v.get('price', '0')))) <= 500}
            
            if midfi_iems:
                df_midfi = pd.DataFrame([
                    {"IEM": name, "Price": data.get("price", "N/A"), 
                     "Best For": ", ".join(data["valence_match"]).title()}
                    for name, data in midfi_iems.items()
                ])
                st.dataframe(df_midfi, use_container_width=True)
            else:
                st.info("No IEMs in $100-500 range in recommendations.")
        
        with price_tabs[2]:  # Hi-Fi
            hifi_iems = {k: v for k, v in iems.items() 
                        if any(char.isdigit() for char in v.get('price', '')) and 
                        int(''.join(filter(str.isdigit, v.get('price', '0')))) > 500}
            
            if hifi_iems:
                df_hifi = pd.DataFrame([
                    {"IEM": name, "Price": data.get("price", "N/A"), 
                     "Best For": ", ".join(data["valence_match"]).title()}
                    for name, data in hifi_iems.items()
                ])
                st.dataframe(df_hifi, use_container_width=True)
            else:
                st.info("No IEMs over $500 in recommendations.")
        
        with price_tabs[3]:  # All
            df_all = pd.DataFrame([
                {"IEM": name, "Price": data.get("price", "N/A"), 
                 "Best For": ", ".join(data["valence_match"]).title()}
                for name, data in iems.items()
            ])
            st.dataframe(df_all, use_container_width=True)
        
        st.info("Visit squig.link for detailed frequency response measurements of these IEMs - it's the go-to resource for audiophiles who want the raw data behind their gear! All recommendations above are carefully curated from an extensive testing database and represent only the highest-value options.")
    
    with tab3:
        st.markdown("""
        <div class="premium-card animate-in" style="max-width: 800px; margin: 0 auto;">
            <h3 style="margin-top: 0; color: #00bcd4; text-align: center;">Frequently Asked Questions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        faqs = [
            {
                "q": "How does ApollodB work?",
                "a": "ApollodB uses a deep learning model trained on the DEAM dataset to analyze the emotional content of your music. It extracts spectrograms from audio files and predicts emotions based on valence and arousal dimensions. The best part? It's completely open source, so you can peek under the hood and see exactly how the magic happens!"
            },
            {
                "q": "How is this better than Spotify's recommendations?",
                "a": "While Spotify uses collaborative filtering and basic audio features, ApollodB performs deep emotional analysis of the actual audio content. It provides personalized EQ settings and detailed emotional profiling that goes beyond simple genre categorization. Plus, it doesn't try to sell you anything - it just wants to make your music sound amazing!"
            },
            {
                "q": "What makes this approach innovative?",
                "a": "ApollodB combines state-of-the-art music emotion recognition with personalized audio engineering. It's one of the first systems to automatically generate EQ curves based on emotional analysis, bridging the gap between psychology and audio engineering. And did we mention it's open source? Because transparency in AI is everything."
            },
            {
                "q": "What audio formats are supported?",
                "a": "ApollodB supports MP3, WAV, M4A, and FLAC audio formats. Files should ideally be of good quality for best analysis results. We're not picky, but your music deserves the best treatment!"
            },
            {
                "q": "How accurate is the emotion detection?",
                "a": "The model achieves competitive accuracy on the DEAM dataset across 4 core emotions (neutral, happy, sad, calm). The system employs sophisticated bias handling to ensure accurate emotion detection. It's not perfect, but it's pretty darn good at understanding what makes your heart sing!"
            },
            {
                "q": "What does the aggression slider do?",
                "a": "The aggression slider controls how pronounced the EQ adjustments will be. A higher setting creates more dramatic frequency adjustments, while a lower setting provides subtle corrections. Think of it as the difference between a gentle nudge and a full transformation!"
            },
            {
                "q": "Can I use these EQ settings on any device?",
                "a": "The EQ curves can be exported in multiple formats (Wavelet, Parametric, Graphic) to work with most audio software and hardware equalizers. Whether you're using a fancy DAC or just your phone, we've got you covered!"
            },
            {
                "q": "What is valence and arousal?",
                "a": "Valence represents the positivity/negativity of emotions (sad to happy), while arousal represents the energy level (calm to excited). Together, they create a 2D emotional space that captures the complexity of musical emotions. It's like a GPS for feelings!"
            },
            {
                "q": "How does the IEM recommendation work?",
                "a": "Based on your music's emotional profile, ApollodB suggests IEMs whose frequency response characteristics complement your listening preferences. This is based on acoustic research linking frequency response to emotional perception. Science meets sound, and your ears win!"
            },
            {
                "q": "Is my audio data stored or shared?",
                "a": "No, all audio analysis is performed locally. Your files are temporarily processed for analysis and immediately deleted afterward. No audio data is stored or transmitted. Your music stays yours, as it should be!"
            }
        ]
        
        for i, faq in enumerate(faqs):
            with st.expander(f"**{faq['q']}**"):
                st.write(faq['a'])
    
    with tab4:
        st.markdown("""
        <div class="premium-card animate-in" style="max-width: 800px; margin: 0 auto;">
            <h3 style="margin-top: 0; color: #00bcd4; text-align: center;">References</h3>
        </div>
        """, unsafe_allow_html=True)
        
        references = [
            {
                "citation": "Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. PLOS ONE, 12(3), e0173392. https://doi.org/10.1371/journal.pone.0173392",
                "description": "The primary dataset used for training our emotion recognition model."
            },
            {
                "citation": "Chen, Y., Ma, Z., Wang, M., & Liu, M. (2024). Advancing music emotion recognition: A transformer encoder-based approach. In Proceedings of the 6th ACM International Conference on Multimedia in Asia (MMAsia '24) (Article 60, pp. 1–5). https://doi.org/10.1145/3696409.3700221",
                "description": "Modern transformer-based approaches to music emotion recognition."
            },
            {
                "citation": "Ghazarian, S., Wen, N., Galstyan, A., & Peng, N. (2022). DEAM: Dialogue Coherence Evaluation using AMR-based Semantic Manipulations. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022).",
                "description": "Advanced evaluation methodologies for coherence in dialogue systems."
            },
            {
                "citation": "Kang, J., & Herremans, D. (2025). Towards unified music emotion recognition across dimensional and categorical models. arXiv preprint arXiv:2502.03979v2. https://arxiv.org/abs/2502.03979",
                "description": "Latest research on unifying different approaches to music emotion recognition."
            },
            {
                "citation": "Soleymani, M., Aljanaki, A., & Yang, Y.-H. (2018). DEAM: MediaEval Database for Emotional Analysis in Music. Swiss Center for Affective Sciences, University of Geneva & Academia Sinica. Presented April 26, 2018.",
                "description": "Comprehensive overview of the DEAM dataset structure and methodology."
            },
            {
                "citation": "Yang, Y.-H., Aljanaki, A., & Soleymani, M. (2024). Are we there yet? A brief survey of music emotion prediction datasets, models and outstanding challenges. arXiv preprint arXiv:2406.08809. https://arxiv.org/abs/2406.08809",
                "description": "Current state-of-the-art review of music emotion recognition field and future challenges."
            },
            {
                "citation": "PlusLab NLP. (n.d.). DEAM GitHub Repository. https://github.com/PlusLabNLP/DEAM",
                "description": "Open source implementation and resources for the DEAM dataset."
            }
        ]
        
        for i, ref in enumerate(references, 1):
            st.markdown(f"""
            <div class="premium-card animate-in">
                <h4 style="color: #00bcd4; margin-top: 0;">{i}. Reference</h4>
                <p style="margin-bottom: 1rem; line-height: 1.6; font-size: 0.95rem;">{ref['citation']}</p>
                <p style="color: rgba(255, 255, 255, 0.8); font-style: italic; margin-bottom: 0;"><em>{ref['description']}</em></p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab5:
        st.markdown("""
        <div class="premium-card animate-in" style="max-width: 800px; margin: 0 auto;">
            <h3 style="margin-top: 0; color: #00bcd4; text-align: center;">About ApollodB</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="max-width: 800px; margin: 0 auto;">
        
        **ApollodB** is an AI-powered music emotion analysis and EQ optimization system that bridges the gap between 
        psychological research and practical audio engineering. The best part? It's completely open source!
        
        #### What Makes ApollodB Special?
        
        - **Deep Emotional Analysis**: Uses state-of-the-art deep learning trained on the DEAM dataset
        - **Personalized EQ Generation**: Creates custom equalizer curves based on your music's emotional profile
        - **Multiple EQ Formats**: Supports Wavelet, Parametric, and Graphic EQ exports
        - **Valence-Arousal Visualization**: Interactive charts showing where your music lands in emotional space
        - **IEM Recommendations**: Suggests in-ear monitors that complement your listening preferences
        - **Aggregate Analysis**: Analyzes multiple songs to understand your overall musical taste
        - **Open Source Philosophy**: Complete transparency in how your music is analyzed and processed
        
        #### Technical Approach
        
        ApollodB analyzes your music using spectrograms and a convolutional neural network trained on 
        emotional annotations. The system maps predictions to the valence-arousal emotional space and generates 
        personalized EQ curves based on psychoacoustic research.
        
        #### The Science Behind It
        
        Our approach is grounded in decades of research in music psychology, audio engineering, and machine learning. 
        By combining emotional analysis with practical audio optimization, ApollodB offers a unique tool for 
        understanding and enhancing your musical experience. And because it's open source, you can explore every 
        algorithm and contribute to making it even better!
        
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Centered coffee link and footer
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <a href="https://coff.ee/parikshitkumar" target="_blank" style="
                background: linear-gradient(135deg, #FFDD44 0%, #FF6B35 100%);
                color: #000000;
                text-decoration: none;
                padding: 0.8rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                font-size: 1rem;
                display: inline-block;
                transition: all 0.2s ease;
                box-shadow: 0 4px 12px rgba(255, 221, 68, 0.3);
            ">☕ Support Development - Buy Me Coffee</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center;">
            <p style="margin: 0; color: rgba(255, 255, 255, 0.8); font-weight: 500;">Built with passion for music and technology</p>
        </div>
        """, unsafe_allow_html=True)

    # Premium footer
    st.markdown("""
    <div class="premium-footer">
        <p style="margin: 0; color: rgba(255, 255, 255, 0.6);">
            Made in California by Parikshit Kumar | 
            <a href="https://github.com/parikshitkumar/apollodb" style="color: #00bcd4; text-decoration: none;">Open Source</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Cleanup function for session
def cleanup_temp_files():
    """Clean up temporary files from session state"""
    try:
        if 'temp_files' in st.session_state:
            cleaned_count = 0
            for temp_file in st.session_state.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not clean up temp file {temp_file}: {e}")
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            del st.session_state.temp_files
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Health check function
def health_check():
    """Basic health check for the application"""
    try:
        # Check if model is loaded
        if 'predictor' not in st.session_state:
            return False, "Model not loaded"
        
        # Check if required files exist
        required_files = ['best_model.h5', 'scaler_mean.npy', 'scaler_scale.npy', 'labels.json']
        for file in required_files:
            if not os.path.exists(file):
                return False, f"Missing required file: {file}"
        
        return True, "All systems operational"
    except Exception as e:
        return False, f"Health check failed: {e}"

# Register cleanup on app exit
atexit.register(cleanup_temp_files)

if __name__ == "__main__":
    main()

# 2025-08-02T06:30:36.394Z - minor update

// 2025-08-09T04:59:11.415Z - minor update