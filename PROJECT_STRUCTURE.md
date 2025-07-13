# apollodB Project Structure

```
apollov5/
├── app.py                    # Main Streamlit application
├── inference.py              # Core inference engine and EQ generation
├── launch.py                 # Application launcher script
├── demo.py                   # Demo script for testing inference
├── config.json              # Configuration settings
├── requirements.txt         # Python dependencies
├── README.md                # Comprehensive documentation
│
├── Model Files (Pre-trained):
├── best_model.h5            # Trained emotion recognition model
├── scaler_mean.npy          # Feature normalization parameters  
├── scaler_scale.npy         # Feature scaling parameters
├── labels.json              # Emotion label mappings
├── training_emotions.json   # Training emotion categories
├── test_metrics.json        # Model performance metrics
└── training_history.png     # Training visualization
```

## File Descriptions

### Core Application Files

**app.py**
- Main Streamlit web application
- Implements all UI components and user interactions
- Handles file uploads, analysis, and visualization
- Cyberpunk/Greek aesthetic with black/grey/cyan color scheme
- Roboto Bold typography throughout

**inference.py** 
- MusicEmotionPredictor class for emotion analysis
- EQ curve generation algorithms
- Valence-arousal space mapping
- Multi-file analysis aggregation
- Handles bias correction for neutral class

**launch.py**
- Simple launcher script with dependency checking
- Automated requirement installation
- User-friendly error messages and guidance

**demo.py**
- Standalone testing script
- Demonstrates inference functionality without UI
- Useful for debugging and development

### Configuration & Dependencies

**config.json**
- Centralized configuration management
- UI colors, fonts, and styling parameters
- Model paths and audio processing settings
- IEM database and EQ frequency bands

**requirements.txt**
- All Python package dependencies
- Pinned to compatible versions
- Includes ML, audio processing, and web framework libraries

### Model Assets

**best_model.h5**
- TensorFlow/Keras trained emotion recognition model
- Architecture: CNN with attention mechanisms
- Input: Log-mel spectrograms (1300 x 128)
- Output: 5-class emotion probabilities

**scaler_*.npy**
- Feature normalization parameters from training
- Essential for consistent preprocessing
- Ensures model compatibility with new audio data

**labels.json** 
- Emotion category mappings
- Original 5-class system: neutral, happy, sad, angry, calm
- Used for model output interpretation

## Key Features Implemented

### 🎵 Music Analysis
- Multi-format audio support (MP3, WAV, M4A, FLAC)
- 30-second analysis windows for consistency
- Aggregate analysis across multiple songs
- Valence-arousal emotional space mapping

### 🎛️ EQ Generation
- Research-based emotion-to-frequency mappings
- Multiple export formats (Wavelet, Parametric, Graphic)
- Adjustable aggression parameter (0.0-1.0)
- ISO 1/3 octave band precision

### 📊 Visualizations
- Interactive valence-arousal plots with Plotly
- Real-time EQ frequency response curves
- Emotion distribution pie charts
- Statistical analysis dashboards

### 🎧 IEM Integration
- Curated database of popular IEMs
- Signature-based recommendations
- Emotional profile matching
- Squig.link compatibility notes

### 🎵 Music Discovery
- Spotify search integration
- Automated playlist suggestions
- Emotion-based mix recommendations
- Direct deep-linking to streaming platforms

### ❓ User Education
- Comprehensive FAQ section
- Scientific references and citations
- Technical documentation
- Usage guides and tips

## Technical Architecture

### Audio Processing Pipeline
1. **Input**: Raw audio files (various formats)
2. **Preprocessing**: Resample to 22.05kHz, 30s segments
3. **Feature Extraction**: Log-mel spectrograms (128 bands)
4. **Normalization**: Z-score using training statistics
5. **Inference**: CNN emotion prediction
6. **Post-processing**: Bias correction and aggregation

### EQ Generation Algorithm
1. **Emotion Mapping**: Predict primary/secondary emotions
2. **Frequency Profiling**: Map emotions to frequency characteristics
3. **Curve Generation**: Apply psychoacoustic research
4. **Aggression Scaling**: User-controllable intensity
5. **Format Export**: Multiple EQ format outputs

### UI/UX Design Principles
- **Accessibility**: High contrast colors, clear typography
- **Responsiveness**: Works on desktop and mobile devices
- **Performance**: Optimized for real-time analysis
- **Aesthetics**: Cyberpunk meets classical Greek design

## Usage Scenarios

### Personal Music Analysis
1. Upload your music library
2. Discover your emotional listening patterns
3. Generate personalized EQ settings
4. Find new music that matches your taste

### Audio Engineering
1. Analyze client preferences
2. Create custom EQ presets
3. Understand emotional impact of frequency changes
4. Optimize mixing decisions

### Research Applications
1. Music psychology studies
2. Audio preference research
3. Cultural emotion analysis
4. Comparative listening tests

## Future Enhancements

### Planned Features
- Real-time audio input analysis
- Extended IEM database with measurements
- Custom emotion model training
- Social sharing and community features
- Mobile app development
- Plugin development for DAWs

### Technical Improvements
- Model accuracy optimization
- Additional emotion dimensions
- Genre-specific analysis
- Cultural adaptation capabilities
- Cloud processing options

---

**apollodB v1.0.0 - Built with ❤️ in California by Parikshit Kumar**
