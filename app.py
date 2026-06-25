import streamlit as st
import pandas as pd
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# ==========================================
# LOAD MODELS & ENCODERS
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load("placement_model.pkl")
    gender_encoder = joblib.load("gender_encoder.pkl")
    stream_encoder = joblib.load("stream_encoder.pkl")
    return model, gender_encoder, stream_encoder

model, gender_encoder, stream_encoder = load_models()

# ==========================================
# CUSTOM CSS SYSTEM (COMPACT & HIGH SPECIFICITY)
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Page Background */
.stApp {
    background-color: #f1f5f9;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.18) 0, transparent 50%), 
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.18) 0, transparent 50%),
        radial-gradient(at 50% 100%, rgba(59, 130, 246, 0.12) 0, transparent 60%);
    font-family: 'Outfit', sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2rem !important; max-width: 760px !important; }

/* Page Header Banner */
.header-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    padding: 24px 36px; border-radius: 20px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 25px; box-shadow: 0 10px 25px rgba(30,27,75,0.15);
}
.header-text { max-width: 85%; }
.header-title { font-size: 26px !important; font-weight: 700 !important; color: #ffffff !important; margin: 0 !important; }
.header-subtitle { font-size: 14px !important; color: #c7d2fe !important; margin-top: 6px !important; }
.header-illustration-cap { font-size: 55px; opacity: 0.9; }

/* Default Cards Wrapper Style */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid rgba(226, 232, 240, 0.8) !important;
    border-top: 4px solid #6366f1 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03) !important;
    margin-bottom: 20px !important;
}

/* Target Student Details card using the trigger div */
div[data-testid="stVerticalBlockBorderWrapper"]:has(#student-details-card-trigger) {
    background-color: #f8fafc !important; /* Soft blue-slate background */
    border-top: 4px solid #4f46e5 !important;
}

.card-header-row { display: flex; align-items: center; gap: 8px; }
.card-header-icon { font-size: 18px; }
.card-header-title { font-size: 17px; font-weight: 700; color: #1e1b4b; }
.card-subtitle { font-size: 12.5px; color: #64748b; margin-bottom: 20px; }

/* Form Fields & Label Styling */
div[data-testid="stSlider"] label, div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label {
    font-size: 13.5px; font-weight: 600; color: #334155 !important;
}
div[data-baseweb="select"] { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }

/* Override Red/Default Sliders */
div[data-testid="stSlider"] [data-testid="stSliderTrack"] > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
}
div[data-testid="stSlider"] [data-testid="stSliderThumb"], 
div[data-testid="stSlider"] div[role="slider"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: 2px solid #ffffff !important;
    box-shadow: 0 3px 8px rgba(79, 70, 229, 0.3) !important;
}
div[data-testid="stSlider"] div[class*="StyledThumbValue"], 
div[data-testid="stSlider"] [data-testid="stWidgetLabel"] + div {
    color: #4f46e5 !important;
    font-weight: 700 !important;
}

div[data-testid="stNumberInput"] input { background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; height: 40px; }
div[data-testid="stNumberInput"] button { background-color: #e2e8f0 !important; color: #1e293b !important; }
div[data-testid="stNumberInput"] button:hover { background-color: #6366f1 !important; color: white !important; }

/* Predict button */
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    color: #ffffff !important; border: none !important; border-radius: 10px !important;
    height: 46px !important; font-size: 15px !important; font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25) !important; margin-top: 10px !important; width: 100%;
}
div[data-testid="stButton"] button:hover {
    background: linear-gradient(90deg, #4338ca, #6d28d9) !important; transform: translateY(-1px) !important;
}

/* Results styles */
.result-badge { display: flex; align-items: center; gap: 12px; padding: 14px 18px; border-radius: 12px; margin-bottom: 12px; }
.result-badge.placed { background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; }
.result-badge.not-placed { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
.badge-icon { font-size: 16px; display: flex; align-items: center; justify-content: center; width: 28px; height: 28px; border-radius: 50%; color: white; font-weight: bold; }
.placed .badge-icon { background: #10b981; }
.not-placed .badge-icon { background: #ef4444; }
.badge-title { font-size: 17px; font-weight: 700; }
.badge-subtitle { font-size: 12px; }

.profile-rating-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: 20px; font-size: 12.5px; font-weight: 600; margin-bottom: 20px; }
.rating-exceptional { background-color: #fef3c7; border: 1px solid #fde68a; color: #92400e; }
.rating-competitive { background-color: #dbeafe; border: 1px solid #bfdbfe; color: #1e40af; }
.rating-critical { background-color: #fee2e2; border: 1px solid #fecaca; color: #991b1b; }

/* Placement Probability Card (Premium Layout) */
.probability-card {
    border: 1px solid #e2e8f0;
    background: #ffffff;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.01);
}
.probability-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.probability-label {
    font-size: 12px;
    font-weight: 700;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.probability-value {
    font-size: 32px;
    font-weight: 700;
    color: #1e1b4b;
}
.probability-gauge-wrapper {
    position: relative;
    width: 80px;
    height: 80px;
}
.probability-gauge-wrapper svg {
    transform: rotate(-90deg);
    width: 80px;
    height: 80px;
}
.probability-gauge-wrapper circle {
    fill: none;
    stroke-width: 7;
}
.probability-gauge-wrapper circle.bg {
    stroke: #f1f5f9;
}
.probability-gauge-wrapper circle.meter {
    stroke-linecap: round;
    transition: stroke-dashoffset 0.8s ease;
}
.probability-gauge-wrapper .gauge-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 16px;
    font-weight: 700;
    color: #1e1b4b;
}

/* Custom comparison chart */
.chart-container { border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; background: #ffffff; margin-bottom: 20px; }
.chart-title { font-size: 12px; font-weight: 600; color: #334155; margin-bottom: 15px; }
.chart-bars { display: flex; height: 150px; }
.chart-y-axis { display: flex; flex-direction: column; justify-content: space-between; align-items: flex-end; width: 30px; font-size: 10px; color: #94a3b8; padding-right: 8px; }
.chart-area { flex: 1; position: relative; height: 100%; border-bottom: 1.5px solid #e2e8f0; border-left: 1.5px solid #e2e8f0; }
.chart-grid-lines { position: absolute; width: 100%; height: 100%; display: flex; flex-direction: column; justify-content: space-between; }
.grid-line { border-top: 1px dashed #f1f5f9; }
.chart-bar-group { position: absolute; width: 100%; height: 100%; display: flex; justify-content: space-around; align-items: flex-end; }
.chart-bar-container { display: flex; flex-direction: column; align-items: center; width: 35%; height: 100%; justify-content: flex-end; }
.chart-bar { width: 40px; border-radius: 4px 4px 0 0; position: relative; }
.bar-not-placed { background: linear-gradient(to top, #fca5a5, #ef4444); }
.bar-placed { background: linear-gradient(to top, #6ee7b7, #10b981); }
.bar-tooltip { position: absolute; top: -24px; left: 50%; transform: translateX(-50%); background: #1e293b; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 600; }
.chart-label { font-size: 11px; color: #64748b; margin-top: 6px; }
.chart-legend { display: flex; justify-content: center; gap: 20px; margin-top: 12px; font-size: 11px; }
.legend-item { display: flex; align-items: center; gap: 6px; color: #64748b; }
.legend-color { width: 10px; height: 10px; border-radius: 2px; }
.color-not-placed { background: #ef4444; }
.color-placed { background: #10b981; }

/* Warnings/Success recommendations */
.rec-warning { background-color: #fffbeb; border-left: 3px solid #f59e0b; padding: 10px 12px; border-radius: 0 6px 6px 0; margin-bottom: 8px; font-size: 12.5px; color: #78350f; font-weight: 500; display: flex; align-items: center; gap: 6px; }
.rec-success { background-color: #ecfdf5; border-left: 3px solid #10b981; padding: 10px 12px; border-radius: 0 6px 6px 0; margin-bottom: 8px; font-size: 12.5px; color: #065f46; font-weight: 500; display: flex; align-items: center; gap: 6px; }

.app-footer { text-align: center; font-size: 11.5px; color: #64748b; margin-top: 35px; padding-top: 15px; border-top: 1px solid #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PAGE HEADER SECTION
# ==========================================
st.markdown("""
<div class="header-container">
    <div class="header-text">
        <div class="header-title-row">
            <span class="header-icon">🎓</span>
            <div class="header-title">Student Placement Predictor</div>
        </div>
        <p class="header-subtitle">Fill the details below to predict the placement chances of a student</p>
    </div>
    <div class="header-illustration-cap">🎓</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# STUDENT DETAILS INPUT CARD (FULL WIDTH)
# ==========================================
with st.container(border=True):
    # Trigger div to target this container parent via CSS :has()
    st.markdown('<div id="student-details-card-trigger"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card-header-row">
        <span class="card-header-icon">👤</span>
        <h3 class="card-header-title">Student Details</h3>
    </div>
    <div class="card-subtitle">Provide accurate information</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        age = st.slider("🎂 Age", 18, 30, 21)
        gender_classes = list(gender_encoder.classes_)
        default_gender_idx = gender_classes.index("Male") if "Male" in gender_classes else 0
        gender = st.selectbox("⚧️ Gender", gender_classes, index=default_gender_idx)
        stream_classes = list(stream_encoder.classes_)
        default_stream_idx = stream_classes.index("Computer Science") if "Computer Science" in stream_classes else 0
        stream = st.selectbox("🎓 Stream", stream_classes, index=default_stream_idx)
        internships = st.number_input("💼 Internships", min_value=0, max_value=10, value=2)
        
    with col2:
        cgpa = st.slider("⭐ CGPA", 0.0, 10.0, 8.20, step=0.1)
        hostel = st.selectbox("🏠 Hostel", [0, 1], index=1, format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)")
        backlogs = st.number_input("📑 Backlogs", min_value=0, max_value=10, value=0)
        
    predict = st.button("🚀 Predict Placement")

# ==========================================
# PREDICTION RESULTS SECTION (AT THE BOTTOM / DOWNSIDE)
# ==========================================
if predict:
    # Transform categories using trained encoders
    gender_encoded = gender_encoder.transform([gender])[0]
    stream_encoded = stream_encoder.transform([stream])[0]
    
    # Map backlogs count to binary flag (0 or 1) expected by the ML model
    backlogs_encoded = 1 if backlogs > 0 else 0
    
    # Arrange dataframe strictly in the training columns order
    input_data = pd.DataFrame(
        [[
            age,
            gender_encoded,
            stream_encoded,
            internships,
            cgpa,
            hostel,
            backlogs_encoded
        ]],
        columns=[
            "Age",
            "Gender",
            "Stream",
            "Internships",
            "CGPA",
            "Hostel",
            "HistoryOfBacklogs"
        ]
    )
    
    # Model inference
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # Robust class lookup to map probabilities correctly
    classes_list = list(model.classes_)
    class_0_idx = classes_list.index(0)
    class_1_idx = classes_list.index(1)
    
    placed_prob = probability[class_1_idx] * 100
    not_prob = probability[class_0_idx] * 100
    
    # Hard Rule Check: Enforce backlog and internship policy overrides
    if backlogs > 0:
        prediction = 0
        placed_prob = 0.0
        not_prob = 100.0
    elif internships == 0:
        prediction = 0
        # If model predicted high, lower it to reflect the lack of internships
        placed_prob = min(placed_prob, 15.0) 
        not_prob = 100.0 - placed_prob
    else:
        # Prevent 100% exactly to make it look more realistic
        if placed_prob > 98.0:
            placed_prob = 98.0
            not_prob = 2.0
        elif not_prob > 98.0:
            not_prob = 98.0
            placed_prob = 2.0
        
    # Dynamic card color styling for the second card (Prediction Result) using the trigger CSS :has()
    card_bg = "#f0fdf4" if prediction == 1 else "#fef2f2"
    card_border = "#86efac" if prediction == 1 else "#fca5a5"
    card_accent = "#10b981" if prediction == 1 else "#ef4444"
    
    st.markdown(f"""
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"]:has(#prediction-result-card-trigger) {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-top: 4px solid {card_accent} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
        
    with st.container(border=True):
        # Trigger div to target this container parent via CSS :has()
        st.markdown('<div id="prediction-result-card-trigger"></div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card-header-row">
            <span class="card-header-icon">🚀</span>
            <h3 class="card-header-title">Prediction Result</h3>
        </div>
        <div class="card-subtitle">Here is the prediction for the given input</div>
        """, unsafe_allow_html=True)
        
        # Render placed/not-placed status badge
        if prediction == 1:
            st.markdown("""
            <div class="result-badge placed">
                <div class="badge-icon">✓</div>
                <div class="badge-text">
                    <div class="badge-title">PLACED</div>
                    <div class="badge-subtitle">The student is likely to get placed!</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-badge not-placed">
                <div class="badge-icon">✗</div>
                <div class="badge-text">
                    <div class="badge-title">NOT PLACED</div>
                    <div class="badge-subtitle">The student is unlikely to get placed.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Profile Status Highlight Badge
        if backlogs > 0:
            st.markdown('<div class="profile-rating-badge rating-critical">⚠️ Profile Status: Placement Blocked</div>', unsafe_allow_html=True)
        elif cgpa >= 8.0 and internships >= 2:
            st.markdown('<div class="profile-rating-badge rating-exceptional">🏆 Profile Status: Exceptional (Highly Placeable)</div>', unsafe_allow_html=True)
        elif cgpa >= 7.0:
            st.markdown('<div class="profile-rating-badge rating-competitive">📈 Profile Status: Competitive (Good Chances)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="profile-rating-badge rating-critical">⚠️ Profile Status: Critical Actions Required</div>', unsafe_allow_html=True)
            
        # SVG Circular Gauge Calculations
        circ = 180.9  # circumference for radius 28.8
        dashoffset = circ * (1 - (placed_prob / 100.0))
        gauge_color = "#10b981" if prediction == 1 else "#ef4444"
        
        # Render Placement Probability card (Premium side-by-side design)
        st.markdown(f"""
        <div class="probability-card">
            <div class="probability-info">
                <div class="probability-label">Placement Probability</div>
                <div class="probability-value">{placed_prob:.2f}%</div>
            </div>
            <div class="probability-gauge-wrapper">
                <svg viewBox="0 0 72 72">
                    <circle class="bg" cx="36" cy="36" r="28.8" />
                    <circle class="meter" cx="36" cy="36" r="28.8" 
                            stroke="{gauge_color}"
                            stroke-dasharray="{circ}" 
                            stroke-dashoffset="{dashoffset}" />
                </svg>
                <div class="gauge-text">{placed_prob:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom CSS probability comparison bar chart
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-title">Probability Comparison</div>
            <div class="chart-bars">
                <div class="chart-y-axis">
                    <span>100</span>
                    <span>80</span>
                    <span>60</span>
                    <span>40</span>
                    <span>20</span>
                    <span>0</span>
                </div>
                <div class="chart-area">
                    <div class="chart-grid-lines">
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                        <div class="grid-line"></div>
                    </div>
                    <div class="chart-bar-group">
                        <div class="chart-bar-container">
                            <div class="chart-bar bar-not-placed" style="height: {not_prob}%">
                                <span class="bar-tooltip">{not_prob:.2f}%</span>
                            </div>
                            <span class="chart-label">Not Placed</span>
                        </div>
                        <div class="chart-bar-container">
                            <div class="chart-bar bar-placed" style="height: {placed_prob}%">
                                <span class="bar-tooltip">{placed_prob:.2f}%</span>
                            </div>
                            <span class="chart-label">Placed</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="chart-legend">
                <span class="legend-item"><span class="legend-color color-not-placed"></span> Not Placed</span>
                <span class="legend-item"><span class="legend-color color-placed"></span> Placed</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Recommendations
        st.write("")
        has_warnings = False
        
        if cgpa < 7.5:
            st.markdown('<div class="rec-warning">⚠️ Improve CGPA above 7.5 to expand core engineering and software job opportunities.</div>', unsafe_allow_html=True)
            has_warnings = True
            
        if internships == 0:
            st.markdown('<div class="rec-warning">⚠️ No Internships: Less chances to place. If there are projects it is good, but complete at least one internship to gain hands-on experience.</div>', unsafe_allow_html=True)
            has_warnings = True
        elif internships < 2:
            st.markdown('<div class="rec-warning">⚠️ Try completing another internship (2+ total) to build wider industry experience and significantly elevate selection chances.</div>', unsafe_allow_html=True)
            has_warnings = True
            
        if backlogs > 0:
            st.markdown(f'<div class="rec-warning">⚠️ Placement blocked: Active backlogs ({backlogs}) make the student ineligible for selection in placement drives. Clear all active backlogs to restore eligibility.</div>', unsafe_allow_html=True)
            has_warnings = True
            
        if not has_warnings:
            st.markdown('<div class="rec-success">✓ Excellent profile! Outstanding CGPA, strong internship record, and clean backlog history.</div>', unsafe_allow_html=True)

# ==========================================
# APP FOOTER
# ==========================================
st.markdown("""
<div class="app-footer">
    Built with ❤️ using Python, Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)