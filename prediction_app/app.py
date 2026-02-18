# ============================================================
# APP.PY - WITH MULTIPLE BODY PARTS + AUTO GROUP ASSIGNMENT
# Save as: prediction_app/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib

# ============================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================
st.set_page_config(
    page_title="ThinkingCode - Medical Claim Predictor",
    page_icon="H",
    layout="wide"
)

# Custom CSS - ALL CONTRAST FIXED
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
    }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a5f !important;
    }
    
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] label {
        color: #1e3a5f !important;
    }
    
    /* Input Labels */
    .stSelectbox label, 
    .stNumberInput label, 
    .stTextInput label,
    .stMultiSelect label {
        color: #1e3a5f !important;
        font-weight: 500 !important;
    }
    
    /* Selectbox/Dropdown - Dark background, white text */
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        background-color: #0f172a !important;
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox [data-baseweb="select"] *,
    .stMultiSelect [data-baseweb="select"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    .stSelectbox [data-baseweb="select"] svg,
    .stMultiSelect [data-baseweb="select"] svg {
        fill: #FFFFFF !important;
    }
    
    /* Dropdown menu items - Sky blue text */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="listbox"] {
        background-color: #0f172a !important;
    }
    
    [data-baseweb="popover"] li,
    [data-baseweb="menu"] li,
    [data-baseweb="listbox"] li,
    [role="option"] {
        color: #7DD3FC !important;
        -webkit-text-fill-color: #7DD3FC !important;
        background-color: #0f172a !important;
    }
    
    [data-baseweb="popover"] li:hover,
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background-color: #1e3a5f !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    [role="option"][aria-selected="true"] {
        background-color: #2563eb !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    /* Multi-select tags */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #2563eb !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #0f172a !important;
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    .stNumberInput button {
        background-color: #2563eb !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #0f172a !important;
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    .stTextInput > div > div > input:disabled {
        background-color: #0f172a !important;
        border: 2px solid #1e3a5f !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 0.9 !important;
    }
    
    /* Expanders/Accordions */
    .streamlit-expanderHeader {
        background-color: #0f172a !important;
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderHeader p,
    .streamlit-expanderHeader span,
    .streamlit-expanderHeader div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    .streamlit-expanderHeader svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    .streamlit-expanderContent {
        color: #1e3a5f !important;
        background-color: #f8fafc !important;
        border: 2px solid #1e3a5f !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* Labels inside expander - Sky Blue */
    .streamlit-expanderContent label,
    .streamlit-expanderContent .stNumberInput label,
    .streamlit-expanderContent .stSelectbox label,
    .streamlit-expanderContent .stTextInput label {
        color: #7DD3FC !important;
        -webkit-text-fill-color: #7DD3FC !important;
        font-weight: 600 !important;
    }
    
    /* Markdown text inside expander (Has ICD: No) */
    .streamlit-expanderContent p,
    .streamlit-expanderContent .stMarkdown p,
    .streamlit-expanderContent span {
        color: #7DD3FC !important;
        -webkit-text-fill-color: #7DD3FC !important;
    }
    
    /* Input boxes - dark with white text */
    .streamlit-expanderContent .stNumberInput > div > div > input,
    .streamlit-expanderContent .stTextInput > div > div > input {
        background-color: #0f172a !important;
        border: 2px solid #3b82f6 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    
    /* Number input buttons */
    .streamlit-expanderContent .stNumberInput button {
        background-color: #3b82f6 !important;
        color: #FFFFFF !important;
    }
            /* ============================================
       EXPANDERS/ACCORDIONS - COLLAPSED STATE FIX
       ============================================ */
    /* Collapsed expander header */
    [data-testid="stExpander"] {
        background-color: #0f172a !important;
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
    }
    
    [data-testid="stExpander"] summary {
        background-color: #0f172a !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
    }
    
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stExpander"] summary svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    
    /* Expanded content area */
    [data-testid="stExpander"] > div:last-child {
        background-color: #f8fafc !important;
        border-top: 2px solid #1e3a5f !important;
    } 
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2563eb 100%) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }
    
    .stButton > button {
        color: #1e3a5f !important;
        background-color: #FFFFFF !important;
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    
    /* Alerts */
    .stSuccess { background-color: #d1fae5 !important; color: #065f46 !important; }
    .stError { background-color: #fee2e2 !important; color: #991b1b !important; }
    .stWarning { background-color: #fef3c7 !important; color: #92400e !important; }
    .stInfo { background-color: #dbeafe !important; color: #1e40af !important; }
    
    /* DataFrames */
    .stDataFrame {
        border: 2px solid #1e3a5f !important;
        border-radius: 8px !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
    }
    .footer p { color: #64748b !important; }
    .footer strong { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models")

# ============================================================
# BODY PART → BODY PART GROUP MAPPING
# ============================================================
BODY_PART_TO_GROUP = {
    # FINGERS
    "FINGER(S)": "FINGERS",
    "FINGER, LEFT: INDEX": "FINGERS",
    "FINGER, LEFT: LITTLE": "FINGERS",
    "FINGER, LEFT: MIDDLE": "FINGERS",
    "FINGER, LEFT: RING": "FINGERS",
    "FINGER, RIGHT: INDEX": "FINGERS",
    "FINGER, RIGHT: LITTLE": "FINGERS",
    "FINGER, RIGHT: MIDDLE": "FINGERS",
    "FINGER, RIGHT: RING": "FINGERS",
    "THUMB(S)": "FINGERS",
    "THUMB, LEFT": "FINGERS",
    "THUMB, RIGHT": "FINGERS",
    
    # HEAD
    "BRAIN": "HEAD",
    "EAR(S)": "HEAD",
    "EAR, LEFT": "HEAD",
    "EAR, RIGHT": "HEAD",
    "EYE(S)": "HEAD",
    "EYE, LEFT": "HEAD",
    "EYE, RIGHT": "HEAD",
    "FACE: INCLUDING EYELIDS": "HEAD",
    "HEAD": "HEAD",
    "HEAD: NOC": "HEAD",
    "MOUTH": "HEAD",
    "MULTIPLE HEAD INJURY": "HEAD",
    "NOSE": "HEAD",
    "SKULL": "HEAD",
    "TEETH": "HEAD",
    
    # LOWER EXTREMITIES
    "ACHILLES TENDON": "LOWER EXTREMITIES",
    "ANKLE(S)": "LOWER EXTREMITIES",
    "ANKLE, LEFT": "LOWER EXTREMITIES",
    "ANKLE, RIGHT": "LOWER EXTREMITIES",
    "CALF, BOTH": "LOWER EXTREMITIES",
    "CALF, LEFT": "LOWER EXTREMITIES",
    "CALF, RIGHT": "LOWER EXTREMITIES",
    "FOOT(FEET)": "LOWER EXTREMITIES",
    "FOOT, LEFT": "LOWER EXTREMITIES",
    "FOOT, RIGHT": "LOWER EXTREMITIES",
    "HIP(S)": "LOWER EXTREMITIES",
    "HIP, LEFT": "LOWER EXTREMITIES",
    "HIP, RIGHT": "LOWER EXTREMITIES",
    "KNEE(S)": "LOWER EXTREMITIES",
    "KNEE, LEFT": "LOWER EXTREMITIES",
    "KNEE, RIGHT": "LOWER EXTREMITIES",
    "LEG(S): LOWER": "LOWER EXTREMITIES",
    "LEG, LEFT: LOWER": "LOWER EXTREMITIES",
    "LEG, RIGHT: LOWER": "LOWER EXTREMITIES",
    "LEG(S): UPPER": "LOWER EXTREMITIES",
    "LEG, LEFT: UPPER": "LOWER EXTREMITIES",
    "LEG, RIGHT: UPPER": "LOWER EXTREMITIES",
    "MULTIPLE LOWER EXTREMITIES": "LOWER EXTREMITIES",
    
    # TOES
    "TOE(S)": "TOES",
    "TOE, LEFT: GREAT": "TOES",
    "TOE, LEFT: SECOND/OTHER": "TOES",
    "TOE, RIGHT: GREAT": "TOES",
    "TOE, RIGHT: SECOND/OTHER": "TOES",
    
    # NECK
    "CERVICAL DISC": "NECK",
    "CERVICAL SPINAL CORD": "NECK",
    "CERVICAL VERTEBRAE": "NECK",
    "NECK": "NECK",
    "TRACHEA": "NECK",
    
    # TRUNK
    "ABDOMEN INCLUDING GROIN": "TRUNK",
    "BACK AREA LOWER: LUMBAR/SACRAL": "TRUNK",
    "BACK AREA MIDDLE": "TRUNK",
    "BACK AREA UPPER: THORACIC AREA": "TRUNK",
    "BACK: NOC": "TRUNK",
    "BACK: SPINAL CORD": "TRUNK",
    "BUTTOCKS": "TRUNK",
    "CHEST: RIBS/STERNUM/TISSUE": "TRUNK",
    "CIRCULATORY SYSTEM": "TRUNK",
    "DIGESTIVE SYSTEM": "TRUNK",
    "HEART": "TRUNK",
    "INTERNAL ORGANS: NOC": "TRUNK",
    "LUMBAR DISC": "TRUNK",
    "LUMBAR SPINAL CORD": "TRUNK",
    "LUMBAR VERTEBRAE": "TRUNK",
    "LUNGS": "TRUNK",
    "MULTIPLE TRUNK": "TRUNK",
    "NERVOUS SYSTEM": "TRUNK",
    "PELVIS": "TRUNK",
    "RESPIRATORY SYSTEM": "TRUNK",
    "SACRUM & COCCYX": "TRUNK",
    "SOFT TISSUE": "TRUNK",
    "SPINAL CORD": "TRUNK",
    "THORACIC DISC": "TRUNK",
    "THORACIC SPINAL CORD": "TRUNK",
    "THORACIC VERTEBRAE": "TRUNK",
    "TRUNK: NOC": "TRUNK",
    "URINARY SYSTEM": "TRUNK",
    
    # UPPER EXTREMITIES
    "ELBOW(S)": "UPPER EXTREMITIES",
    "ELBOW, LEFT": "UPPER EXTREMITIES",
    "ELBOW, RIGHT": "UPPER EXTREMITIES",
    "HAND(S)": "UPPER EXTREMITIES",
    "HAND, LEFT": "UPPER EXTREMITIES",
    "HAND, RIGHT": "UPPER EXTREMITIES",
    "MULTIPLE UPPER EXTREMITIES": "UPPER EXTREMITIES",
    "SHOULDER(S)": "UPPER EXTREMITIES",
    "SHOULDER, LEFT": "UPPER EXTREMITIES",
    "SHOULDER, RIGHT": "UPPER EXTREMITIES",
    "WRIST(S)": "UPPER EXTREMITIES",
    "WRIST, LEFT": "UPPER EXTREMITIES",
    "WRIST, RIGHT": "UPPER EXTREMITIES",
    
    # MULTIPLE BODY PARTS
    "ARTIFICIAL APPLIANCE": "MULTIPLE BODY PARTS",
    "BODY SYSTEMS & MULT BODY SYS": "MULTIPLE BODY PARTS",
    "INSUFFICIENT INFO TO IDENTIFY": "MULTIPLE BODY PARTS",
    "MULTIPLE BODY PARTS": "MULTIPLE BODY PARTS",
    "NO PHYSICAL INJURY": "MULTIPLE BODY PARTS",
    "WHOLE BODY": "MULTIPLE BODY PARTS",
    "UNKNOWN": "MULTIPLE BODY PARTS",
    "OTHER": "MULTIPLE BODY PARTS",
    
    # MISSING
    "MISSING": "MISSING",
}

# ============================================================
# FEATURE OPTIONS
# ============================================================
GENDER = ["Male", "Female"]

BODY_PART_DESC = sorted(BODY_PART_TO_GROUP.keys())

BODY_PART_GROUP_DESC = [
    "TRUNK", "UPPER EXTREMITIES", "LOWER EXTREMITIES",
    "MULTIPLE BODY PARTS", "FINGERS", "HEAD", "NECK", "TOES", "MISSING"
]

CLAIM_CAUSE_GROUP_DESC = [
    "STRAIN OR INJURY", "FALL/SLIP/TRIP", "MISC CAUSES", "STRUCK",
    "RUBBED/ABRADED", "MOTOR VEHICLE", "CAUGHT", "STRIKE/STEP",
    "CUT/PUNCTURE", "HEAT/COLD", "MISSING"
]

NATURE_OF_INJURY_DESC = [
    "ABRASION/SCRATCH","AIDS/HIV","ALL OTHER","ALL OTHER OCCUPATIONAL DISEASE","ALLERGIC REACTION","AMPUTATION","ANGINA PECTORIS","ASBESTOSIS","ASPHYXIATION","BITES","BITES/STINGS","BLACK LUNG",
    "BODY LICE","BROKEN TOOTH","BURN","CANCER","CARDIOVASCULAR DISEASE","CARPAL TUNNEL SYNDROME","CHEMICAL EXPOSURE","CHRONIC CONDITION","CONCUSSION","CONJUNCTIVITIS",
    "CONTAGIOUS DISEASE","CONTUSION","CRUSHING","CUMULATIVE INJURIES","CUMULATIVE INJURY HEARING LOSS","DEATH","DEATH/ILLNESS","DERMATITIS","DISLOCATION","DIZZINESS",
    "DRUG REACTION","DUST DISEASE NOC (ALL OTHER)","ELECTRIC SHOCK","ENUCLEATION","EXPOSURE BLOODBORNE PATHOGEN","EXPOSURE MOLD","FOREIGN BODY","FRACTURE","FREEZING","GANGLION TUMOR",
    "GUNSHOT WOUND","HEADACHE","HEARING LOSS (TRAUMATIC ONLY)","HEAT PROSTRATION","HEPATITIS","HEPATITIS B","HEPATITIS C","HEPATITIS EXPOSURE","HERNIA","HYPERTENSION",
    "ILLNESS","INFECTION","INFLAMMATION","INHALATION","LACERATION","MENINGITIS","MENTAL DISORDER","MISSING","MULT INJURIES PHYSICAL & PSYC","MULT PHYSICAL INJURIES ONLY",
    "MYOCARDIAL INFARCTION","NO PHYSICAL INJURY","OTHER THAN CARPAL TUNNEL SYND","PARALYSIS","PNEUMONIA","POISONING-GENERAL NOT OD/CULMU","POISONING: CHEMICAL",
    "POISONING: METAL","PSYCHIATRIC/MENTAL STRESS","PUNCTURE","RADIATION","RESPIRATORY DISORDERS","RUPTURE","SEIZURE","SEVERANCE","SHOCK","SMOKE INHALATION","SPRAIN","STINGS",
    "STRAIN","STRESS","SYNCOPE","TORN LIGAMENT","TUBERCULOSIS","TUBERCULOSIS EXPOSURE","ULCER","UNCONSCIOUS","UNKNOWN","VASCULAR","VISION LOSS"
]

INCIDENT_STATE = [
    "CA", "CO", "AZ", "TX", "WA", "OR", "NV", "UT",
    "IL", "MO", "NY", "FL", "GA", "OH", "PA", "Other"
]

CLAIMANT_TYPE_DESC = [
    "Medical Only", "TD", "Indemnity", "Future Medical III",
    "Minor PD", "Indemnity-No Comp Lost Time", "Major PD",
    "Record Only", "Future Medical II", "Future Medical I",
    "Future medical", "Death", "Total PD"
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_age_group(age):
    if 18 <= age <= 25:
        return "Young_Adult"
    elif 26 <= age < 40:
        return "Adult"
    elif 40 <= age < 55:
        return "Middle_Aged"
    elif age >= 55:
        return "Senior_Citizen"
    else:
        return "Unknown"

def get_body_part_group(body_part):
    """Get body part group from body part"""
    return BODY_PART_TO_GROUP.get(body_part, "MULTIPLE BODY PARTS")

def get_primary_body_part_group(body_parts_list):
    """Get primary body part group from list of body parts"""
    if not body_parts_list:
        return "TRUNK"
    
    groups = [BODY_PART_TO_GROUP.get(bp, "MULTIPLE BODY PARTS") for bp in body_parts_list]
    
    # If multiple different groups, return MULTIPLE BODY PARTS
    unique_groups = set(groups)
    if len(unique_groups) > 1:
        return "MULTIPLE BODY PARTS"
    
    return groups[0]

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def encode_categorical(value, label_encoder):
    try:
        if value in label_encoder.classes_:
            return label_encoder.transform([value])[0]
        else:
            return 0
    except:
        return 0

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models():
    models = {}
    
    from tensorflow.keras.models import load_model
    models['encoder'] = load_model(MODEL_PATH / 'lstm_encoder.keras')
    
    with open(MODEL_PATH / 'lstm_artifacts.pkl', 'rb') as f:
        models['lstm_artifacts'] = pickle.load(f)
    
    models['router'] = joblib.load(MODEL_PATH / 'router_xgb_best.joblib')
    with open(MODEL_PATH / 'router_artifacts.pkl', 'rb') as f:
        models['router_artifacts'] = pickle.load(f)
    
    models['low_model'] = joblib.load(MODEL_PATH / 'low_model.joblib')
    with open(MODEL_PATH / 'low_artifacts.pkl', 'rb') as f:
        models['low_artifacts'] = pickle.load(f)
    
    models['med_model'] = joblib.load(MODEL_PATH / 'med_model.joblib')
    with open(MODEL_PATH / 'med_artifacts.pkl', 'rb') as f:
        models['med_artifacts'] = pickle.load(f)
    
    models['high_model'] = joblib.load(MODEL_PATH / 'high_model.joblib')
    with open(MODEL_PATH / 'high_artifacts.pkl', 'rb') as f:
        models['high_artifacts'] = pickle.load(f)
    
    sample_claims_path = MODEL_PATH / 'sample_claims.pkl'
    if sample_claims_path.exists():
        with open(sample_claims_path, 'rb') as f:
            models['sample_claims'] = pickle.load(f)
    else:
        models['sample_claims'] = {}
    
    return models

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_claim_cost(models, visits_data, claim_id):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    lstm_artifacts = models['lstm_artifacts']
    router_artifacts = models['router_artifacts']
    
    sequence_features = lstm_artifacts['sequence_features']
    label_encoders = lstm_artifacts['label_encoders']
    scaler_seq = router_artifacts['scaler_seq']
    
    sequence = []
    total_paid = 0
    
    for visit in visits_data:
        visit_features = []
        
        for feat in sequence_features:
            if feat.endswith('_ENC'):
                original_col = feat.replace('_ENC', '')
                if original_col in label_encoders:
                    raw_value = visit.get(original_col, 'UNKNOWN')
                    if pd.isna(raw_value) or raw_value is None:
                        raw_value = 'UNKNOWN'
                    encoded_value = encode_categorical(str(raw_value), label_encoders[original_col])
                    visit_features.append(float(encoded_value))
                else:
                    visit_features.append(0.0)
            elif feat == 'medical_amount':
                val = safe_float(visit.get('medical_amount', 0))
                visit_features.append(val)
                total_paid += val
            elif feat in ['ICD_COUNT', 'unique_icd_codes_count']:
                val = safe_int(visit.get('unique_icd_codes_count', visit.get('ICD_COUNT', 0)))
                visit_features.append(float(val))
            elif feat == 'HAS_ICD':
                visit_features.append(float(safe_int(visit.get('HAS_ICD', 0))))
            elif feat == 'NO_OF_VISIT':
                visit_features.append(float(safe_int(visit.get('NO_OF_VISIT', 1))))
            elif feat == 'AGE_FINAL':
                visit_features.append(safe_float(visit.get('AGE_FINAL', 45)))
            elif feat == 'NUM_BODY_PARTS':
                visit_features.append(float(safe_int(visit.get('NUM_BODY_PARTS', 1))))
            elif feat in visit:
                visit_features.append(safe_float(visit[feat]))
            else:
                visit_features.append(0.0)
        
        sequence.append(visit_features)
    
    sequence = np.array(sequence, dtype=np.float32)
    
    try:
        sequence_scaled = scaler_seq.transform(sequence)
    except Exception as e:
        st.error(f"Scaler error: {e}")
        return None
    
    sequence_padded = pad_sequences(
        [sequence_scaled], maxlen=50, dtype='float32',
        padding='post', truncating='post', value=0.0
    )
    
    encoder = models['encoder']
    lstm_input_dim = encoder.input_shape[-1]
    our_dim = sequence_padded.shape[-1]
    
    if lstm_input_dim != our_dim:
        embedding = np.mean(sequence_padded, axis=1)
    else:
        embedding = encoder.predict(sequence_padded, verbose=0)
    
    router_scaler = router_artifacts['scaler']
    embedding_router_scaled = router_scaler.transform(embedding)
    
    router = models['router']
    complexity_pred = router.predict(embedding_router_scaled)[0]
    complexity_label = router_artifacts['label_encoder'].inverse_transform([complexity_pred])[0]
    
    complexity_proba = router.predict_proba(embedding_router_scaled)[0]
    proba_dict = {
        label: float(prob) 
        for label, prob in zip(router_artifacts['label_encoder'].classes_, complexity_proba)
    }
    
    if complexity_label == 'LOW':
        model = models['low_model']
        scaler_emb = models['low_artifacts']['scaler_emb']
        is_hybrid = models['low_artifacts'].get('is_hybrid', False)
    elif complexity_label == 'MED':
        model = models['med_model']
        scaler_emb = models['med_artifacts']['scaler_emb']
        is_hybrid = models['med_artifacts'].get('is_hybrid', False)
    else:
        model = models['high_model']
        scaler_emb = models['high_artifacts']['scaler_emb']
        is_hybrid = models['high_artifacts'].get('is_hybrid', False)
    
    embedding_model_scaled = scaler_emb.transform(embedding)
    
    if is_hybrid and isinstance(model, dict):
        pred_log = np.zeros(len(embedding_model_scaled))
        for name, m in model.items():
            pred_log += m.predict(embedding_model_scaled)
        pred_log /= len(model)
    else:
        pred_log = model.predict(embedding_model_scaled)
    
    predicted_cost = float(np.expm1(pred_log[0]))
    predicted_cost = max(0, predicted_cost)
    
    return {
        'claim_id': claim_id,
        'predicted_total': round(predicted_cost, 2),
        'predicted_remaining': round(max(0, predicted_cost - total_paid), 2),
        'current_spent': round(total_paid, 2),
        'complexity': complexity_label,
        'complexity_probabilities': proba_dict,
        'high_cost_flag': predicted_cost >= 10000,
        'num_visits': len(visits_data)
    }

# ============================================================
# RENDER PROBABILITY BARS
# ============================================================
def render_probability_bars(proba_dict):
    fill_colors = {
        'LOW': '#6ee7b7',
        'MED': '#fde68a',
        'HIGH': '#fca5a5'
    }
    
    for comp, prob in proba_dict.items():
        fill_color = fill_colors.get(comp, '#60a5fa')
        width_pct = prob * 100
        
        st.markdown(f"""
        <div style="background: #0f172a; padding: 10px 15px; border-radius: 8px; margin: 5px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                             font-size: 13px; font-weight: 600;">{comp}</span>
                <span style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                             font-size: 14px; font-weight: 700;">{prob*100:.1f}%</span>
            </div>
            <div style="height: 8px; border-radius: 4px; background: rgba(255,255,255,0.2); overflow: hidden;">
                <div style="height: 100%; width: {width_pct}%; background: {fill_color}; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3a5f 0%, #2563eb 100%); 
                padding: 35px 50px; border-radius: 16px; 
                margin-bottom: 25px; text-align: center; 
                box-shadow: 0 6px 20px rgba(30, 58, 95, 0.4);">
        <div style="display: inline-block;">
            <span style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                         font-size: 2.2rem; font-weight: 400; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                         letter-spacing: 0.5px; text-transform: lowercase;">thinking</span><span 
                  style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                         font-size: 3rem; font-weight: 700; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                         letter-spacing: -0.5px; text-transform: lowercase;">code</span>
        </div>
        <div style="margin-top: 12px;">
            <span style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                         font-size: 1.6rem; font-weight: 400; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                         letter-spacing: 2px; text-transform: uppercase;">Medical Claim Predictor</span>
        </div>
        <div style="margin-top: 8px;">
            <span style="color: rgba(255, 255, 255, 0.8); -webkit-text-fill-color: rgba(255, 255, 255, 0.8); 
                         font-size: 1rem; font-weight: 300; font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                         letter-spacing: 1px;">Tristar Claim-Reserving Guide</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
  # ============================================================
    # STREAMLIT-NATIVE LOADING EXPERIENCE
    # ============================================================
    
    # Check if models are already loaded (cached)
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    if not st.session_state.models_loaded:
        # Create a status container for loading
        with st.status("🚀 Initializing ThinkingCode Prediction Engine...", expanded=True) as status:
            
            st.write("Loading LSTM Encoder...")
            progress_bar = st.progress(0)
            
            try:
                from tensorflow.keras.models import load_model as keras_load
                
                # Stage 1: LSTM Encoder (25%)
                progress_bar.progress(10)
                st.write("📊 Loading sequence encoder...")
                encoder = keras_load(MODEL_PATH / 'lstm_encoder.keras')
                progress_bar.progress(25)
                
                # Stage 2: LSTM Artifacts (35%)
                st.write("📦 Loading encoder artifacts...")
                with open(MODEL_PATH / 'lstm_artifacts.pkl', 'rb') as f:
                    lstm_artifacts = pickle.load(f)
                progress_bar.progress(35)
                
                # Stage 3: Router (50%)
                st.write("🔀 Loading complexity router...")
                router = joblib.load(MODEL_PATH / 'router_xgb_best.joblib')
                with open(MODEL_PATH / 'router_artifacts.pkl', 'rb') as f:
                    router_artifacts = pickle.load(f)
                progress_bar.progress(50)
                
                # Stage 4: LOW Model (65%)
                st.write("📈 Loading LOW complexity model...")
                low_model = joblib.load(MODEL_PATH / 'low_model.joblib')
                with open(MODEL_PATH / 'low_artifacts.pkl', 'rb') as f:
                    low_artifacts = pickle.load(f)
                progress_bar.progress(65)
                
                # Stage 5: MED Model (80%)
                st.write("📈 Loading MED complexity model...")
                med_model = joblib.load(MODEL_PATH / 'med_model.joblib')
                with open(MODEL_PATH / 'med_artifacts.pkl', 'rb') as f:
                    med_artifacts = pickle.load(f)
                progress_bar.progress(80)
                
                # Stage 6: HIGH Model (95%)
                st.write("📈 Loading HIGH complexity model...")
                high_model = joblib.load(MODEL_PATH / 'high_model.joblib')
                with open(MODEL_PATH / 'high_artifacts.pkl', 'rb') as f:
                    high_artifacts = pickle.load(f)
                progress_bar.progress(95)
                
                # Stage 7: Sample Claims (100%)
                st.write("📋 Loading sample claims...")
                sample_claims_path = MODEL_PATH / 'sample_claims.pkl'
                if sample_claims_path.exists():
                    with open(sample_claims_path, 'rb') as f:
                        sample_claims = pickle.load(f)
                else:
                    sample_claims = {}
                progress_bar.progress(100)
                
                # Assemble models dict
                models = {
                    'encoder': encoder,
                    'lstm_artifacts': lstm_artifacts,
                    'router': router,
                    'router_artifacts': router_artifacts,
                    'low_model': low_model,
                    'low_artifacts': low_artifacts,
                    'med_model': med_model,
                    'med_artifacts': med_artifacts,
                    'high_model': high_model,
                    'high_artifacts': high_artifacts,
                    'sample_claims': sample_claims
                }
                
                st.session_state.models = models
                st.session_state.models_loaded = True
                
                status.update(label="✅ Prediction Engine Ready!", state="complete", expanded=False)
                
            except Exception as e:
                status.update(label="❌ Error Loading Models", state="error", expanded=True)
                st.error(f"Failed to load models: {e}")
                st.exception(e)
                return
        
        # Brief success message
        st.success("✅ All models loaded successfully! Ready to predict.")
        models = st.session_state.models
        
    else:
        models = st.session_state.models

    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        st.markdown("---")
        
        input_method = st.radio(
            "Input Method",
            ["Sample Claims", "Manual Entry"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("## Model Performance")
        
        st.metric("LOW R²", f"{models['low_artifacts']['best_r2']:.3f}")
        st.metric("MED R²", f"{models['med_artifacts']['best_r2']:.3f}")
        st.metric("HIGH R²", f"{models['high_artifacts']['best_r2']:.3f}")
        
        st.markdown("---")
        st.markdown("## ThinkingCode")
        st.markdown("*Intelligent Healthcare Analytics*")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    actual_total = None
    actual_complexity = None
    visits_data = []
    claim_id = ""
    
    with col1:
        st.markdown("### Claim Information")
        
        if input_method == "Sample Claims":
            sample_claims = models.get('sample_claims', {})
            
            if not sample_claims:
                st.error("No sample claims found! Run: python extract_sample_claims.py")
                return
            
            all_claims_list = list(sample_claims.keys())
            selected_claim = st.selectbox(
                "Select a Sample Claim",
                all_claims_list,
                index=0
            )
            
            sample_data = sample_claims[selected_claim]
            claim_id = sample_data['claim_id']
            visits_data = sample_data['visits']
            actual_total = sample_data['actual_total']
            actual_complexity = sample_data['complexity']
            
            # Display claim info
            st.markdown("---")
            
            st.markdown(f"""
            <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                <div style="flex: 1; background: #0f172a; padding: 15px; border-radius: 12px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2); text-align: center; 
                            min-height: 85px; display: flex; flex-direction: column; justify-content: center;">
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF;
                              margin: 0; font-size: 11px; text-transform: uppercase; 
                              font-weight: 600; letter-spacing: 0.5px;">CLAIM ID</p>
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                              margin: 8px 0 0 0; font-size: 14px; font-weight: 700;">{claim_id}</p>
                </div>
                <div style="flex: 1; background: #0f172a; padding: 15px; border-radius: 12px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2); text-align: center; 
                            min-height: 85px; display: flex; flex-direction: column; justify-content: center;
                            border-bottom: 4px solid {'#10b981' if actual_complexity == 'LOW' else '#f59e0b' if actual_complexity == 'MED' else '#ef4444'};">
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF;
                              margin: 0; font-size: 11px; text-transform: uppercase; 
                              font-weight: 600; letter-spacing: 0.5px;">ACTUAL COMPLEXITY</p>
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                              margin: 8px 0 0 0; font-size: 20px; font-weight: 700;">{actual_complexity}</p>
                </div>
                <div style="flex: 1; background: #0f172a; padding: 15px; border-radius: 12px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2); text-align: center; 
                            min-height: 85px; display: flex; flex-direction: column; justify-content: center;">
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF;
                              margin: 0; font-size: 11px; text-transform: uppercase; 
                              font-weight: 600; letter-spacing: 0.5px;">ACTUAL TOTAL</p>
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                              margin: 8px 0 0 0; font-size: 20px; font-weight: 700;">${actual_total:,.2f}</p>
                </div>
                <div style="flex: 1; background: #0f172a; padding: 15px; border-radius: 12px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2); text-align: center; 
                            min-height: 85px; display: flex; flex-direction: column; justify-content: center;">
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF;
                              margin: 0; font-size: 11px; text-transform: uppercase; 
                              font-weight: 600; letter-spacing: 0.5px;">VISITS</p>
                    <p style="color: #FFFFFF; -webkit-text-fill-color: #FFFFFF; 
                              margin: 8px 0 0 0; font-size: 20px; font-weight: 700;">{len(visits_data)}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display visits
            st.markdown("#### Visit Details")
            df_display = pd.DataFrame(visits_data)
            
            display_cols = [
                'NO_OF_VISIT', 'medical_amount', 'GENDER', 'AGE_FINAL',
                'BODY_PART_DESC', 'BODY_PART_GROUP_DESC', 'NATURE_OF_INJURY_DESC',
                'CLAIM_CAUSE_GROUP_DESC', 'HAS_ICD', 'unique_icd_codes_count'
            ]
            display_cols = [c for c in display_cols if c in df_display.columns]
            
            if display_cols:
                st.dataframe(df_display[display_cols], hide_index=True)
            else:
                st.dataframe(df_display, hide_index=True)
            
            total_spent = sum(safe_float(v.get('medical_amount', 0)) for v in visits_data)
            st.markdown(f"**Total Spent:** ${total_spent:,.2f}")
            
        else:
            # ============================================================
            # MANUAL ENTRY - WITH MULTIPLE BODY PARTS
            # ============================================================
            st.markdown("#### Claim Details")
            
            claim_col1, claim_col2 = st.columns(2)
            with claim_col1:
                claim_id = st.text_input("Claim ID", value="CLM-MANUAL-001")
            with claim_col2:
                num_visits = st.number_input("Number of Visits", 1, 50, 3)
            
            st.markdown("#### Claimant Information")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", GENDER)
            with c2:
                age = st.number_input("Age", 18, 100, 45)
            with c3:
                age_group = get_age_group(age)
                st.text_input("Age Group (Auto)", value=age_group, disabled=True)
            
            claimant_type = st.selectbox("Claimant Type", CLAIMANT_TYPE_DESC)
            
            st.markdown("#### Injury Information")
            
            # Body part selection - UNLIMITED
            body_parts_selected = st.multiselect(
                "Select All Affected Body Parts",
                BODY_PART_DESC,
                default=[BODY_PART_DESC[0]]
            )
            
            # Auto-calculate number of body parts
            num_body_parts = len(body_parts_selected) if body_parts_selected else 1
            
            # Auto-assign body part group
            body_part_group = get_primary_body_part_group(body_parts_selected)
            
            # Display auto-calculated fields
            bp_col1, bp_col2 = st.columns(2)
            with bp_col1:
                st.text_input(
                    "Body Parts Affected (Auto)", 
                    value=str(num_body_parts), 
                    disabled=True
                )
            with bp_col2:
                st.text_input(
                    "Primary Body Part Group (Auto)", 
                    value=body_part_group, 
                    disabled=True
                )
            
            # Show all affected groups if multiple
            if num_body_parts > 1:
                groups_list = list(set([get_body_part_group(bp) for bp in body_parts_selected]))
                if len(groups_list) > 1:
                    st.text_input(
                        "All Affected Groups", 
                        value=", ".join(groups_list), 
                        disabled=True
                    )
            
            # Other injury info
            i1, i2 = st.columns(2)
            with i1:
                nature_of_injury = st.selectbox("Nature of Injury", NATURE_OF_INJURY_DESC)
            with i2:
                claim_cause = st.selectbox("Claim Cause", CLAIM_CAUSE_GROUP_DESC)
            
            incident_state = st.selectbox("Incident State", INCIDENT_STATE)
            
            st.markdown("---")
            st.markdown("#### Visit Details")
            
            visits_data = []
            
            # Primary body part for encoding (use first selected)
            primary_body_part = body_parts_selected[0] if body_parts_selected else BODY_PART_DESC[0]
            
            for i in range(num_visits):
                with st.expander(f"Visit {i+1}", expanded=(i < 3)):
                    v1, v2 = st.columns(2)
                    
                    with v1:
                        medical_amount = st.number_input(
                            "Medical Amount ($)", 0.01, 50000.0, 50.0 + (i * 25.0),
                            key=f"amt_{i}"
                        )
                    
                    with v2:
                        icd_count = st.number_input(
                            "Unique ICD Code Count", 0, 5, 0,
                            key=f"icd_count_{i}"
                        )
                    
                    has_icd = 1 if icd_count > 0 else 0
                    st.markdown(f"**Has ICD:** {'Yes' if has_icd else 'No'} *(auto-set)*")
                    
                    visits_data.append({
                        'CLAIM_ID': claim_id,
                        'NO_OF_VISIT': i + 1,
                        'medical_amount': float(medical_amount),
                        'GENDER': gender,
                        'AGE_FINAL': float(age),
                        'AGE_GROUP_FINAL': age_group,
                        'CLAIMANT_TYPE_DESC': claimant_type,
                        'BODY_PART_DESC': primary_body_part,
                        'BODY_PART_GROUP_DESC': body_part_group,
                        'NUM_BODY_PARTS': num_body_parts,
                        'BODY_PARTS_LIST': body_parts_selected,
                        'NATURE_OF_INJURY_DESC': nature_of_injury,
                        'CLAIM_CAUSE_GROUP_DESC': claim_cause,
                        'INCIDENT_STATE': incident_state,
                        'HAS_ICD': has_icd,
                        'unique_icd_codes_count': int(icd_count),
                        'ICD_COUNT': int(icd_count),
                    })
            
            st.markdown("#### Visit Summary")
            df_summary = pd.DataFrame([{
                'Visit': v['NO_OF_VISIT'],
                'Amount': f"${v['medical_amount']:,.2f}",
                'Body Parts': v['NUM_BODY_PARTS'],
                'ICD Count': v['unique_icd_codes_count'],
                'Has ICD': 'Yes' if v['HAS_ICD'] else 'No'
            } for v in visits_data])
            st.dataframe(df_summary, hide_index=True)
            
            total_spent = sum(v['medical_amount'] for v in visits_data)
            st.markdown(f"**Total Entered:** ${total_spent:,.2f}")
            
            # Show body parts info
            if num_body_parts > 1:
                st.info(f"**{num_body_parts} Body Parts Affected:** {', '.join(body_parts_selected)}")
    
    with col2:
        st.markdown("### Prediction Results")
        
        if st.button("PREDICT COST", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                try:
                    result = predict_claim_cost(models, visits_data, claim_id)
                    
                    if result is None:
                        st.error("Prediction failed!")
                        return
                    
                    # Predicted Total
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
                                padding: 25px; border-radius: 12px; text-align: center;
                                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
                                border: 3px solid #1e3a5f; margin-bottom: 15px;">
                        <p style="color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
                                  margin: 0; font-size: 14px; font-weight: 700;
                                  text-transform: uppercase; letter-spacing: 1px;">PREDICTED TOTAL</p>
                        <h2 style="color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; 
                                   margin: 12px 0 0 0; font-size: 42px; font-weight: 800; 
                                   text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                            ${result['predicted_total']:,.2f}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Actual Total + Error
                    if actual_total is not None:
                        error = result['predicted_total'] - actual_total
                        error_pct = (error / actual_total) * 100 if actual_total > 0 else 0
                        
                        border_color = "#10b981" if abs(error_pct) < 20 else "#f59e0b" if abs(error_pct) < 50 else "#ef4444"
                        bg_color = "#f0fdf4" if abs(error_pct) < 20 else "#fffbeb" if abs(error_pct) < 50 else "#fef2f2"
                        
                        st.markdown(f"""
                        <div style="background: {bg_color}; padding: 18px; border-radius: 12px;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
                                    border: 2px solid {border_color}; margin-bottom: 15px;">
                            <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                      margin: 0; font-size: 11px; text-transform: uppercase;
                                      font-weight: 600; letter-spacing: 0.5px;">Actual Total</p>
                            <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                      margin: 8px 0 0 0; font-size: 22px; font-weight: 700;">
                                ${actual_total:,.2f}
                            </p>
                            <p style="color: #64748b !important; -webkit-text-fill-color: #64748b !important;
                                      margin: 5px 0 0 0; font-size: 12px; font-weight: 500;">
                                Error: ${error:,.2f} ({error_pct:+.1f}%)
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Complexity box
                    comp_bg = "#10b981" if result['complexity'] == "LOW" else "#f59e0b" if result['complexity'] == "MED" else "#ef4444"
                    
                    st.markdown(f"""
                    <div style="background: {comp_bg}; padding: 20px; border-radius: 12px;
                                text-align: center; border: 3px solid #1e3a5f; margin-bottom: 15px;">
                        <h3 style="color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; 
                                   margin: 0; font-weight: 700; font-size: 20px; letter-spacing: 1px;">
                            {result['complexity']} COMPLEXITY
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Routing check
                    if actual_complexity is not None:
                        if result['complexity'] == actual_complexity:
                            st.success("Routing CORRECT")
                        else:
                            st.error(f"Routing WRONG (Actual: {actual_complexity})")
                    
                    # Stats boxes
                    st.markdown(f"""
                    <div style="display: flex; gap: 15px; margin: 15px 0;">
                        <div style="flex: 1; background: #eff6ff; padding: 18px; border-radius: 12px;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
                                    border: 2px solid #2563eb;">
                            <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                      margin: 0; font-size: 11px; text-transform: uppercase;
                                      font-weight: 600; letter-spacing: 0.5px;">Current Spent</p>
                            <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                      margin: 8px 0 0 0; font-size: 20px; font-weight: 700;">
                                ${result['current_spent']:,.2f}
                            </p>
                        </div>
                        <div style="flex: 1; background: #eff6ff; padding: 18px; border-radius: 12px;
                                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
                                    border: 2px solid #2563eb;">
                            <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                      margin: 0; font-size: 11px; text-transform: uppercase;
                                      font-weight: 600; letter-spacing: 0.5px;">Predicted Remaining</p>
                            <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                      margin: 8px 0 0 0; font-size: 20px; font-weight: 700;">
                                ${result['predicted_remaining']:,.2f}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visits box
                    st.markdown(f"""
                    <div style="background: #eff6ff; padding: 18px; border-radius: 12px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
                                border: 2px solid #2563eb; margin-bottom: 15px;">
                        <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                  margin: 0; font-size: 11px; text-transform: uppercase;
                                  font-weight: 600; letter-spacing: 0.5px;">Visits Analyzed</p>
                        <p style="color: #1e3a5f !important; -webkit-text-fill-color: #1e3a5f !important;
                                  margin: 8px 0 0 0; font-size: 20px; font-weight: 700;">
                            {result['num_visits']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # High cost flag
                    if result['high_cost_flag']:
                        st.error("⚠️ HIGH COST FLAG (>$10K)")
                    
                    # Probabilities
                    st.markdown("#### Routing Probabilities")
                    render_probability_bars(result['complexity_probabilities'])
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p><strong>ThinkingCode</strong> - Medical Claim Cost Prediction System</p>
        <p style="font-size: 12px;">Tristar Claim-Reserving Guide | LSTM + XGBoost Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
