import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PAGE STYLE
# =========================
st.set_page_config(page_title="AI Market Signal", layout="wide")

st.markdown("""
<style>
.main {background-color: #0e1117;}
.block-container {padding-top: 2rem;}
.metric-box {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2937;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Market Direction Predictor")
st_autorefresh(interval=60000, limit=None)

# =========================
# LOAD DATA (no upload)
# =========================
DATA_PATH = "market_data.pkl"

try:
    df = pd.read_pickle(DATA_PATH)
except Exception as e:
    st.error("Data file nahi mila. Jupyter me df.to_pickle('market_data.pkl') run karo.")
    st.stop()

features = ['open', 'high', 'low', 'volume', 'vwap', 'ma_5', 'ma_10', 'volatility']

if not all(col in df.columns for col in features + ['close']):
    st.error("Required columns missing hain.")
    st.stop()

# =========================
# TARGET CREATE
# =========================
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
df = df.dropna()

X = df[features]
y = df['target']

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.subheader("Model Accuracy")
    st.metric("Accuracy", f"{acc*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# LIVE SIGNAL (latest row)
# =========================
X_live = X.tail(1)
proba = model.predict_proba(X_live)[0]

up_prob = proba[1]
down_prob = proba[0]

if up_prob > 0.6:
    signal = "UP"
elif down_prob > 0.6:
    signal = "DOWN"
else:
    signal = "NO TRADE"

with col2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.subheader("UP Probability")
    st.metric("Confidence", f"{up_prob*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.subheader("DOWN Probability")
    st.metric("Confidence", f"{down_prob*100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)

st.subheader("🚦 Live Market Signal")
if signal == "UP":
    st.success("UP TREND")
elif signal == "DOWN":
    st.error("DOWN TREND")
else:
    st.warning("NO TRADE ZONE")

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("Feature Importance")
importance = pd.Series(model.feature_importances_, index=features)
st.bar_chart(importance)