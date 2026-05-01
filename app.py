import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle, time, os, pathlib, json, hashlib

MODEL_LOADED = False

if os.path.exists("model.pkl") and os.path.exists("columns.pkl"):
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    MODEL_LOADED = True

st.set_page_config(
    page_title="E-Ticket Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_cfg = pathlib.Path(".streamlit/config.toml")
_cfg.parent.mkdir(exist_ok=True)
if not _cfg.exists():
    _cfg.write_text('[theme]\nbase="light"\nbackgroundColor="#f1f5f9"\nsecondaryBackgroundColor="#ffffff"\ntextColor="#1e293b"\nprimaryColor="#2563eb"\n')

@st.cache_resource
def load_model():
    try:
        m = pickle.load(open("model.pkl","rb"))
        c = pickle.load(open("columns.pkl","rb"))
        return m, c, True
    except:
        return None, None, False

model, MODEL_COLUMNS, MODEL_LOADED = load_model()

USERS_FILE = "users.json"
def _load_users():
    if os.path.exists(USERS_FILE):
        return json.load(open(USERS_FILE))
    defaults = {"admin":hashlib.sha256(b"admin123").hexdigest(),"analyst":hashlib.sha256(b"analyst123").hexdigest(),"demo":hashlib.sha256(b"demo123").hexdigest()}
    json.dump(defaults,open(USERS_FILE,"w"),indent=2)
    return defaults
def _save_users(u): json.dump(u,open(USERS_FILE,"w"),indent=2)
def _hp(p): return hashlib.sha256(p.encode()).hexdigest()
def verify_user(u,p):
    users=_load_users(); return u in users and users[u]==_hp(p)
def register_user(u,p):
    users=_load_users()
    if u in users: return False,"Username already exists."
    if len(u)<3:   return False,"Username must be 3+ characters."
    if len(p)<6:   return False,"Password must be 6+ characters."
    users[u]=_hp(p); _save_users(users); return True,"Account created! You can now log in."
def delete_user(u):
    users=_load_users(); users.pop(u,None); _save_users(users)

for k,v in {"logged_in":False,"current_user":None,"history":[],"notifications":0,"accent":"#2563eb"}.items():
    if k not in st.session_state: st.session_state[k]=v

def ml_predict_single(amount, tickets, device, location, hour):
    
    # If model not loaded → fallback
    if not MODEL_LOADED:
        return _fallback(amount, tickets, device, location, hour)

    try:
        # Create empty row with all columns
        row = {col: 0 for col in MODEL_COLUMNS}

        # Fill numerical values
        if "amount" in row:
            row["amount"] = amount
        if "tickets_booked" in row:
            row["tickets_booked"] = tickets
        if "hour" in row:
            row["hour"] = hour   # (only if used in training)

        # One-hot encoding
        device_col = f"device_type_{device}"
        location_col = f"location_{location}"

        if device_col in row:
            row[device_col] = 1

        if location_col in row:
            row[location_col] = 1

        # Convert to DataFrame
        X = pd.DataFrame([row])

        # Ensure correct column order
        X = X[MODEL_COLUMNS]

        # Predict
        prob = model.predict_proba(X)[0][1]

        return int(round(prob * 100))   # ✅ FIXED (no comma)

    except Exception as e:
        print("ML ERROR:", e)
        return _fallback(amount, tickets, device, location, hour)

def ml_predict_batch(df_input):
    df=df_input.copy()
    df.columns=df.columns.str.lower().str.strip()
    required={"amount","tickets_booked","device_type","location"}
    missing=required-set(df.columns)
    if missing: return None,f"Missing columns: {missing}"
    scores=[]
    for _,row in df.iterrows():
        try: s=ml_predict_single(float(row["amount"]),int(row["tickets_booked"]),str(row["device_type"]).strip(),str(row["location"]).strip(),int(row.get("hour",12)))
        except: s=0
        scores.append(s)
    return scores,None
def _fallback(amount, tickets, device, location, hour):
    s = 0   # ✅ IMPORTANT: initialize variable

    # Amount risk
    if amount > 10000:
        s += 30
    elif amount > 5000:
        s += 20
    elif amount > 4000:
        s += 15
    elif amount > 2000:
        s += 8

    # Tickets risk
    if tickets > 10:
        s += 25
    elif tickets > 5:
        s += 12

    # Time risk
    if 0 <= hour <= 4:
        s += 20
    elif 22 <= hour <= 23:
        s += 15

    # Device risk
    device_scores = {
        "Mobile": 5,
        "Laptop": 10,
        "Tablet": 8,
        "Unknown": 20
    }
    s += device_scores.get(device, 10)

    # Location risk
    location_scores = {
        "Bangalore": 5,
        "Delhi": 12,
        "Mumbai": 8,
        "Other": 15
    }
    s += location_scores.get(location, 10)

    return min(max(int(s), 0), 100)

def risk_label(score):
    if score>=65:   return "HIGH","pill-high","🔴"
    elif score>=35: return "MEDIUM","pill-medium","🟡"
    else:           return "LOW","pill-low","🟢"

def get_fi():
    if not MODEL_LOADED: return []
    return sorted(zip(MODEL_COLUMNS,model.feature_importances_),key=lambda x:x[1],reverse=True)[:6]

def inject_css():
    ac=st.session_state.accent
    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"],.block-container,.main,.main>div,.stApp>div{{background-color:#f1f5f9!important;font-family:'Inter',sans-serif!important;}}
html,body,.stApp,p,span,div,li,td,th,small,strong,em,a,label,.stMarkdown,.stText,.stCaption,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] span{{color:#1e293b!important;font-family:'Inter',sans-serif!important;}}
h1,h2,h3,h4,h5,h6{{color:{ac}!important;font-weight:700!important;font-family:'Inter',sans-serif!important;}}
section[data-testid="stSidebar"],section[data-testid="stSidebar"]>div,section[data-testid="stSidebar"]>div>div{{background-color:#1e293b!important;}}
section[data-testid="stSidebar"] p,section[data-testid="stSidebar"] span,section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] div,section[data-testid="stSidebar"] small,section[data-testid="stSidebar"] strong{{color:#e2e8f0!important;}}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{{color:{ac}!important;}}
[data-testid="stMetric"]{{background:#ffffff!important;border-radius:12px!important;padding:16px!important;border:1px solid #e2e8f0!important;box-shadow:0 1px 6px rgba(0,0,0,0.06)!important;}}
[data-testid="stMetricValue"]{{color:{ac}!important;font-size:clamp(16px,3vw,28px)!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important;}}
[data-testid="stMetricLabel"]{{color:#64748b!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:0.06em!important;font-weight:600!important;}}
.stButton>button{{background-color:{ac}!important;color:#ffffff!important;border:none!important;border-radius:8px!important;font-weight:600!important;font-size:14px!important;padding:10px 22px!important;font-family:'Inter',sans-serif!important;transition:filter .2s,transform .15s!important;width:100%!important;}}
.stButton>button:hover{{filter:brightness(1.1)!important;transform:translateY(-1px)!important;color:#ffffff!important;}}
.stButton>button p{{color:#ffffff!important;}}
.stTextInput input,.stNumberInput input,textarea,[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{{background:#ffffff!important;color:#1e293b!important;border:1.5px solid #cbd5e1!important;border-radius:8px!important;font-family:'Inter',sans-serif!important;font-size:14px!important;}}
[data-baseweb="input"]{{background:#ffffff!important;border:1.5px solid #cbd5e1!important;border-radius:8px!important;}}
[data-baseweb="select"]>div{{background:#ffffff!important;border:1.5px solid #cbd5e1!important;border-radius:8px!important;color:#1e293b!important;}}
[data-baseweb="select"] span{{color:#1e293b!important;}}
[data-baseweb="menu"],[data-baseweb="popover"]>div{{background:#ffffff!important;border:1px solid #e2e8f0!important;border-radius:8px!important;box-shadow:0 4px 16px rgba(0,0,0,0.1)!important;}}
[data-baseweb="option"],li[role="option"]{{background:#ffffff!important;color:#1e293b!important;}}
[data-testid="stFileUploader"]{{background:#ffffff!important;border:2px dashed #cbd5e1!important;border-radius:12px!important;padding:12px!important;}}
.stDataFrame{{border-radius:10px!important;border:1px solid #e2e8f0!important;overflow:hidden!important;}}
::-webkit-scrollbar{{width:5px;height:5px;}}::-webkit-scrollbar-track{{background:#f1f5f9;}}::-webkit-scrollbar-thumb{{background:#cbd5e1;border-radius:4px;}}::-webkit-scrollbar-thumb:hover{{background:{ac};}}
.card{{background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;box-shadow:0 1px 4px rgba(0,0,0,0.04);margin-bottom:12px;}}
.card p,.card div,.card span,.card small{{color:#475569!important;font-size:13px;}}
.brand{{font-size:clamp(14px,2.5vw,20px)!important;font-weight:700!important;color:{ac}!important;}}
.badge{{background:#ef4444!important;color:#fff!important;padding:2px 7px;border-radius:50px;font-size:11px;font-weight:700;}}
.pill-high{{display:inline-block;background:#fef2f2;border:1.5px solid #ef4444;color:#dc2626!important;padding:8px 16px;border-radius:8px;font-weight:700;font-size:14px;margin-top:12px;}}
.pill-medium{{display:inline-block;background:#fffbeb;border:1.5px solid #f59e0b;color:#d97706!important;padding:8px 16px;border-radius:8px;font-weight:700;font-size:14px;margin-top:12px;}}
.pill-low{{display:inline-block;background:#f0fdf4;border:1.5px solid #22c55e;color:#16a34a!important;padding:8px 16px;border-radius:8px;font-weight:700;font-size:14px;margin-top:12px;}}
.model-ok{{display:inline-block;background:#f0fdf4;border:1.5px solid #22c55e;color:#16a34a!important;padding:3px 10px;border-radius:20px;font-weight:600;font-size:11px;}}
.model-warn{{display:inline-block;background:#fffbeb;border:1.5px solid #f59e0b;color:#d97706!important;padding:3px 10px;border-radius:20px;font-weight:600;font-size:11px;}}
.stTabs [data-baseweb="tab-list"]{{background:#f1f5f9!important;border-radius:8px!important;gap:4px!important;}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:#64748b!important;border-radius:6px!important;font-weight:500!important;}}
.stTabs [aria-selected="true"]{{background:{ac}!important;color:#ffffff!important;}}
.stTabs [aria-selected="true"] p{{color:#ffffff!important;}}
hr{{border-color:#e2e8f0!important;margin:12px 0!important;}}
@media(max-width:768px){{.block-container{{padding:8px!important;}}h1{{font-size:18px!important;}}h2{{font-size:15px!important;}}[data-testid="stMetric"]{{padding:10px!important;}}}}
@media(max-width:480px){{.block-container{{padding:4px!important;}}h1{{font-size:16px!important;}}}}
/* ✅ FIX: restore Material Icons */
    .material-icons,
    button[kind="header"] span,
    [data-testid="collapsedControl"] span {{
        font-family: 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
    }}
</style>""",unsafe_allow_html=True)

def topbar():
    c1,c2,c3=st.columns([5,1,1])
    with c1:
        badge="<span class='model-ok'>✅ RF Model</span>" if MODEL_LOADED else "<span class='model-warn'>⚠️ Fallback</span>"
        st.markdown(f"<div class='brand'>🛡️ E-Ticket Fraud Detection &nbsp;{badge}</div>",unsafe_allow_html=True)
    with c2:
        n=st.session_state.notifications
        st.markdown(f"🔔 <span class='badge'>{n}</span>" if n else "🔔",unsafe_allow_html=True)
    with c3:
        uname=st.session_state.get("current_user","User")
        ch=st.selectbox("u",[f"👤 {uname}","🚪 Logout"],label_visibility="collapsed")
        if ch=="🚪 Logout":
            st.session_state.logged_in=False; st.session_state.current_user=None
            st.session_state.history=[]; st.session_state.notifications=0
            st.rerun()

def sidebar():
    st.sidebar.markdown("## 🛡️ Navigation")
    menu=st.sidebar.radio("nav",[
        "🏠 Dashboard","🔍 Single Prediction","📂 Dataset Prediction",
        "📊 Analytics","📡 Live Monitor","🤖 Model Insights",
        "👥 User Management","🎨 Theme"
    ],label_visibility="collapsed")
    st.sidebar.markdown("---")
    total=len(st.session_state.history)
    high=sum(1 for h in st.session_state.history if h.get("risk",0)>=65)
    st.sidebar.markdown(f"**Predictions:** {total}")
    if total: st.sidebar.markdown(f"**High-risk:** 🔴 {high}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<small>Engine: {'Random Forest ML' if MODEL_LOADED else 'Rule-based'}<br>v3.0 — Internship Major Project</small>",unsafe_allow_html=True)
    return menu

# ── DASHBOARD ──
def page_dashboard():
    st.title("📊 Fraud Intelligence Overview")
    total=len(st.session_state.history)
    high=sum(1 for h in st.session_state.history if h.get("risk",0)>=65)
    medium=sum(1 for h in st.session_state.history if 35<=h.get("risk",0)<65)
    avg=round(np.mean([h["risk"] for h in st.session_state.history]),1) if total else 0.0
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Predictions",total)
    c2.metric("High-Risk 🔴",high,delta=f"{round(high/total*100,1)}% of total" if total else None)
    c3.metric("Medium-Risk 🟡",medium)
    c4.metric("Avg Risk Score",f"{avg}%")
    st.markdown("---")
    if MODEL_LOADED:
        fi=get_fi()
        ca,cb=st.columns(2)
        with ca:
            st.markdown("""<div class='card' style='border-left:4px solid #22c55e;'>
            <strong style='color:#16a34a!important;font-size:15px;'>✅ Random Forest Model Active</strong><br><br>
            <span>Using <strong>RandomForestClassifier</strong> — predictions via <code>predict_proba()</code>.</span><br><br>
            <span style='color:#64748b!important;'>Features: amount, tickets_booked, device_type (OHE), location (OHE)</span>
            </div>""",unsafe_allow_html=True)
        with cb:
            if fi:
                st.markdown("**Top Feature Importances**")
                for feat,imp in fi:
                    pct=round(imp*100,1)
                    lbl=feat.replace("device_type_","📱 ").replace("location_","📍 ").replace("_"," ").title()
                    st.markdown(f"""<div style='margin-bottom:8px;'>
                    <div style='font-size:12px;color:#64748b!important;font-weight:500;margin-bottom:2px;'>{lbl} — {pct}%</div>
                    <div style='background:#e2e8f0;border-radius:4px;height:9px;'>
                    <div style='background:{st.session_state.accent};height:9px;width:{min(pct*3,100)}%;border-radius:4px;'></div></div></div>""",unsafe_allow_html=True)
    if total:
        st.subheader("Recent Predictions")
        df=pd.DataFrame(st.session_state.history[-50:]).copy()
        df.index=range(1,len(df)+1)
        df.columns=[c.replace("_"," ").title() for c in df.columns]
        st.dataframe(df,use_container_width=True)
    else:
        st.info("👋 No predictions yet — use **🔍 Single Prediction** or **📂 Dataset Prediction** to start.")

# ── SINGLE PREDICTION ──
def page_single():
    st.title("🔍 Single Transaction Prediction")
    st.markdown(f"**Engine:** {'🤖 Random Forest `predict_proba()`' if MODEL_LOADED else '⚠️ Rule-based fallback'}")
    st.markdown("---")
    col1,col2=st.columns(2)
    with col1:
        st.subheader("Transaction Details")
        amount=st.number_input("Transaction Amount (₹)",min_value=0.0,step=100.0,value=5000.0)
        tickets=st.number_input("Tickets Booked",min_value=1,max_value=20,step=1,value=3)
    with col2:
        st.subheader("Context")
        device=st.selectbox("Device Used",["Mobile","Laptop","Tablet"])
        location=st.selectbox("Location",["Bangalore","Chennai","Delhi","Hyderabad","Mumbai","Pune"])
        # st.markdown("<div class='card'><strong>Note</strong><br><span><code>hour</code> was dropped during training so it is not used here.</span></div>",unsafe_allow_html=True)
        hour = st.number_input("Transaction Hour (0–23)", min_value=0, max_value=23, value=12)
    if st.button("🔍 Predict Fraud Risk",use_container_width=True):
        with st.spinner("Running model inference…"): time.sleep(0.35)
        risk=ml_predict_single(amount,tickets,device,location,hour)
        label,pill,icon=risk_label(risk)
        st.session_state.history.append({"risk":risk,"amount":amount,"tickets":int(tickets),"device":device,"location":location,"label":label,"source":"manual"})
        if risk>=65: st.session_state.notifications+=1
        r1,r2,r3=st.columns(3)
        r1.metric("Fraud Probability",f"{risk}%")
        r2.metric("Risk Level",f"{icon} {label}")
        r3.metric("Transaction ID",f"TXN{np.random.randint(100000,999999)}")
        st.markdown(f"<div class='{pill}'>{icon} <strong>{label} RISK</strong> — {risk}% fraud probability</div>",unsafe_allow_html=True)
        if risk>=65: st.error("🚨 HIGH RISK — Block or manual review recommended.")
        elif risk>=35: st.warning("⚠️ MEDIUM RISK — Additional verification recommended.")
        else: st.success("✅ LOW RISK — Transaction appears legitimate.")
        with st.expander("🔎 View model input vector"):
            if MODEL_LOADED:
                row={c:0 for c in MODEL_COLUMNS}
                row["amount"]=amount; row["tickets_booked"]=tickets; row["hour"]=hour
                if f"device_type_{device}" in row: row[f"device_type_{device}"]=1
                if f"location_{location}"  in row: row[f"location_{location}"]=1
                st.dataframe(pd.DataFrame([row])[MODEL_COLUMNS],use_container_width=True)

# ── DATASET PREDICTION ──
def page_dataset():
    st.title("📂 Dataset Batch Prediction")
    st.markdown("Upload a **CSV file** (same format as training dataset) and get fraud predictions for every row.")
    st.markdown("---")

    with st.expander("📋 Required CSV columns & sample download"):
        st.markdown("""Your CSV must have **at minimum** these columns (case-insensitive):

| Column | Example values |
|---|---|
| `amount` | 5000, 1200, 9800 |
| `tickets_booked` | 3, 10, 14 |
| `device_type` | Mobile, Laptop, Tablet |
| `location` | Bangalore, Delhi, Mumbai, Chennai, Hyderabad, Pune |
| `hour` | 0–23 |

Optional (kept but not used for prediction): `transaction_id`, `user_id`, `hour`, `is_fraud`""")
        sample=pd.DataFrame({"transaction_id":["T0","T1","T2"],"user_id":["U100","U101","U102"],"amount":[7320,910,5440],"tickets_booked":[11,12,12],"device_type":["Laptop","Laptop","Mobile"],"location":["Bangalore","Chennai","Hyderabad"],"hour":[2,14,5]})
        st.download_button("⬇️ Download Sample CSV",sample.to_csv(index=False).encode(),"sample_upload.csv","text/csv")

    uploaded=st.file_uploader("📁 Upload your CSV file here",type=["csv"],help="Max 200 MB. Must have: amount, tickets_booked, device_type, location")

    if uploaded is not None:
        try:
            df_raw=pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}"); return
        st.success(f"✅ File loaded — **{len(df_raw)} rows**, **{len(df_raw.columns)} columns**")
        st.subheader("Preview (first 5 rows)")
        st.dataframe(df_raw.head(),use_container_width="stretch")

        if st.button("🚀 Run Batch Prediction on Entire Dataset",use_container_width="stretch"):
            with st.spinner(f"Scoring {len(df_raw)} transactions…"):
                scores,err=ml_predict_batch(df_raw)
            if err: st.error(f"❌ {err}"); return

            df_result=df_raw.copy()
            df_result["fraud_probability_%"]=scores
            df_result["risk_level"]=[risk_label(s)[0] for s in scores]

            total_r=len(scores)
            high_r=sum(1 for s in scores if s>=65)
            med_r=sum(1 for s in scores if 35<=s<65)
            low_r=sum(1 for s in scores if s<35)
            avg_r=round(np.mean(scores),1)

            st.markdown("---")
            st.subheader("📊 Batch Results Summary")
            m1,m2,m3,m4,m5=st.columns(5)
            m1.metric("Total Rows",total_r)
            m2.metric("🔴 High Risk",high_r,delta=f"{round(high_r/total_r*100,1)}%")
            m3.metric("🟡 Medium Risk",med_r,delta=f"{round(med_r/total_r*100,1)}%")
            m4.metric("🟢 Low Risk",low_r,delta=f"{round(low_r/total_r*100,1)}%")
            m5.metric("Avg Score",f"{avg_r}%")

            ch1,ch2=st.columns(2)
            with ch1:
                st.subheader("Risk Distribution")
                st.bar_chart(pd.DataFrame({"Count":[high_r,med_r,low_r]},index=["🔴 High","🟡 Medium","🟢 Low"]))
            with ch2:
                st.subheader("Score Histogram")
                ranges=["0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-100"]
                counts=[0]*10
                for s in scores:
                    idx=min(int(s//10),9)
                    counts[idx]+=1
                st.bar_chart(pd.DataFrame({"Count":counts},index=ranges))

            if "is_fraud" in df_result.columns:
                pred_bin=[1 if s>=65 else 0 for s in scores]
                correct=sum(p==a for p,a in zip(pred_bin,df_result["is_fraud"]))
                acc=round(correct/len(pred_bin)*100,2)
                st.info(f"📈 Accuracy vs actual `is_fraud` labels: **{acc}%** ({correct}/{len(pred_bin)} correct)")

            st.subheader("📋 Full Results Table")
            st.dataframe(df_result,use_container_width=True,height=400)
            st.download_button("⬇️ Download Results CSV",df_result.to_csv(index=False).encode(),"fraud_predictions.csv","text/csv",use_container_width=True)

            for _,row in df_result[df_result["risk_level"]=="HIGH"].head(50).iterrows():
                st.session_state.history.append({"risk":int(row["fraud_probability_%"]),"amount":row.get("amount",0),"tickets":row.get("tickets_booked",0),"device":row.get("device_type","Unknown"),"location":row.get("location","Unknown"),"label":"HIGH","source":"batch"})
            st.session_state.notifications+=high_r

            if high_r>0: st.error(f"🚨 {high_r} HIGH-RISK transactions detected!")
            else: st.success("✅ No HIGH-RISK transactions found.")
    else:
        st.markdown("---")
        st.markdown("**Or try the built-in training dataset:**")
        if st.button("⚡ Run on Built-in Dataset (1000 rows)",use_container_width=True):
            try:
                df_builtin=pd.read_csv("eticket_fraud_data.csv")
                with st.spinner("Scoring 1000 transactions…"):
                    scores,err=ml_predict_batch(df_builtin)
                if err: st.error(err); return
                df_result=df_builtin.copy()
                df_result["fraud_probability_%"]=scores
                df_result["risk_level"]=[risk_label(s)[0] for s in scores]
                high_r=sum(1 for s in scores if s>=65)
                med_r=sum(1 for s in scores if 35<=s<65)
                low_r=sum(1 for s in scores if s<35)
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Total",len(scores)); m2.metric("🔴 High",high_r)
                m3.metric("🟡 Medium",med_r); m4.metric("🟢 Low",low_r)
                if "is_fraud" in df_result.columns:
                    pred_bin=[1 if s>=65 else 0 for s in scores]
                    correct=sum(p==a for p,a in zip(pred_bin,df_result["is_fraud"]))
                    acc=round(correct/len(pred_bin)*100,2)
                    st.info(f"📈 Accuracy vs actual labels: **{acc}%** ({correct}/{len(pred_bin)} correct)")
                st.subheader("Results (first 100 rows)")
                st.dataframe(df_result.head(100),use_container_width=True,height=400)
                st.download_button("⬇️ Download Full Results",df_result.to_csv(index=False).encode(),"builtin_predictions.csv","text/csv",use_container_width=True)
            except FileNotFoundError:
                st.error("eticket_fraud_data.csv not found. Place it in the same folder as app.py")

# ── ANALYTICS ──
def page_analytics():
    st.title("📊 Analytics Dashboard")
    if not st.session_state.history: st.info("Run some predictions first."); return
    df=pd.DataFrame(st.session_state.history)
    c1,c2,c3=st.columns(3)
    c1.metric("Total Predictions",len(df)); c2.metric("Average Risk",f"{df['risk'].mean():.1f}%"); c3.metric("Peak Risk",f"{df['risk'].max()}%")
    st.markdown("---")
    cl,cr=st.columns(2)
    with cl:
        st.subheader("Risk Score Trend")
        st.line_chart(pd.DataFrame({"#":range(1,len(df)+1),"Risk %":df["risk"].values}).set_index("#"))
    with cr:
        st.subheader("Risk Distribution")
        bins={"LOW (0–34)":0,"MEDIUM (35–64)":0,"HIGH (65–100)":0}
        for r in df["risk"]:
            if r>=65: bins["HIGH (65–100)"]+=1
            elif r>=35: bins["MEDIUM (35–64)"]+=1
            else: bins["LOW (0–34)"]+=1
        st.bar_chart(pd.DataFrame.from_dict(bins,orient="index",columns=["Count"]))
    cd,cloc=st.columns(2)
    with cd:
        st.subheader("By Device")
        d=df["device"].value_counts().reset_index(); d.columns=["Device","Count"]
        st.bar_chart(d.set_index("Device"))
    with cloc:
        st.subheader("By Location")
        l=df["location"].value_counts().reset_index(); l.columns=["Location","Count"]
        st.bar_chart(l.set_index("Location"))
    st.subheader("Avg Risk by Device")
    st.bar_chart(df.groupby("device")["risk"].mean().rename("Avg Risk %"))
    if "source" in df.columns:
        st.subheader("Prediction Source")
        st.bar_chart(df["source"].value_counts().rename("Count"))

# ── LIVE MONITOR ──
def page_live():
    st.title("📡 Live Fraud Monitor")
    if not st.session_state.history: st.info("No predictions yet."); return
    risks=[h["risk"] for h in st.session_state.history]
    ph=st.empty(); st.caption("Replaying session predictions in real-time…")
    for i in range(1,len(risks)+1):
        sub=risks[:i]
        df=pd.DataFrame({"#":range(1,i+1),"Risk %":sub})
        with ph.container():
            c1,c2,c3=st.columns(3)
            c1.metric("Latest",f"{sub[-1]}%"); c2.metric("Average",f"{round(np.mean(sub),1)}%"); c3.metric("Max",f"{max(sub)}%")
            st.line_chart(df.set_index("#"))
        time.sleep(0.25)

# ── MODEL INSIGHTS ──
def page_model():
    st.title("🤖 Model Insights")
    if not MODEL_LOADED: st.error("model.pkl not found. Run train.py first."); return
    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Architecture")
        st.markdown(f"""<div class='card'>
        <p><strong>Algorithm:</strong> Random Forest Classifier</p>
        <p><strong>Estimators:</strong> {model.n_estimators}</p>
        <p><strong>Max Depth:</strong> {model.max_depth or 'None (unlimited)'}</p>
        <p><strong>Features:</strong> {len(MODEL_COLUMNS)}</p>
        <p><strong>Classes:</strong> 0=Legitimate, 1=Fraud</p>
        <p><strong>Prediction:</strong> predict_proba() → fraud %</p>
        </div>""",unsafe_allow_html=True)
    with c2:
        st.subheader("Feature Columns")
        for col in MODEL_COLUMNS:
            clean=col.replace("device_type_","📱 ").replace("location_","📍 ").replace("_"," ").title()
            st.markdown(f"• `{col}` → {clean}")
    st.markdown("---")
    st.subheader("Feature Importances")
    all_imp=sorted(zip(MODEL_COLUMNS,model.feature_importances_),key=lambda x:x[1],reverse=True)
    st.bar_chart(pd.DataFrame(all_imp,columns=["Feature","Importance"]).set_index("Feature"))
    st.markdown("---")
    st.subheader("Quick Probe")
    p1,p2,p3,p4=st.columns(4)
    with p1: pa=st.number_input("Amount",100.0,15000.0,5000.0,key="pa")
    with p2: pt=st.number_input("Tickets",1,20,5,key="pt")
    with p3: pd_=st.selectbox("Device",["Mobile","Laptop","Tablet"],key="pd_")
    with p4: pl=st.selectbox("Location",["Bangalore","Chennai","Delhi","Hyderabad","Mumbai","Pune"],key="pl")
    if st.button("⚡ Quick Predict"):
        sc=ml_predict_single(pa,pt,pd_,pl); lb,pil,ic=risk_label(sc)
        st.markdown(f"<div class='{pil}'>{ic} <strong>{lb} RISK</strong> — {sc}%</div>",unsafe_allow_html=True)

# ── USER MANAGEMENT ──
def page_users():
    st.title("👥 User Management")
    if st.session_state.current_user!="admin": st.warning("⚠️ Only **admin** can manage users."); return
    st.markdown("---")
    users=_load_users()
    st.subheader(f"Registered Users ({len(users)})")
    for uname in list(users.keys()):
        c1,c2=st.columns([5,1])
        with c1: st.markdown(f"{'👑' if uname=='admin' else '👤'} **{uname}**")
        with c2:
            if uname!="admin" and uname!=st.session_state.current_user:
                if st.button("🗑️",key=f"del_{uname}"): delete_user(uname); st.success(f"'{uname}' deleted."); st.rerun()
            else: st.markdown("<small style='color:#94a3b8'>protected</small>",unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("➕ Add New User")
    a1,a2,a3=st.columns(3)
    with a1: au=st.text_input("Username",key="au")
    with a2: ap_=st.text_input("Password",type="password",key="ap_")
    with a3:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("Add User",use_container_width=True):
            if au and ap_:
                ok,msg=register_user(au,ap_)
                (st.success if ok else st.error)(msg)
            else: st.error("Fill in both fields.")

# ── THEME ──
def page_theme():
    st.title("🎨 Theme")
    st.markdown("---")
    st.subheader("Presets")
    presets=[("🔵 Blue","#2563eb"),("🟣 Violet","#7c3aed"),("🟢 Emerald","#059669"),("⚫ Slate","#475569")]
    cols=st.columns(4)
    for i,(name,color) in enumerate(presets):
        with cols[i%4]:
            if st.button(name,use_container_width=True): st.session_state.accent=color; st.rerun()
    st.markdown("---")
    cp,_,_=st.columns([1,1,2])
    with cp: custom=st.color_picker("Custom colour",value=st.session_state.accent)
    if st.button("✅ Apply"): st.session_state.accent=custom; st.rerun()
    ac=st.session_state.accent
    st.markdown(f"""<div class='card' style='border-top:3px solid {ac};'>
    <div style='font-size:17px;font-weight:700;color:{ac};margin-bottom:10px;'>🛡️ E-Ticket Fraud Detection</div>
    <div style='display:flex;gap:8px;flex-wrap:wrap;'>
    <span style='background:{ac};color:#fff;padding:7px 14px;border-radius:8px;font-weight:600;font-size:13px;'>Button</span>
    <span style='background:#f0fdf4;border:1.5px solid #22c55e;color:#16a34a;padding:7px 12px;border-radius:8px;font-weight:600;font-size:12px;'>LOW 🟢</span>
    <span style='background:#fffbeb;border:1.5px solid #f59e0b;color:#d97706;padding:7px 12px;border-radius:8px;font-weight:600;font-size:12px;'>MEDIUM 🟡</span>
    <span style='background:#fef2f2;border:1.5px solid #ef4444;color:#dc2626;padding:7px 12px;border-radius:8px;font-weight:600;font-size:12px;'>HIGH 🔴</span>
    </div></div>""",unsafe_allow_html=True)

def main_app():
    inject_css(); topbar(); menu=sidebar()
    {"🏠 Dashboard":page_dashboard,"🔍 Single Prediction":page_single,"📂 Dataset Prediction":page_dataset,
    "📊 Analytics":page_analytics,"📡 Live Monitor":page_live,"🤖 Model Insights":page_model,
    "👥 User Management":page_users,"🎨 Theme":page_theme}.get(menu,page_dashboard)()

def login():
    inject_css()
    _,col,_=st.columns([1,1.8,1])
    with col:
        st.markdown("<br><br>",unsafe_allow_html=True)
        ac=st.session_state.accent
        st.markdown(f"""<div style='text-align:center;margin-bottom:24px;'>
        <div style='font-size:52px;'>🛡️</div>
        <div style='font-size:clamp(18px,4vw,24px);font-weight:700;color:{ac};margin-top:8px;'>E-Ticket Fraud Detection</div>
        <div style='font-size:13px;color:#64748b;margin-top:6px;'>Internship Major Project — Powered by Random Forest ML</div>
        </div>""",unsafe_allow_html=True)
        tab_l,tab_r=st.tabs(["🔐 Login","📝 Register"])
        with tab_l:
            st.markdown("<br>",unsafe_allow_html=True)
            u=st.text_input("Username",placeholder="Enter username",key="lu")
            p=st.text_input("Password",type="password",placeholder="Enter password",key="lp")
            if st.button("🔐 Login",use_container_width=True,key="btn_l"):
                if not u or not p: st.error("Please enter both fields.")
                elif verify_user(u,p): st.session_state.logged_in=True; st.session_state.current_user=u; st.rerun()
                else: st.error("❌ Invalid username or password.")
            st.markdown("<p style='text-align:center;color:#94a3b8;font-size:12px;margin-top:10px;'>Default: <strong>admin</strong>/admin123 &nbsp;·&nbsp; <strong>analyst</strong>/analyst123 &nbsp;·&nbsp; <strong>demo</strong>/demo123</p>",unsafe_allow_html=True)
        with tab_r:
            st.markdown("<br>",unsafe_allow_html=True)
            nu=st.text_input("Choose Username",placeholder="Min 3 characters",key="ru")
            np_=st.text_input("Choose Password",type="password",placeholder="Min 6 characters",key="rp")
            cp2=st.text_input("Confirm Password",type="password",placeholder="Re-enter password",key="rc")
            if st.button("✅ Create Account",use_container_width=True,key="btn_r"):
                if not nu or not np_ or not cp2: st.error("Fill in all fields.")
                elif np_!=cp2: st.error("❌ Passwords do not match.")
                else:
                    ok,msg=register_user(nu,np_)
                    (st.success if ok else st.error)(("✅ " if ok else "❌ ")+msg+(" You can now log in." if ok else ""))

if st.session_state.logged_in:
    main_app()
else:
    login()
