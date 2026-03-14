"""
╔══════════════════════════════════════════════════════════════════════╗
║     AL AMEN BANK — Système Intelligent d'Analyse de Risque Crédit       ║
║     Version 4.0 PROFESSIONAL — AL Amen Bank Tunisie                     ║
║     pip install streamlit pandas numpy scikit-learn xgboost plotly   ║
║                 lightgbm scipy                                        ║
║     streamlit run amen_bank_app_v4.py                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, datetime, hashlib, warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64 as _b64

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, AdaBoostClassifier)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              roc_curve, precision_recall_curve,
                              mean_squared_error, mean_absolute_error, r2_score)
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
#  COULEURS AL AMEN BANK
# ═══════════════════════════════════════════════════
VERT      = "#006B3C"
VERT_C    = "#00A651"
OR        = "#F5A623"
NOIR      = "#1A1A1A"
BLANC     = "#FFFFFF"
FOND      = "#F0F4F2"
VERT_DARK = "#004D2C"
VERT_BG   = "#E8F5EE"
ROUGE     = "#DC2626"
BLEU      = "#1A4FA0"

HISTORIQUE_CSV = "amen_bank_historique_analyses.csv"

# ═══════════════════════════════════════════════════
#  LOGO SVG
# ═══════════════════════════════════════════════════
_LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 220 220'>"
    "<defs>"
    "<radialGradient id='bg' cx='40%' cy='35%' r='65%'>"
    "<stop offset='0%' stop-color='#EBEBEB'/>"
    "<stop offset='100%' stop-color='#C4C4C4'/>"
    "</radialGradient>"
    "<filter id='sh'><feDropShadow dx='0' dy='3' stdDeviation='5' flood-opacity='0.20'/></filter>"
    "</defs>"
    "<circle cx='110' cy='110' r='109' fill='#BBBBBB' filter='url(#sh)'/>"
    "<circle cx='110' cy='110' r='106' fill='url(#bg)'/>"
    "<path d='M 58 50 C 80 14,156 16,174 68 C 186 100,172 134,150 145"
    " C 138 152,122 154,108 148 C 124 142,140 128,142 108"
    " C 144 86,126 64,104 60 C 84 56,64 70,54 92"
    " C 52 76,50 64,58 50 Z' fill='#1A4FA0'/>"
    "<path d='M 50 172 C 28 150,24 112,44 84 C 56 66,76 56,96 58"
    " C 80 66,66 82,66 104 C 66 128,84 146,110 149"
    " C 124 151,140 144,150 132 C 144 150,128 166,108 172"
    " C 86 180,64 182,50 172 Z' fill='#00A651'/>"
    "<circle cx='110' cy='150' r='13' fill='#1A4FA0'/>"
    "<ellipse cx='84' cy='68' rx='25' ry='13' fill='rgba(255,255,255,0.22)'"
    " transform='rotate(-30,84,68)'/>"
    "</svg>"
)
LOGO_B64 = _b64.b64encode(_LOGO_SVG.encode()).decode()
LOGO_SRC = f"data:image/svg+xml;base64,{LOGO_B64}"

# ═══════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════
st.set_page_config(
    page_title="Amen Bank — Risque Crédit v4.0",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Cairo:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{{font-family:'Cairo',sans-serif;background-color:{FOND};}}
section[data-testid="stSidebar"]{{background:linear-gradient(180deg,{VERT_DARK} 0%,{VERT} 55%,{VERT_C} 100%);border-right:4px solid {OR};}}
section[data-testid="stSidebar"] *{{color:{BLANC}!important;}}
section[data-testid="stSidebar"] .stRadio label{{color:{OR}!important;font-weight:700;}}
.amen-header{{background:linear-gradient(135deg,{VERT_DARK} 0%,{VERT} 55%,{VERT_C} 100%);padding:1.2rem 2rem;border-radius:14px;display:flex;align-items:center;justify-content:space-between;margin-bottom:1.6rem;box-shadow:0 6px 28px rgba(0,107,60,0.28);border-bottom:4px solid {OR};}}
.amen-header h1{{font-family:'Playfair Display',serif;font-size:1.8rem;color:{BLANC};margin:0;}}
.amen-header .sub{{color:rgba(255,255,255,.72);font-size:.8rem;margin-top:3px;}}
.logo-right{{display:flex;align-items:center;gap:12px;}}
.logo-right img{{height:56px;width:56px;border-radius:50%;border:3px solid {OR};box-shadow:0 2px 12px rgba(0,0,0,.3);}}
.logo-text{{text-align:right;}}
.logo-name{{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:{OR};letter-spacing:3px;line-height:1;}}
.logo-since{{font-size:.6rem;color:rgba(255,255,255,.65);letter-spacing:2px;text-transform:uppercase;}}
.kpi-card{{background:{BLANC};border-radius:14px;padding:1.2rem 1rem;text-align:center;box-shadow:0 4px 16px rgba(0,107,60,.1);border-top:5px solid {VERT};transition:transform .2s,box-shadow .2s;margin-bottom:.5rem;}}
.kpi-card:hover{{transform:translateY(-4px);box-shadow:0 8px 24px rgba(0,107,60,.18);}}
.kpi-icon{{font-size:1.6rem;}}
.kpi-value{{font-size:1.9rem;font-weight:700;color:{VERT};font-family:'Playfair Display',serif;margin:4px 0 2px;}}
.kpi-label{{font-size:.68rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;}}
.kpi-danger{{border-top-color:{ROUGE}!important;}}.kpi-danger .kpi-value{{color:{ROUGE}!important;}}
.kpi-success{{border-top-color:{VERT_C}!important;}}.kpi-success .kpi-value{{color:{VERT_C}!important;}}
.kpi-or{{border-top-color:{OR}!important;}}.kpi-or .kpi-value{{color:{OR}!important;}}
.kpi-blue{{border-top-color:{BLEU}!important;}}.kpi-blue .kpi-value{{color:{BLEU}!important;}}
.section-title{{font-family:'Playfair Display',serif;font-size:1.2rem;color:{VERT_DARK};border-left:5px solid {OR};padding-left:12px;margin:1.6rem 0 1rem;}}
.stButton>button{{background:linear-gradient(135deg,{VERT} 0%,{VERT_C} 100%)!important;color:{BLANC}!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-family:'Cairo',sans-serif!important;font-size:.95rem!important;padding:.55rem 1.8rem!important;transition:all .2s!important;box-shadow:0 4px 14px rgba(0,107,60,.35)!important;}}
.stButton>button:hover{{transform:translateY(-2px)!important;box-shadow:0 7px 20px rgba(0,107,60,.45)!important;}}
.login-card{{background:{BLANC};border-radius:18px;padding:2.5rem 2.8rem;box-shadow:0 16px 48px rgba(0,107,60,.15);border-top:7px solid {VERT};}}
.login-logo-wrap{{display:flex;align-items:center;justify-content:center;gap:16px;margin-bottom:.5rem;}}
.login-logo-img{{height:72px;width:72px;border-radius:50%;border:3px solid {OR};}}
.login-logo-txt{{font-family:'Playfair Display',serif;font-size:3rem;font-weight:700;color:{VERT};letter-spacing:4px;line-height:1;}}
.login-or{{color:{OR};}}
.login-bank{{text-align:center;font-size:.68rem;color:#9CA3AF;letter-spacing:3px;text-transform:uppercase;margin-bottom:.3rem;}}
.login-tag{{text-align:center;color:#6B7280;font-size:.82rem;margin-bottom:1.6rem;border-top:1px solid #E5E7EB;padding-top:.8rem;margin-top:.5rem;}}
.result-good{{background:#D1FAE5;border:2px solid {VERT_C};border-radius:14px;padding:1.5rem;text-align:center;}}
.result-bad{{background:#FEE2E2;border:2px solid {ROUGE};border-radius:14px;padding:1.5rem;text-align:center;}}
.result-title{{font-size:1.8rem;font-weight:700;font-family:'Playfair Display',serif;}}
.info-bar{{background:{VERT_BG};border-left:4px solid {VERT};border-radius:0 8px 8px 0;padding:.7rem 1.2rem;margin-bottom:1rem;font-size:.87rem;color:{VERT_DARK};}}
.badge{{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.68rem;font-weight:700;letter-spacing:.5px;text-transform:uppercase;}}
.badge-red{{background:#FEE2E2;color:{ROUGE};}}.badge-green{{background:#D1FAE5;color:{VERT};}}.badge-or{{background:#FEF3C7;color:#92400E;}}
.progress-wrap{{background:#E5E7EB;border-radius:99px;height:12px;overflow:hidden;margin:6px 0;}}
.progress-fill{{height:100%;border-radius:99px;transition:width .6s ease;}}
.footer{{text-align:center;padding:1.5rem;color:#9CA3AF;font-size:.72rem;border-top:1px solid #E5E7EB;margin-top:3rem;}}
.footer span{{color:{VERT};font-weight:600;}}
.nav-section-label{{font-size:.58rem;letter-spacing:2.5px;text-transform:uppercase;
  color:rgba(255,255,255,.35);font-weight:700;padding:.6rem .3rem .2rem;margin-top:.3rem;}}
.nav-item{{display:flex;align-items:center;gap:10px;padding:.52rem .8rem;border-radius:9px;
  margin-bottom:3px;cursor:pointer;transition:all .18s;border:1px solid transparent;}}
.nav-item:hover{{background:rgba(255,255,255,.10);border-color:rgba(245,166,35,.25);}}
.nav-item.active{{background:linear-gradient(90deg,rgba(245,166,35,.22),rgba(245,166,35,.06));
  border-color:{OR}66;box-shadow:0 2px 10px rgba(245,166,35,.2);}}
.nav-icon{{font-size:1.05rem;width:22px;text-align:center;flex-shrink:0;}}
.nav-label{{font-size:.82rem;font-weight:600;color:{BLANC};flex:1;}}
.nav-item.active .nav-label{{color:{OR};}}
.nav-badge{{font-size:.58rem;font-weight:700;padding:1px 7px;border-radius:99px;
  background:rgba(245,166,35,.2);color:{OR};border:1px solid {OR}55;}}
.nav-badge-new{{background:rgba(0,166,81,.25);color:#6EE7B7;border-color:#6EE7B733;}}
.nav-divider{{border:none;border-top:1px solid rgba(255,255,255,.1);margin:.6rem 0;}}
.data-stat-chip{{display:inline-flex;align-items:center;gap:5px;background:rgba(255,255,255,.08);
  border-radius:8px;padding:3px 9px;font-size:.7rem;color:rgba(255,255,255,.75);
  border:1px solid rgba(255,255,255,.12);margin:2px;}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  AUTH
# ═══════════════════════════════════════════════════
def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()
USERS = {
    "koussay":   {"hash": _hash("koussay2004"), "role": "Administrateur",  "name": "Koussay Hassana"},
    "bechir":  {"hash": _hash("bechir2001"),   "role": "Analyste Crédit", "name": "Bechir Ghoudi"},
    "tarek": {"hash": _hash("tarek2026"),     "role": "Directeur",       "name": "Tarek Bouchaddekh"},
}
def check_password(u, p): return u in USERS and USERS[u]["hash"] == _hash(p)

for k, v in [("logged_in",False),("username",""),("login_error","")]:
    if k not in st.session_state: st.session_state[k] = v

# ═══════════════════════════════════════════════════
#  HISTORIQUE
# ═══════════════════════════════════════════════════
def save_prediction(client, pred, risk_proba, conf, analyste, model_name=""):
    row = {"Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "Analyste": analyste, "Modele": model_name,
           "Age": client["Age"], "Sexe": client["Sex"],
           "Emploi": client["Job"], "Logement": client["Housing"],
           "Epargne": client["Saving accounts"],
           "Compte_Courant": client["Checking account"],
           "Montant_Credit": client["Credit amount"],
           "Duree_Mois": client["Duration"], "Objet": client["Purpose"],
           "Score_Risque_pct": round(risk_proba*100, 2),
           "Confiance_pct": round(conf, 2),
           "Decision": "RISQUE ÉLEVÉ" if pred==1 else "BON CLIENT",
           "Statut": "bad" if pred==1 else "good"}
    df_row = pd.DataFrame([row])
    if os.path.exists(HISTORIQUE_CSV):
        df_row.to_csv(HISTORIQUE_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(HISTORIQUE_CSV, mode="w", header=True, index=False)

def load_historique():
    if os.path.exists(HISTORIQUE_CSV): return pd.read_csv(HISTORIQUE_CSV)
    return pd.DataFrame(columns=["Date","Analyste","Modele","Age","Sexe","Emploi",
        "Logement","Epargne","Compte_Courant","Montant_Credit","Duree_Mois",
        "Objet","Score_Risque_pct","Confiance_pct","Decision","Statut"])

# ═══════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════
@st.cache_data
def load_data():
    for fname, sep, idx in [("german_credit_FINAL_ML_READY.csv",";",None),
                              ("german_credit_data.csv",",",0)]:
        try:
            df = pd.read_csv(fname, sep=sep, index_col=idx)
            if "Risk" in df.columns: break
        except FileNotFoundError: continue
    else:
        raise FileNotFoundError("Aucun fichier de données trouvé.")
    for col in ["Saving accounts","Checking account"]:
        if col in df.columns: df[col] = df[col].fillna(df[col].mode()[0])
    df["Risk"] = df["Risk"].str.strip().str.lower()
    return df[df["Risk"].isin(["good","bad"])].copy()

# ═══════════════════════════════════════════════════
#  FEATURES
# ═══════════════════════════════════════════════════
@st.cache_data
def prepare_features(df):
    CAT = ["Sex","Housing","Saving accounts","Checking account","Purpose"]
    NUM = ["Age","Job","Credit amount","Duration"]
    cat_values = {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT}
    df2 = df.copy()
    df2["Credit_per_month"] = df2["Credit amount"] / df2["Duration"]
    df2["Age_group"] = pd.cut(df2["Age"],bins=[0,25,35,50,100],labels=[0,1,2,3]).astype(int)
    df2["Risk_bin"]  = (df2["Risk"]=="bad").astype(int)
    FEAT = NUM + ["Credit_per_month","Age_group"] + CAT
    X    = df2[FEAT].copy()
    X_enc = pd.get_dummies(X, columns=CAT, drop_first=False)
    return X_enc, df2["Risk_bin"], X_enc.columns.tolist(), cat_values, NUM, CAT

# ═══════════════════════════════════════════════════
#  ENTRAÎNEMENT — 8+ modèles + Ensemble
# ═══════════════════════════════════════════════════
@st.cache_resource
def train_models(_X, _y, feat_cols):
    X_tr, X_te, y_tr, y_te = train_test_split(_X, _y, test_size=0.2, random_state=42, stratify=_y)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(X_tr)
    Xte_sc = sc.transform(X_te)

    def metrics(model, Xtr, Xte, ytr, yte):
        model.fit(Xtr, ytr)
        yp = model.predict(Xte)
        try:
            yp_proba = model.predict_proba(Xte)[:,1]
            auc  = roc_auc_score(yte, yp_proba)
            fpr, tpr, _ = roc_curve(yte, yp_proba)
            prec_c, rec_c, _ = precision_recall_curve(yte, yp_proba)
            mse = mean_squared_error(yte, yp_proba)
            mae = mean_absolute_error(yte, yp_proba)
            r2  = r2_score(yte, yp_proba)
        except Exception:
            yp_proba = yp.astype(float)
            auc = float("nan")
            fpr = tpr = prec_c = rec_c = np.array([0,1])
            mse = mae = r2 = float("nan")
        cv = cross_val_score(model, Xtr, ytr, cv=5, scoring="accuracy")
        return {
            "Acc Train": accuracy_score(ytr, model.predict(Xtr)),
            "Acc Test":  accuracy_score(yte, yp),
            "Précision": precision_score(yte, yp, zero_division=0),
            "Rappel":    recall_score(yte, yp, zero_division=0),
            "F1-Score":  f1_score(yte, yp, zero_division=0),
            "F1_bad":    f1_score(yte, yp, pos_label=1, zero_division=0),
            "Rappel_bad":    recall_score(yte, yp, pos_label=1, zero_division=0),
            "Précision_bad": precision_score(yte, yp, pos_label=1, zero_division=0),
            "AUC-ROC":   auc, "MSE": mse, "MAE": mae, "R²": r2,
            "CV_mean":   cv.mean(), "CV_std": cv.std(),
            "_fpr": fpr, "_tpr": tpr, "_prec_c": prec_c, "_rec_c": rec_c,
            "_cm": confusion_matrix(yte, yp),
            "_model": model, "_yp_proba": yp_proba,
        }

    res = {}
    gb = GradientBoostingClassifier(n_estimators=150,max_depth=4,learning_rate=0.08,subsample=0.85,random_state=42)
    res["Gradient Boosting"] = metrics(gb, X_tr.values, X_te.values, y_tr, y_te)
    rf = RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=10,random_state=42)
    res["Forêt Aléatoire"]   = metrics(rf, X_tr.values, X_te.values, y_tr, y_te)
    if XGB_AVAILABLE:
        res["XGBoost"] = metrics(
            xgb.XGBClassifier(n_estimators=150,max_depth=5,eta=0.08,subsample=0.85,
                               colsample_bytree=0.85,verbosity=0,random_state=42,eval_metric="logloss"),
            X_tr.values, X_te.values, y_tr, y_te)
    if LGB_AVAILABLE:
        res["LightGBM"] = metrics(
            lgb.LGBMClassifier(n_estimators=150,max_depth=5,learning_rate=0.08,subsample=0.85,random_state=42,verbose=-1),
            X_tr.values, X_te.values, y_tr, y_te)
    res["AdaBoost"] = metrics(
        AdaBoostClassifier(n_estimators=100,learning_rate=0.5,random_state=42),
        X_tr.values, X_te.values, y_tr, y_te)
    res["Régression Logistique"] = metrics(
        LogisticRegression(max_iter=2000,C=0.5,random_state=42), Xtr_sc, Xte_sc, y_tr, y_te)
    res["Arbre Décision"] = metrics(
        DecisionTreeClassifier(max_depth=6,min_samples_split=15,random_state=42),
        X_tr.values, X_te.values, y_tr, y_te)
    k_sc = [cross_val_score(KNeighborsClassifier(n_neighbors=k),Xtr_sc,y_tr,cv=5,scoring="accuracy").mean()
            for k in range(3,20)]
    bk = int(np.argmax(k_sc))+3
    res[f"KNN (K={bk})"] = metrics(KNeighborsClassifier(n_neighbors=bk), Xtr_sc, Xte_sc, y_tr, y_te)

    # Ensemble Voting
    top3 = sorted(res.items(), key=lambda x: x[1]["AUC-ROC"] if not np.isnan(x[1]["AUC-ROC"]) else 0, reverse=True)[:3]
    est = [(n.replace(" ","_").replace("(","").replace(")","").replace("=","")
             .replace("ê","e").replace("é","e").replace("è","e"), m["_model"]) for n,m in top3]
    try:
        vc = VotingClassifier(estimators=est, voting="soft")
        vc.fit(X_tr.values, y_tr)
        res["Ensemble Voting"] = metrics(vc, X_tr.values, X_te.values, y_tr, y_te)
    except Exception:
        pass

    feat_imp = None
    try:
        feat_imp = dict(zip(feat_cols, rf.feature_importances_))
    except Exception:
        pass

    return res, sc, X_tr, X_te, y_tr, y_te, feat_imp

# ═══════════════════════════════════════════════════
#  ENCODER CLIENT
# ═══════════════════════════════════════════════════
def encode_client(df, client_dict, feat_cols):
    CAT = ["Sex","Housing","Saving accounts","Checking account","Purpose"]
    NUM = ["Age","Job","Credit amount","Duration"]
    base = df[NUM+CAT].copy()
    new_row = pd.DataFrame([{c: client_dict[c] for c in NUM+CAT}])
    combined = pd.concat([base, new_row], ignore_index=True)
    combined["Credit_per_month"] = combined["Credit amount"] / combined["Duration"]
    combined["Age_group"] = pd.cut(combined["Age"],bins=[0,25,35,50,100],labels=[0,1,2,3]).astype(int)
    encoded = pd.get_dummies(combined, columns=CAT, drop_first=False)
    for col in feat_cols:
        if col not in encoded.columns: encoded[col] = 0
    return encoded[feat_cols].iloc[[-1]].values

# ═══════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════
def render_header(title, sub="Direction des Risques — Amen Bank Tunisie"):
    st.markdown(f"""
    <div class="amen-header">
      <div><h1>🏦 {title}</h1><div class="sub">{sub}</div></div>
      <div class="logo-right">
        <img src="{LOGO_SRC}" alt="Logo"/>
        <div class="logo-text">
          <div class="logo-name">AMEN</div>
          <div class="logo-since">Banque Tunisienne · Since 1967</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def kpi(col, val, lbl, icon, cls=""):
    col.markdown(f"""
    <div class="kpi-card {cls}">
      <div class="kpi-icon">{icon}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════
def render_sidebar():
    u = USERS[st.session_state.username]

    # Nav items : (key, icon, label, badge_text, badge_class)
    NAV_ITEMS = [
        ("Tableau de Bord",    "🏠", "Tableau de Bord",       "",       ""),
        ("Analyse Exploratoire","📊","Analyse Exploratoire",  "",       ""),
        ("Modèles",            "🤖", "Modèles & Évaluation",  "8+",     ""),
        ("Prédiction",         "🔮", "Prédiction Intelligente","IA",    ""),
        ("Délai",              "⏳", "Délai de Défaut",        "NEW",   "nav-badge-new"),
        ("Données",            "📋", "Données & Historique",  "",       ""),
    ]

    # Initialiser page dans session state pour nav custom
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Tableau de Bord"

    with st.sidebar:
        # ── Profil utilisateur ──
        role_icon = {"Administrateur":"👑","Analyste Crédit":"🔍","Directeur":"🏛️"}.get(u["role"],"👤")
        st.markdown(f"""
        <div style="text-align:center;padding:1.3rem 0 .6rem">
          <div style="position:relative;display:inline-block;margin-bottom:10px">
            <img src="{LOGO_SRC}" style="height:68px;width:68px;border-radius:50%;
              border:3px solid {OR};display:block;
              box-shadow:0 0 0 6px rgba(245,166,35,.15),0 4px 16px rgba(0,0,0,.3)"/>
            <div style="position:absolute;bottom:0;right:0;background:{OR};
              border-radius:50%;width:20px;height:20px;display:flex;align-items:center;
              justify-content:center;font-size:.7rem;border:2px solid {VERT_DARK}">
              {role_icon}</div>
          </div>
          <div style="font-family:'Playfair Display',serif;font-size:1.0rem;
            color:{OR};font-weight:700;letter-spacing:.5px">{u['name']}</div>
          <div style="font-size:.68rem;color:rgba(255,255,255,.55);margin-top:3px;
            letter-spacing:.5px">{u['role']}</div>
          <div style="margin-top:8px;display:flex;justify-content:center;gap:6px;flex-wrap:wrap">
            <span class="data-stat-chip">🟢 En ligne</span>
            <span class="data-stat-chip">👤 {st.session_state.username}</span>
          </div>
        </div>
        <hr class="nav-divider">""", unsafe_allow_html=True)

        # ── Titre navigation ──
        st.markdown(f"""
        <div class="nav-section-label">📌 Navigation principale</div>""",
        unsafe_allow_html=True)

        # ── Items de navigation cliquables ──
        for key, icon, label, badge, badge_cls in NAV_ITEMS:
            is_active = key in st.session_state.nav_page
            active_cls = "active" if is_active else ""
            badge_html = (f'<span class="nav-badge {badge_cls}">{badge}</span>'
                          if badge else "")
            # Séparateur visuel avant Données
            if key == "Données":
                st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
            btn_key = f"nav_btn_{key}"
            st.markdown(f"""
            <div class="nav-item {active_cls}" id="navitem_{key}">
              <span class="nav-icon">{icon}</span>
              <span class="nav-label">{label}</span>
              {badge_html}
            </div>""", unsafe_allow_html=True)
            if st.button(label, key=btn_key, use_container_width=True,
                         help=f"Aller à : {label}"):
                st.session_state.nav_page = key
                st.rerun()

        # ── Masquer les vrais boutons (gardés pour la logique) ──
        st.markdown("""<style>
        [data-testid="stSidebar"] .stButton button {
          opacity:0!important;height:0!important;padding:0!important;
          margin:-28px 0 3px!important;border:none!important;
          box-shadow:none!important;background:transparent!important;
          pointer-events:all!important;cursor:pointer!important;
        }
        </style>""", unsafe_allow_html=True)

        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)

        # ── Infos dataset ──
        st.markdown(f"""
        <div class="nav-section-label">📁 Contexte</div>
        <div style="background:rgba(245,166,35,.08);border-radius:10px;padding:.75rem;
          border:1px solid {OR}33;margin-bottom:.6rem">
          <div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:.4rem">
            <span style="font-size:.72rem;color:{OR};font-weight:700;
              text-transform:uppercase;letter-spacing:1px">Dataset</span>
            <span style="font-size:.65rem;background:rgba(0,166,81,.2);color:#6EE7B7;
              border-radius:99px;padding:1px 7px;border:1px solid #6EE7B733">Actif</span>
          </div>
          <div style="font-size:.82rem;font-weight:600;margin-bottom:.4rem">
            German Credit · ML Ready</div>
          <div style="display:flex;gap:5px;flex-wrap:wrap">
            <span class="data-stat-chip">👥 1 000 clients</span>
            <span class="data-stat-chip">🏷️ 10 variables</span>
            <span class="data-stat-chip">✅ Labels réels</span>
          </div>
        </div>
        <div style="background:rgba(26,79,160,.12);border-radius:10px;padding:.75rem;
          border:1px solid {BLEU}33;margin-bottom:.6rem">
          <div style="font-size:.65rem;color:#93C5FD;font-weight:700;text-transform:uppercase;
            letter-spacing:1px;margin-bottom:.3rem">🚀 Version</div>
          <div style="font-size:.88rem;font-weight:700;color:{BLANC}">v5.0 Professional</div>
          <div style="margin-top:.4rem;display:flex;gap:4px;flex-wrap:wrap">
            <span class="data-stat-chip">XGBoost</span>
            <span class="data-stat-chip">LightGBM</span>
            <span class="data-stat-chip">Survival</span>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Déconnexion ──
        st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
        logout_key = "nav_btn_logout"
        st.markdown(f"""
        <div style="padding:.1rem 0 0">
          <div class="nav-item" style="border-color:rgba(220,38,38,.3)">
            <span class="nav-icon">🚪</span>
            <span class="nav-label" style="color:rgba(255,100,100,.85)!important">
              Se déconnecter</span>
          </div>
        </div>""", unsafe_allow_html=True)
        if st.button("Se déconnecter", key=logout_key, use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.nav_page  = "Tableau de Bord"
            st.rerun()

        # ── Footer sidebar ──
        st.markdown(f"""
        <div style="text-align:center;padding:.7rem 0 0;font-size:.6rem;
          color:rgba(255,255,255,.25);border-top:1px solid {OR}18;margin-top:.3rem">
          © 2025 <span style="color:{OR}88">Amen Bank</span>
          &nbsp;·&nbsp; Direction des Risques<br>
          <span style="color:rgba(255,255,255,.18)">
            Système Risque Crédit · v5.0 Pro</span>
        </div>""", unsafe_allow_html=True)

    return st.session_state.nav_page

# ═══════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════
def page_login():
    st.markdown("""<style>section[data-testid="stSidebar"]{display:none!important}header{display:none!important}</style>""", unsafe_allow_html=True)
    st.markdown(f"""<div style="background:linear-gradient(90deg,{VERT_DARK},{VERT},{VERT_C});height:6px;border-radius:3px;margin-bottom:2rem"></div>""", unsafe_allow_html=True)
    _, col, _ = st.columns([1,1.4,1])
    with col:
        st.markdown(f"""
        <div class="login-card">
          <div class="login-logo-wrap">
            <img class="login-logo-img" src="{LOGO_SRC}" alt="Logo"/>
            <div class="login-logo-txt">AMEN<span class="login-or">●</span></div>
          </div>
          <div class="login-bank">Amen Bank Tunisie</div>
          <div class="login-tag">🏦 Système Intelligent de Gestion du Risque Crédit<br>
            <span style="color:{VERT};font-weight:600">Direction des Risques · Tunis — v4.0</span></div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        username = st.text_input("👤  Identifiant", placeholder="Entrez votre identifiant")
        password = st.text_input("🔒  Mot de passe", type="password", placeholder="••••••••")
        if st.session_state.login_error: st.error(st.session_state.login_error)
        if st.button("🔐  Se connecter", use_container_width=True):
            if check_password(username, password):
                st.session_state.logged_in=True; st.session_state.username=username
                st.session_state.login_error=""; st.rerun()
           

# ═══════════════════════════════════════════════════
#  TABLEAU DE BORD
# ═══════════════════════════════════════════════════
def page_dashboard(df):
    render_header("Tableau de Bord Exécutif")
    bad  = df[df["Risk"]=="bad"]
    good = df[df["Risk"]=="good"]
    pct  = len(bad)/len(df)*100

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    kpi(c1, len(df),                                  "Clients Analysés",  "🏦", "")
    kpi(c2, len(bad),                                 "Risques Élevés",    "⚠️",  "kpi-danger")
    kpi(c3, len(good),                                "Bons Clients",      "✅",  "kpi-success")
    kpi(c4, f"{df['Credit amount'].mean():,.0f} TND", "Montant Moyen",     "💰",  "kpi-or")
    kpi(c5, f"{pct:.1f}%",                            "Taux de Risque",    "📊",  "kpi-danger")
    kpi(c6, f"{df['Duration'].mean():.0f} mois",      "Durée Moyenne",     "⏱️",  "kpi-blue")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.2,1.2,1])

    with col1:
        section("Répartition du Risque Crédit")
        rc = df["Risk"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["✅ Bon Client","⚠️ Risque Élevé"],
            values=[rc.get("good",0),rc.get("bad",0)],
            hole=0.58, marker_colors=[VERT_C,ROUGE],
            textfont_size=13, pull=[0,0.06], textinfo="label+percent"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=10,b=30),
            showlegend=True,legend=dict(orientation="h",y=-0.12),
            annotations=[dict(text=f"<b>{pct:.0f}%</b><br>Risque",
                x=0.5,y=0.5,font_size=18,showarrow=False,font_color=ROUGE)])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Montant Moyen par Objet du Crédit")
        pa = df.groupby("Purpose")["Credit amount"].mean().sort_values()
        fig2 = go.Figure(go.Bar(x=pa.values,y=pa.index,orientation="h",
            marker_color=[VERT if i%2==0 else VERT_C for i in range(len(pa))],
            text=[f"{v:,.0f}" for v in pa.values],textposition="outside"))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=380,margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        section("Taux BAD par Logement")
        grp = df.groupby("Housing")["Risk"].apply(lambda x:(x=="bad").sum()/len(x)*100).reset_index()
        grp.columns=["Housing","pct_bad"]
        grp=grp.sort_values("pct_bad",ascending=False)
        fig_h=go.Figure(go.Bar(x=grp["pct_bad"],y=grp["Housing"],orientation="h",
            marker_color=[ROUGE if v>30 else OR if v>20 else VERT_C for v in grp["pct_bad"]],
            text=[f"{v:.0f}%" for v in grp["pct_bad"]],textposition="outside"))
        fig_h.update_layout(xaxis_title="% BAD",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",height=250,margin=dict(t=10,b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        section("Risque par Tranche d'Âge")
        df2=df.copy()
        df2["Age_bin"]=pd.cut(df2["Age"],bins=[18,25,35,50,75],labels=["18-25","26-35","36-50","51+"])
        ar=df2.groupby("Age_bin",observed=True)["Risk"].apply(lambda x:(x=="bad").sum()/len(x)*100).reset_index()
        fig_a=go.Figure(go.Bar(x=ar["Age_bin"].astype(str),y=ar["Risk"],
            marker_color=[ROUGE if v>35 else OR if v>25 else VERT_C for v in ar["Risk"]],
            text=[f"{v:.0f}%" for v in ar["Risk"]],textposition="outside"))
        fig_a.update_layout(yaxis_title="% BAD",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",height=200,margin=dict(t=10,b=10))
        st.plotly_chart(fig_a, use_container_width=True)

    col3b,col4b = st.columns(2)
    with col3b:
        section("Distribution des Âges — Good vs Bad")
        fig3=go.Figure()
        fig3.add_trace(go.Histogram(x=good["Age"],name="✅ Bon",nbinsx=25,marker_color=VERT_C,opacity=0.75))
        fig3.add_trace(go.Histogram(x=bad["Age"], name="⚠️ Risqué",nbinsx=25,marker_color=ROUGE,opacity=0.75))
        fig3.update_layout(barmode="overlay",paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=10,b=10),legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with col4b:
        section("Durée vs Montant Crédit (avec tendance)")
        fig4=px.scatter(df,x="Duration",y="Credit amount",color="Risk",
            color_discrete_map={"good":VERT_C,"bad":ROUGE},opacity=0.55,
            labels={"Duration":"Durée (mois)","Credit amount":"Montant","Risk":"Risque"},
            trendline="lowess")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=10,b=10))
        st.plotly_chart(fig4, use_container_width=True)

    section("🔥 Heatmap Risque — Épargne × Catégorie Emploi")
    pivot=df.pivot_table(index="Saving accounts",columns="Job",values="Risk",
        aggfunc=lambda x:(x=="bad").sum()/len(x)*100)
    fig5=go.Figure(go.Heatmap(z=pivot.values,x=[f"Job {c}" for c in pivot.columns],
        y=pivot.index,colorscale="RdYlGn_r",
        text=np.round(pivot.values,1),texttemplate="%{text}%",
        showscale=True,colorbar=dict(title="% BAD")))
    fig5.update_layout(height=300,paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=10,b=10))
    st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════
#  ANALYSE EXPLORATOIRE
# ═══════════════════════════════════════════════════
def page_eda(df):
    render_header("Analyse Exploratoire des Données")
    NUM = ["Age","Credit amount","Duration","Job"]
    tab1,tab2,tab3,tab4 = st.tabs(["📈  Distributions","🔗  Corrélations",
                                    "🏷️  Catégorielles","📐  Statistiques"])

    with tab1:
        section("Histogrammes Overlappés — Good vs Bad")
        fig=make_subplots(rows=2,cols=2,subplot_titles=NUM)
        for i,col in enumerate(NUM):
            r,c=(i//2)+1,(i%2)+1
            for risk,clr,op in [("good",VERT_C,0.7),("bad",ROUGE,0.7)]:
                fig.add_trace(go.Histogram(x=df[df["Risk"]==risk][col],name=risk,
                    nbinsx=25,marker_color=clr,opacity=op,showlegend=(i==0)),row=r,col=c)
        fig.update_layout(barmode="overlay",height=480,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=50,b=10),legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        section("Boxplots Good vs Bad")
        fig2=make_subplots(rows=1,cols=4,subplot_titles=NUM)
        for i,col in enumerate(NUM):
            for risk,color in [("good",VERT_C),("bad",ROUGE)]:
                fig2.add_trace(go.Box(y=df[df["Risk"]==risk][col],
                    name=f"{'✅' if risk=='good' else '⚠️'}",
                    marker_color=color,showlegend=(i==0)),row=1,col=i+1)
        fig2.update_layout(height=380,paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=50,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        section("Matrice de Corrélation")
        corr=df[NUM].corr()
        fig3=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
            colorscale="RdBu_r",text=np.round(corr.values,3),texttemplate="<b>%{text}</b>",
            showscale=True,zmin=-1,zmax=1))
        fig3.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",margin=dict(t=20,b=10))
        _,mc,_=st.columns([1,2,1])
        with mc: st.plotly_chart(fig3, use_container_width=True)

        section("Scatter Matrix (Pair Plot)")
        fig_pm=px.scatter_matrix(df,dimensions=NUM,color="Risk",
            color_discrete_map={"good":VERT_C,"bad":ROUGE},opacity=0.4)
        fig_pm.update_traces(diagonal_visible=False)
        fig_pm.update_layout(height=500,paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pm, use_container_width=True)

    with tab3:
        for cat in ["Housing","Saving accounts","Purpose","Sex"]:
            cnt=df.groupby([cat,"Risk"]).size().reset_index(name="count")
            fig4=px.bar(cnt,x=cat,y="count",color="Risk",barmode="group",
                color_discrete_map={"good":VERT_C,"bad":ROUGE},
                title=f"Répartition du Risque par {cat}",
                labels={"count":"Nombre","Risk":"Risque"})
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=50,b=10),height=300)
            st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        section("Statistiques Descriptives — Good vs Bad")
        bad2=df[df["Risk"]=="bad"]; good2=df[df["Risk"]=="good"]
        rows=[]
        for col in NUM:
            rows.append({"Variable":col,
                "Moy GOOD":f"{good2[col].mean():.2f}","Méd GOOD":f"{good2[col].median():.2f}",
                "Éc.T GOOD":f"{good2[col].std():.2f}",
                "Moy BAD":f"{bad2[col].mean():.2f}","Méd BAD":f"{bad2[col].median():.2f}",
                "Éc.T BAD":f"{bad2[col].std():.2f}",
                "Diff %":f"{(bad2[col].mean()-good2[col].mean())/good2[col].mean()*100:+.1f}%"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

        section("Taux de Défaut par Variable Catégorielle")
        for cat in ["Saving accounts","Checking account","Purpose"]:
            grp=df.groupby(cat)["Risk"].apply(lambda x:(x=="bad").sum()/len(x)*100).reset_index()
            grp.columns=[cat,"taux_bad"]; grp=grp.sort_values("taux_bad",ascending=False)
            fig_t=go.Figure(go.Bar(x=grp[cat],y=grp["taux_bad"],
                marker_color=[ROUGE if v>40 else OR if v>25 else VERT_C for v in grp["taux_bad"]],
                text=[f"{v:.1f}%" for v in grp["taux_bad"]],textposition="outside"))
            fig_t.update_layout(title=f"Taux BAD par {cat}",yaxis_title="% BAD",
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=280,margin=dict(t=40,b=10))
            st.plotly_chart(fig_t, use_container_width=True)

# ═══════════════════════════════════════════════════
#  MODÈLES & ÉVALUATION
# ═══════════════════════════════════════════════════
def page_models(df):
    render_header("Modèles ML & Évaluation Complète")
    X_enc,y,feat_cols,_,_,_=prepare_features(df)
    with st.spinner("⚙️  Entraînement de tous les modèles…"):
        res,sc,X_tr,X_te,y_tr,y_te,feat_imp=train_models(X_enc,y,feat_cols)

    rows=[{"Algorithme":n,
           "Acc Train":f"{m['Acc Train']:.3f}","Acc Test":f"{m['Acc Test']:.3f}",
           "Précision":f"{m['Précision']:.3f}","Rappel":f"{m['Rappel']:.3f}",
           "F1-Score":f"{m['F1-Score']:.3f}","F1(bad)":f"{m['F1_bad']:.3f}",
           "AUC-ROC":f"{m['AUC-ROC']:.3f}" if not np.isnan(m["AUC-ROC"]) else "N/A",
           "CV Mean":f"{m['CV_mean']:.3f}","CV Std":f"±{m['CV_std']:.3f}",
           "MSE":f"{m['MSE']:.4f}" if not np.isnan(m['MSE']) else "N/A",
           "MAE":f"{m['MAE']:.4f}" if not np.isnan(m['MAE']) else "N/A",
           "R²":f"{m['R²']:.4f}"  if not np.isnan(m['R²'])  else "N/A"}
          for n,m in res.items()]
    df_r=pd.DataFrame(rows)
    bi=df_r["AUC-ROC"].replace("N/A","0").astype(float).idxmax()
    def hl(row):
        s=f"background-color:{VERT_BG};font-weight:bold;color:{VERT_DARK}"
        return [s if row.name==bi else ""]*len(row)

    tab1,tab2,tab3,tab4=st.tabs(["📊 Comparaison","📈 Courbes ROC & PR",
                                   "🔥 Importance Features","📐 MSE·MAE·R²"])
    with tab1:
        section("Tableau Comparatif — Tous les Algorithmes")
        st.dataframe(df_r.style.apply(hl,axis=1),use_container_width=True,hide_index=True)
        bn=df_r.loc[bi,"Algorithme"]
        st.success(f"🏆 Meilleur modèle : **{bn}** | AUC-ROC : **{df_r.loc[bi,'AUC-ROC']}** | F1(bad) : **{df_r.loc[bi,'F1(bad)']}**")

        section("Accuracy Train vs Test")
        fig_b=go.Figure()
        for name,clr in [("Acc Train",VERT),("Acc Test",OR)]:
            fig_b.add_trace(go.Bar(name=name,x=[r["Algorithme"] for r in rows],
                y=[float(r[name]) for r in rows],marker_color=clr,
                text=[r[name] for r in rows],textposition="outside"))
        fig_b.update_layout(barmode="group",yaxis_range=[0.4,1.15],
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=400,margin=dict(t=10,b=60),legend=dict(orientation="h"))
        fig_b.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_b, use_container_width=True)

        section(f"Matrice de Confusion — {bn}")
        cm=res[bn]["_cm"]
        fig_cm=go.Figure(go.Heatmap(z=cm,x=["Prédit BAD","Prédit GOOD"],
            y=["Réel BAD","Réel GOOD"],colorscale=[[0,"#FFF7ED"],[0.5,OR],[1,VERT]],
            text=cm,texttemplate="<b>%{text}</b>",textfont_size=22,showscale=False))
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=350,margin=dict(t=10,b=10))
        _,mc,_=st.columns([1,2,1])
        with mc: st.plotly_chart(fig_cm, use_container_width=True)
        try:
            tn,fp,fn,tp=cm.ravel()
            c1,c2,c3,c4=st.columns(4)
            kpi(c1,tp,"Vrais Positifs (BAD)","🎯","kpi-danger")
            kpi(c2,tn,"Vrais Négatifs (GOOD)","✅","kpi-success")
            kpi(c3,fp,"Faux Positifs","⚠️","kpi-or")
            kpi(c4,fn,"Faux Négatifs","❌","kpi-danger")
        except Exception:
            pass

        section("Cross-Validation 5-fold — Stabilité")
        cv_data=[(r["Algorithme"],float(r["CV Mean"]),float(r["CV Std"].replace("±",""))) for r in rows]
        fig_cv=go.Figure(go.Bar(
            x=[n for n,_,_ in cv_data],y=[m for _,m,_ in cv_data],
            error_y=dict(type="data",array=[s for _,_,s in cv_data],visible=True),
            marker_color=VERT_C,text=[f"{m:.3f}" for _,m,_ in cv_data],textposition="outside"))
        fig_cv.update_layout(yaxis_range=[0.4,1.1],yaxis_title="CV Accuracy",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=360,margin=dict(t=10,b=60))
        fig_cv.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_cv, use_container_width=True)

    with tab2:
        section("Courbes ROC — Comparaison Tous Modèles")
        colors_roc=[VERT,OR,ROUGE,VERT_C,BLEU,"#7C3AED","#0891B2","#F97316","#10B981","#EC4899"]
        fig_roc=go.Figure()
        fig_roc.add_shape(type="line",x0=0,y0=0,x1=1,y1=1,line=dict(dash="dot",color="grey",width=1))
        for i,(name,m) in enumerate(res.items()):
            if not np.isnan(m["AUC-ROC"]):
                fig_roc.add_trace(go.Scatter(x=m["_fpr"],y=m["_tpr"],
                    name=f"{name} (AUC={m['AUC-ROC']:.3f})",
                    line=dict(color=colors_roc[i%len(colors_roc)],width=2.5),mode="lines"))
        fig_roc.update_layout(xaxis_title="Taux Faux Positifs",yaxis_title="Taux Vrais Positifs",
            xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1.02]),
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=500,legend=dict(orientation="v",x=1.01),margin=dict(t=10,b=40))
        st.plotly_chart(fig_roc, use_container_width=True)

        section("Courbes Precision-Recall — Classe BAD")
        fig_pr=go.Figure()
        for i,(name,m) in enumerate(res.items()):
            if not np.isnan(m["AUC-ROC"]):
                fig_pr.add_trace(go.Scatter(x=m["_rec_c"],y=m["_prec_c"],
                    name=name,line=dict(color=colors_roc[i%len(colors_roc)],width=2.5),mode="lines"))
        fig_pr.update_layout(xaxis_title="Rappel (BAD)",yaxis_title="Précision (BAD)",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=460,legend=dict(orientation="h",y=-0.2),margin=dict(t=10,b=80))
        st.plotly_chart(fig_pr, use_container_width=True)

        section("Radar — Performance Globale par Modèle")
        cats=["Acc Test","Précision","Rappel","F1-Score","F1(bad)"]
        fig_rad=go.Figure()
        for i,row_r in df_r.iterrows():
            vals=[float(row_r[c]) if row_r[c] not in ("N/A","") else 0.0 for c in cats]
            vals.append(vals[0])
            fig_rad.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],
                fill="toself",opacity=0.5,name=row_r["Algorithme"],
                line=dict(color=colors_roc[i%len(colors_roc)],width=2)))
        fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
            paper_bgcolor="rgba(0,0,0,0)",height=480,
            legend=dict(orientation="h",y=-0.15),margin=dict(t=30,b=80))
        _,rc2,_=st.columns([1,3,1])
        with rc2: st.plotly_chart(fig_rad, use_container_width=True)

    with tab3:
        section("🔥 Importance des Variables (Random Forest)")
        if feat_imp:
            fi_df=pd.DataFrame(list(feat_imp.items()),columns=["Feature","Importance"])
            fi_df=fi_df.sort_values("Importance",ascending=False).head(20)
            fig_fi=go.Figure(go.Bar(x=fi_df["Importance"],y=fi_df["Feature"],orientation="h",
                marker_color=[VERT if i<3 else VERT_C if i<7 else OR for i in range(len(fi_df))],
                text=[f"{v:.3f}" for v in fi_df["Importance"]],textposition="outside"))
            fig_fi.update_layout(yaxis=dict(autorange="reversed"),xaxis_title="Importance (Gini)",
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=550,margin=dict(t=10,b=10))
            st.plotly_chart(fig_fi, use_container_width=True)
            st.markdown(f"""
            <div class="info-bar">
            💡 Top 3 features : <b>{fi_df['Feature'].iloc[0]}</b>,
            <b>{fi_df['Feature'].iloc[1]}</b> et <b>{fi_df['Feature'].iloc[2]}</b>.
            Ces variables expliquent la majorité de la détection du risque crédit.
            </div>""", unsafe_allow_html=True)

        section("Permutation Importance — Robustesse des Features")
        try:
            rf_model=res["Forêt Aléatoire"]["_model"]
            perm=permutation_importance(rf_model,X_te.values,y_te,n_repeats=10,random_state=42)
            perm_df=pd.DataFrame({"Feature":feat_cols,"Mean":perm.importances_mean,
                "Std":perm.importances_std}).sort_values("Mean",ascending=False).head(15)
            fig_p=go.Figure(go.Bar(x=perm_df["Mean"],y=perm_df["Feature"],orientation="h",
                error_x=dict(type="data",array=perm_df["Std"].values),
                marker_color=BLEU,text=[f"{v:.3f}" for v in perm_df["Mean"]],textposition="outside"))
            fig_p.update_layout(yaxis=dict(autorange="reversed"),
                xaxis_title="Importance par Permutation",
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=500,margin=dict(t=10,b=10))
            st.plotly_chart(fig_p, use_container_width=True)
        except Exception as e:
            st.info(f"Permutation importance : {e}")

    with tab4:
        section("Métriques MSE · MAE · R² — Qualité des Probabilités Prédites")
        st.markdown(f"""
        <div class="info-bar">
        📌 MSE, MAE et R² évaluent la <b>calibration des probabilités prédites P(BAD)</b>
        vs la vérité terrain (0=GOOD, 1=BAD). Un modèle bien calibré est critique
        pour la prise de décision en gestion du risque crédit.
        </div>""", unsafe_allow_html=True)

        rows_r=[{"Algorithme":n,
                 "MSE ↓":f"{m['MSE']:.5f}" if not np.isnan(m['MSE']) else "N/A",
                 "MAE ↓":f"{m['MAE']:.5f}" if not np.isnan(m['MAE']) else "N/A",
                 "R² ↑":f"{m['R²']:.4f}"   if not np.isnan(m['R²'])  else "N/A",
                 "F1(bad)":f"{m['F1_bad']:.3f}",
                 "Rappel(bad)":f"{m['Rappel_bad']:.3f}",
                 "Préc(bad)":f"{m['Précision_bad']:.3f}"}
                for n,m in res.items()]
        df_reg=pd.DataFrame(rows_r)
        bi_r2 =df_reg["R² ↑"].replace("N/A","0").astype(float).idxmax()
        bi_mse=df_reg["MSE ↓"].replace("N/A","999").astype(float).idxmin()
        bi_f1b=df_reg["F1(bad)"].astype(float).idxmax()

        def hl_r2(row):
            s=f"background-color:{VERT_BG};font-weight:bold;color:{VERT_DARK}"
            return [s if row.name==bi_r2 else ""]*len(row)
        st.dataframe(df_reg.style.apply(hl_r2,axis=1),use_container_width=True,hide_index=True)

        c1,c2,c3=st.columns(3)
        kpi(c1,df_reg.loc[bi_f1b,"Algorithme"],f"Meilleur F1 BAD → {df_reg.loc[bi_f1b,'F1(bad)']}","🥇","kpi-danger")
        kpi(c2,df_reg.loc[bi_r2,"Algorithme"], f"Meilleur R² → {df_reg.loc[bi_r2,'R² ↑']}",       "🥈","kpi-or")
        kpi(c3,df_reg.loc[bi_mse,"Algorithme"],f"MSE minimal → {df_reg.loc[bi_mse,'MSE ↓']}",      "🥉","kpi-success")

        st.markdown("<br>", unsafe_allow_html=True)
        algos_r=[r["Algorithme"] for r in rows_r]
        mse_v=[float(r["MSE ↓"]) if r["MSE ↓"]!="N/A" else 0 for r in rows_r]
        mae_v=[float(r["MAE ↓"]) if r["MAE ↓"]!="N/A" else 0 for r in rows_r]
        r2_v =[float(r["R² ↑"])  if r["R² ↑"] !="N/A" else 0 for r in rows_r]

        fig_reg=make_subplots(rows=1,cols=3,subplot_titles=["MSE (↓ meilleur)","MAE (↓ meilleur)","R² (↑ meilleur)"])
        fig_reg.add_trace(go.Bar(x=algos_r,y=mse_v,marker_color=ROUGE,
            text=[f"{v:.4f}" for v in mse_v],textposition="outside",name="MSE"),row=1,col=1)
        fig_reg.add_trace(go.Bar(x=algos_r,y=mae_v,marker_color=OR,
            text=[f"{v:.4f}" for v in mae_v],textposition="outside",name="MAE"),row=1,col=2)
        fig_reg.add_trace(go.Bar(x=algos_r,y=r2_v,marker_color=VERT_C,
            text=[f"{v:.4f}" for v in r2_v],textposition="outside",name="R²"),row=1,col=3)
        fig_reg.update_layout(height=440,showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(t=40,b=80))
        fig_reg.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:1rem">
          <div style="background:#FEE2E2;border-radius:12px;padding:1rem;border-left:5px solid {ROUGE}">
            <b style="color:#991B1B">📉 MSE — Erreur Quadratique Moyenne</b><br>
            <span style="font-size:.84rem;color:#7F1D1D">
            Pénalise les grandes erreurs. Crucial pour éviter les faux négatifs en crédit (défauts non détectés = pertes financières directes). <b>↓ = meilleur.</b></span>
          </div>
          <div style="background:#FEF3C7;border-radius:12px;padding:1rem;border-left:5px solid {OR}">
            <b style="color:#92400E">📏 MAE — Erreur Absolue Moyenne</b><br>
            <span style="font-size:.84rem;color:#78350F">
            Écart moyen absolu entre proba prédite et classe réelle.
            MAE=0.25 → erreur moyenne de 25% sur chaque probabilité. <b>↓ = meilleur.</b></span>
          </div>
          <div style="background:{VERT_BG};border-radius:12px;padding:1rem;border-left:5px solid {VERT_C}">
            <b style="color:{VERT_DARK}">📈 R² — Coefficient de Détermination</b><br>
            <span style="font-size:.84rem;color:{VERT_DARK}">
            Proportion de variance expliquée par le modèle.
            R²=1 → parfait. R²<0 → pire qu'une prédiction naïve. <b>↑ = meilleur.</b></span>
          </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  PRÉDICTION INTELLIGENTE
# ═══════════════════════════════════════════════════
def page_prediction(df):
    render_header("🔮 Prédiction Intelligente du Risque")
    X_enc,y,feat_cols,cat_map,NUM,CAT=prepare_features(df)
    with st.spinner("⚙️  Optimisation des modèles…"):
        res,sc,X_tr,X_te,y_tr,y_te,feat_imp=train_models(X_enc,y,feat_cols)

    best_auc_name=max(res,key=lambda k:res[k]["AUC-ROC"] if not np.isnan(res[k]["AUC-ROC"]) else 0)
    all_model_names=list(res.keys())

    col_sel1,col_sel2=st.columns([2,1])
    with col_sel1:
        selected_model=st.selectbox("🤖 Sélectionner le Modèle",all_model_names,
            index=all_model_names.index(best_auc_name))
    with col_sel2:
        threshold=st.slider("⚙️ Seuil de décision BAD",0.30,0.70,0.50,0.01,
            help="Abaissez pour être plus prudent (plus de refus)")

    model=res[selected_model]["_model"]
    use_scale=any(k in selected_model for k in ["Logistique","Naive","KNN"])
    m_info=res[selected_model]

    st.markdown(f"""
    <div class="info-bar">
      🤖 <b>{selected_model}</b> &nbsp;|&nbsp;
      AUC-ROC : <b>{m_info['AUC-ROC']:.3f}</b> &nbsp;|&nbsp;
      F1(bad) : <b>{m_info['F1_bad']:.3f}</b> &nbsp;|&nbsp;
      Rappel(bad) : <b>{m_info['Rappel_bad']:.3f}</b> &nbsp;|&nbsp;
      MSE : <b>{m_info['MSE']:.4f}</b> &nbsp;|&nbsp;
      R² : <b>{m_info['R²']:.4f}</b> &nbsp;|&nbsp;
      CV : <b>{m_info['CV_mean']:.3f}±{m_info['CV_std']:.3f}</b> &nbsp;|&nbsp;
      Seuil : <b>{threshold:.0%}</b>
    </div>""", unsafe_allow_html=True)

    section("📋 Informations du Client")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("**👤 Données Personnelles**")
        age     = st.slider("Âge",18,75,35)
        sex     = st.selectbox("Sexe",cat_map["Sex"])
        job     = st.selectbox("Catégorie Emploi",[0,1,2,3],
                    format_func=lambda x:["0–Sans emploi","1–Non qualifié","2–Qualifié","3–Très qualifié"][x])
        housing = st.selectbox("Logement",cat_map["Housing"])
    with c2:
        st.markdown("**💳 Situation Financière**")
        saving   = st.selectbox("Compte Épargne", cat_map["Saving accounts"])
        checking = st.selectbox("Compte Courant", cat_map["Checking account"])
        credit   = st.number_input("Montant Crédit (TND)",250,20000,3000,step=100)
    with c3:
        st.markdown("**📋 Détails du Crédit**")
        duration = st.slider("Durée (mois)",4,72,24)
        purpose  = st.selectbox("Objet du Crédit",cat_map["Purpose"])
        mensualite=credit/duration
        st.markdown(f"""
        <div style="background:{VERT_BG};border-radius:10px;padding:.9rem;border:1px solid {VERT_C}55;margin-top:.5rem">
          <div style="font-size:.7rem;color:{VERT_DARK};font-weight:700">📊 Résumé Crédit</div>
          <div style="font-size:.82rem;color:{VERT_DARK};margin-top:4px">
            Mensualité estimée : <b>{mensualite:,.0f} TND/mois</b><br>
            Âge : <b>{age} ans</b> · Durée : <b>{duration} mois</b><br>
            Ratio durée/âge : <b>{duration/age:.2f}</b>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔮  Lancer l'Analyse de Risque IA", use_container_width=True):
        client={"Age":age,"Job":job,"Credit amount":credit,"Duration":duration,
                "Sex":sex,"Housing":housing,"Saving accounts":saving,
                "Checking account":checking,"Purpose":purpose}
        X_cl=encode_client(df,client,feat_cols)
        if use_scale: X_cl=sc.transform(X_cl)
        proba=model.predict_proba(X_cl)[0]
        risk_proba=proba[1]
        pred=1 if risk_proba>=threshold else 0
        conf=max(proba)*100

        # Vote de tous les modèles
        votes_bad=0; votes_good=0; all_probas={}
        for mname,mdata in res.items():
            mdl=mdata["_model"]
            use_sc2=any(k in mname for k in ["Logistique","Naive","KNN"])
            Xcl_t=encode_client(df,client,feat_cols)
            if use_sc2: Xcl_t=sc.transform(Xcl_t)
            p_bad=mdl.predict_proba(Xcl_t)[0][1]
            all_probas[mname]=p_bad
            if p_bad>=threshold: votes_bad+=1
            else: votes_good+=1

        save_prediction(client,pred,float(risk_proba),float(conf),
                        st.session_state.username,selected_model)
        st.toast("✅ Analyse enregistrée dans l'historique", icon="💾")

        risk_pct=round(risk_proba*100,1)
        if pred==1:
            risk_level="CRITIQUE" if risk_pct>75 else "ÉLEVÉ"
            risk_color="#991B1B" if risk_pct>75 else ROUGE
            bg_color="#FEE2E2"; border_col=ROUGE; icon_main="🔴"
        else:
            risk_level="FAIBLE" if risk_pct<25 else "MODÉRÉ"
            risk_color=VERT if risk_pct<25 else "#D97706"
            bg_color="#D1FAE5" if risk_pct<25 else "#FEF9C3"
            border_col=VERT_C if risk_pct<25 else "#F59E0B"
            icon_main="🟢" if risk_pct<25 else "🟡"

        # ── Résultat principal ──
        st.markdown(f"""
        <div style="background:{bg_color};border:2px solid {border_col};
                    border-radius:18px;padding:1.8rem;text-align:center;
                    box-shadow:0 8px 32px rgba(0,0,0,.12);margin:1rem 0">
          <div style="font-size:3rem">{icon_main}</div>
          <div style="font-family:'Playfair Display',serif;font-size:2.2rem;
                      font-weight:700;color:{risk_color};margin:4px 0">RISQUE {risk_level}</div>
          <div style="font-size:4rem;font-weight:700;color:{risk_color};
                      font-family:'Playfair Display',serif;line-height:1;margin:8px 0">{risk_pct}%</div>
          <div style="font-size:.9rem;color:#6B7280;margin-top:4px">
            Score de Risque BAD &nbsp;·&nbsp; Confiance : <b>{conf:.1f}%</b>
            &nbsp;·&nbsp; Seuil : <b>{threshold:.0%}</b>
          </div>
          <div style="margin-top:12px">
            <span class="badge {'badge-red' if pred==1 else 'badge-green'}">
              {'⛔ REFUS RECOMMANDÉ' if pred==1 else '✅ APPROBATION RECOMMANDÉE'}
            </span>
            &nbsp;
            <span class="badge badge-or">Votes BAD : {votes_bad}/{len(res)} modèles</span>
          </div>
        </div>""", unsafe_allow_html=True)

        gc1,gc2,gc3=st.columns([1.2,1.2,0.8])
        with gc1:
            section("⚡ Jauge de Risque")
            fig_g=go.Figure(go.Indicator(
                mode="gauge+number+delta",value=risk_pct,
                number={"suffix":"%","font":{"color":risk_color,"size":36}},
                delta={"reference":50,"increasing":{"color":ROUGE},"decreasing":{"color":VERT}},
                title={"text":"Score Risque BAD","font":{"size":14,"color":VERT_DARK}},
                gauge={"axis":{"range":[0,100]},"bar":{"color":risk_color,"thickness":0.28},
                       "bgcolor":"white",
                       "steps":[{"range":[0,25],"color":VERT_BG},{"range":[25,50],"color":"#FEF9C3"},
                                 {"range":[50,75],"color":"#FED7AA"},{"range":[75,100],"color":"#FEE2E2"}],
                       "threshold":{"line":{"color":NOIR,"width":3},"thickness":0.85,"value":threshold*100}}))
            fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=300,margin=dict(t=30,b=10,l=20,r=20))
            st.plotly_chart(fig_g, use_container_width=True)

        with gc2:
            section("📊 Consensus des Modèles")
            sorted_proba=sorted(all_probas.items(),key=lambda x:x[1],reverse=True)
            fig_cons=go.Figure(go.Bar(
                x=[v*100 for _,v in sorted_proba],y=[n for n,_ in sorted_proba],
                orientation="h",
                marker_color=[ROUGE if v>=threshold else VERT_C for _,v in sorted_proba],
                text=[f"{v*100:.1f}%" for _,v in sorted_proba],textposition="outside"))
            fig_cons.add_vline(x=threshold*100,line_dash="dash",line_color=NOIR,
                annotation_text=f"Seuil {threshold:.0%}")
            fig_cons.update_layout(xaxis_range=[0,115],
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                height=300,margin=dict(t=10,b=10))
            st.plotly_chart(fig_cons, use_container_width=True)

        with gc3:
            section("🗳️ Vote")
            fig_vote=go.Figure(go.Pie(labels=["BAD","GOOD"],values=[votes_bad,votes_good],
                marker_colors=[ROUGE,VERT_C],hole=0.5,textfont_size=14,pull=[0.05,0]))
            fig_vote.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=300,
                margin=dict(t=10,b=10),
                annotations=[dict(text=f"<b>{votes_bad}</b>/{len(res)}<br>BAD",
                    x=0.5,y=0.5,font_size=14,showarrow=False,font_color=ROUGE)])
            st.plotly_chart(fig_vote, use_container_width=True)

        # ── Facteurs de risque détaillés ──
        section("🔍 Analyse Détaillée des Facteurs de Risque")
        factors_data=[
            ("Montant crédit élevé",   credit>df["Credit amount"].median(),
             f"{credit:,} TND vs médiane {df['Credit amount'].median():,.0f} TND"),
            ("Durée longue",            duration>df["Duration"].median(),
             f"{duration} mois vs médiane {df['Duration'].median():.0f} mois"),
            ("Jeune emprunteur",        age<30,
             f"{age} ans — profil jeune = risque statistiquement plus élevé"),
            ("Épargne insuffisante",    saving in ["little","NA"],
             f"Niveau : {saving}"),
            ("Compte courant faible",   checking in ["little","NA"],
             f"Niveau : {checking}"),
            ("Emploi non qualifié",     job<=1,
             f"Niveau emploi : {job}/3"),
            ("Mensualité élevée",       mensualite>300,
             f"{mensualite:,.0f} TND/mois — effort financier important"),
            ("Objet à risque",          purpose in ["vacation","repairs"],
             f"Objet : {purpose}"),
        ]
        risk_count=sum(1 for _,is_r,_ in factors_data if is_r)
        ok_count  =sum(1 for _,is_r,_ in factors_data if not is_r)

        frow1,frow2=st.columns(2)
        for i,(factor,is_risk,detail) in enumerate(factors_data):
            col_f=frow1 if i%2==0 else frow2
            icon="🔴" if is_risk else "🟢"
            bg="#FEE2E2" if is_risk else "#D1FAE5"
            clr="#991B1B" if is_risk else "#065F46"
            bdclr=ROUGE if is_risk else VERT_C
            col_f.markdown(f"""
            <div style="background:{bg};border-radius:8px;padding:.6rem .9rem;
                        margin-bottom:.5rem;border-left:3px solid {bdclr}">
              <div style="font-weight:700;color:{clr}">{icon} {factor}</div>
              <div style="font-size:.78rem;color:{clr};opacity:.8;margin-top:2px">
                <span class="badge {'badge-red' if is_risk else 'badge-green'}">
                  {'Facteur de risque' if is_risk else 'Facteur favorable'}
                </span>&nbsp; {detail}
              </div>
            </div>""", unsafe_allow_html=True)

        risk_score_f=risk_count/len(factors_data)*100
        bar_clr=ROUGE if risk_score_f>50 else "#F59E0B" if risk_score_f>30 else VERT_C
        st.markdown(f"""
        <div style="background:{VERT_BG};border-radius:10px;padding:1rem;
                    margin-top:.5rem;display:flex;align-items:center;gap:1.5rem">
          <div style="font-size:1.8rem;font-weight:700;color:{VERT_DARK}">{risk_count}/{len(factors_data)}</div>
          <div>
            <div style="font-size:.85rem;font-weight:700;color:{VERT_DARK}">Facteurs de risque détectés</div>
            <div class="progress-wrap" style="width:300px">
              <div class="progress-fill" style="width:{risk_score_f}%;background:{bar_clr}"></div>
            </div>
            <div style="font-size:.72rem;color:#6B7280">
              {ok_count} favorable(s) · {risk_count} à risque</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Recommandation officielle ──
        section("📝 Recommandation Officielle Amen Bank")
        if pred==1:
            montant_max=int(credit*0.6/100)*100; duree_max=min(duration,36)
            st.markdown(f"""
            <div style="background:#FEF2F2;border:2px solid {ROUGE};border-radius:14px;
                        padding:1.4rem;box-shadow:0 4px 16px rgba(220,38,38,.15)">
              <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                          color:{ROUGE};font-weight:700;margin-bottom:.8rem">
                ⛔ DÉCISION : CRÉDIT REFUSÉ — Dossier à Risque {risk_level}</div>
              <div style="font-size:.9rem;color:#7F1D1D;line-height:1.7">
                🔹 Réduire le montant à <b>max {montant_max:,} TND</b> (60% du demandé)<br>
                🔹 Raccourcir la durée à <b>max {duree_max} mois</b><br>
                🔹 Exiger des garanties : hypothèque ou caution solidaire<br>
                🔹 Demander un co-emprunteur avec revenus stables<br>
                🔹 Soumettre au <b>Comité des Risques</b> pour validation finale<br>
                🔹 Vérifier l'historique BCT (Banque Centrale de Tunisie)<br>
                🔹 Proposer un microcrédit garanti comme alternative
              </div>
              <div style="margin-top:.8rem;padding:.6rem;background:rgba(220,38,38,.08);
                          border-radius:8px;font-size:.78rem;color:#991B1B">
                ⚠️ Score : <b>{risk_pct}%</b> &nbsp;·&nbsp; Votes BAD : <b>{votes_bad}/{len(res)}</b>
                &nbsp;·&nbsp; Confiance : <b>{conf:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            taux="standard" if risk_pct<20 else "légèrement majoré"
            st.markdown(f"""
            <div style="background:#F0FDF4;border:2px solid {VERT_C};border-radius:14px;
                        padding:1.4rem;box-shadow:0 4px 16px rgba(0,166,81,.15)">
              <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                          color:{VERT};font-weight:700;margin-bottom:.8rem">
                ✅ DÉCISION : CRÉDIT APPROUVÉ — Profil {risk_level}</div>
              <div style="font-size:.9rem;color:{VERT_DARK};line-height:1.7">
                🔹 Taux d'intérêt : conditions <b>{taux}</b><br>
                🔹 Montant accordé : <b>{credit:,} TND</b> (demandé)<br>
                🔹 Durée : <b>{duration} mois</b> — conforme au profil<br>
                🔹 Suivi <b>semestriel</b> du compte recommandé<br>
                🔹 Dossier conforme aux critères Amen Bank<br>
                🔹 Proposer assurance crédit <b>TAAMINE</b> en couverture complémentaire
              </div>
              <div style="margin-top:.8rem;padding:.6rem;background:rgba(0,166,81,.08);
                          border-radius:8px;font-size:.78rem;color:{VERT_DARK}">
                ✅ Score : <b>{risk_pct}%</b> &nbsp;·&nbsp; Votes GOOD : <b>{votes_good}/{len(res)}</b>
                &nbsp;·&nbsp; Confiance : <b>{conf:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  DONNÉES & HISTORIQUE — AMÉLIORÉ v5
# ═══════════════════════════════════════════════════
def page_data(df):
    render_header("📋 Données Brutes & Historique des Analyses")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📂  Dataset Original",
        "🔍  Historique des Analyses",
        "📈  Statistiques Avancées",
        "🗃️  Gestion & Export",
    ])

    # ══════════════════════════════════════════
    #  TAB 1 — DATASET ORIGINAL
    # ══════════════════════════════════════════
    with tab1:
        section("Dataset German Credit — 1 000 clients (labels réels)")

        st.markdown(f"""
        <div class="info-bar">
          <b>📂 Source :</b> German Credit Dataset (UCI) · 1 000 observations · 10 variables ·
          Label binaire <b>good/bad</b> · Utilisé pour l'entraînement de tous les modèles ML.
        </div>""", unsafe_allow_html=True)

        # Filtres avancés
        fc1,fc2,fc3,fc4,fc5 = st.columns(5)
        rf  = fc1.selectbox("Risque",    ["Tous","good","bad"],                     key="rf")
        sf  = fc2.selectbox("Sexe",      ["Tous","male","female"],                   key="sf")
        hf  = fc3.selectbox("Logement",  ["Tous"]+sorted(df["Housing"].unique()),    key="hf")
        pf  = fc4.selectbox("Objet",     ["Tous"]+sorted(df["Purpose"].unique()),    key="pf")
        savf= fc5.selectbox("Épargne",   ["Tous"]+sorted(df["Saving accounts"].astype(str).unique()), key="savf")

        d = df.copy()
        if rf   != "Tous": d = d[d["Risk"]            == rf]
        if sf   != "Tous": d = d[d["Sex"]             == sf]
        if hf   != "Tous": d = d[d["Housing"]         == hf]
        if pf   != "Tous": d = d[d["Purpose"]         == pf]
        if savf != "Tous": d = d[d["Saving accounts"].astype(str) == savf]

        # KPIs dynamiques
        k1,k2,k3,k4,k5 = st.columns(5)
        kpi(k1, len(d),                   "Clients filtrés",     "👥", "kpi-blue")
        kpi(k2, len(d[d["Risk"]=="bad"]),  "Risques BAD",         "⚠️", "kpi-danger")
        kpi(k3, len(d[d["Risk"]=="good"]), "Profils GOOD",        "✅", "kpi-success")
        kpi(k4, f"{d['Credit amount'].mean():,.0f}" if len(d)>0 else "–",
            "Montant Moyen (TND)", "💰", "kpi-or")
        kpi(k5, f"{d['Duration'].mean():.1f}" if len(d)>0 else "–",
            "Durée Moy. (mois)",  "📅", "kpi-blue")

        st.markdown("<br>", unsafe_allow_html=True)

        # Recherche textuelle simple
        search = st.text_input("🔎  Recherche rapide (valeur dans n'importe quelle colonne)",
                               placeholder="ex: furniture, own, male…", key="data_search")
        if search:
            mask = d.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False)).any(axis=1)
            d = d[mask]
            st.caption(f"🔍 {len(d)} résultat(s) pour « {search} »")

        def cr(v):
            if v == "bad":  return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
            return "background-color:#D1FAE5;color:#065F46;font-weight:700"

        st.dataframe(d.style.applymap(cr, subset=["Risk"]),
                     use_container_width=True, height=460)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("⬇️  Télécharger la sélection (CSV)",
                data=d.to_csv(index=True).encode("utf-8"),
                file_name="amen_bank_dataset_selection.csv", mime="text/csv",
                key="dl_dataset_sel", use_container_width=True)
        with dl2:
            st.download_button("⬇️  Dataset complet (CSV)",
                data=df.to_csv(index=True).encode("utf-8"),
                file_name="amen_bank_dataset_complet.csv", mime="text/csv",
                key="dl_dataset_full", use_container_width=True)

    # ══════════════════════════════════════════
    #  TAB 2 — HISTORIQUE DES ANALYSES
    # ══════════════════════════════════════════
    with tab2:
        section("🔍 Historique des Analyses de Risque")
        hist = load_historique()

        if len(hist) == 0:
            st.markdown(f"""
            <div style="background:{VERT_BG};border:2px dashed {VERT_C};border-radius:14px;
                        padding:3rem;text-align:center;margin-top:1rem">
              <div style="font-size:3rem;margin-bottom:.8rem">📭</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                          color:{VERT_DARK};font-weight:700">Aucune analyse enregistrée</div>
              <div style="color:#6B7280;font-size:.85rem;margin-top:.5rem">
                Effectuez une prédiction dans la page
                <b>🔮 Prédiction Intelligente</b> pour voir l'historique ici.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # KPIs historique
            taux_bad_h = len(hist[hist["Statut"]=="bad"])/len(hist)*100
            h1,h2,h3,h4,h5,h6 = st.columns(6)
            kpi(h1, len(hist),                              "Total Analyses",     "📋","kpi-blue")
            kpi(h2, len(hist[hist["Statut"]=="bad"]),       "Risques BAD",        "⚠️","kpi-danger")
            kpi(h3, len(hist[hist["Statut"]=="good"]),      "Bons Clients",       "✅","kpi-success")
            kpi(h4, f"{hist['Score_Risque_pct'].mean():.1f}%","Score Moyen",      "📊","kpi-or")
            kpi(h5, f"{taux_bad_h:.1f}%",                   "Taux BAD",           "🔴",
                "kpi-danger" if taux_bad_h>40 else "kpi-or")
            kpi(h6, hist["Analyste"].nunique(),             "Analystes Actifs",   "👤","kpi-blue")

            st.markdown("<br>", unsafe_allow_html=True)

            # Filtres
            fc1,fc2,fc3 = st.columns(3)
            hrf  = fc1.selectbox("Filtrer Décision", ["Tous","BON CLIENT","RISQUE ÉLEVÉ"], key="hrf")
            hana = fc2.selectbox("Filtrer Analyste",
                ["Tous"]+sorted(hist["Analyste"].unique().tolist()), key="hana")
            hmod = fc3.selectbox("Filtrer Modèle",
                ["Tous"]+sorted(hist["Modele"].unique().tolist()), key="hmod")

            h = hist.copy()
            if hrf  != "Tous": h = h[h["Decision"] == hrf]
            if hana != "Tous": h = h[h["Analyste"] == hana]
            if hmod != "Tous": h = h[h["Modele"]   == hmod]

            def cr_h(v):
                if v=="bad": return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
                return "background-color:#D1FAE5;color:#065F46;font-weight:700"
            def cr_d(v):
                if "RISQUE" in str(v): return f"color:{ROUGE};font-weight:700"
                return f"color:{VERT};font-weight:700"

            st.dataframe(
                h.style.applymap(cr_h, subset=["Statut"]).applymap(cr_d, subset=["Decision"]),
                use_container_width=True, height=380, hide_index=True)

            # Graphiques historique
            if len(h) >= 2:
                gc1, gc2 = st.columns(2)
                with gc1:
                    section("📉 Évolution du Score de Risque")
                    fig_ev = go.Figure()
                    fig_ev.add_trace(go.Scatter(
                        x=list(range(1,len(h)+1)), y=h["Score_Risque_pct"].values,
                        mode="lines+markers", line=dict(color=VERT,width=2),
                        marker=dict(
                            color=[ROUGE if s=="bad" else VERT_C for s in h["Statut"].values],
                            size=10, line=dict(color="white",width=2)),
                        fill="tozeroy", fillcolor="rgba(0,107,60,0.07)", name="Score"))
                    fig_ev.add_hline(y=50, line_dash="dash", line_color=OR,
                        annotation_text="Seuil 50%")
                    fig_ev.update_layout(xaxis_title="N° Analyse", yaxis_title="Score (%)",
                        yaxis_range=[0,105], paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(t=10,b=10))
                    st.plotly_chart(fig_ev, use_container_width=True)

                with gc2:
                    section("📊 Distribution des Décisions par Analyste")
                    ana_cnt = h.groupby(["Analyste","Statut"]).size().reset_index(name="N")
                    fig_ana = go.Figure()
                    for stat, clr in [("good",VERT_C),("bad",ROUGE)]:
                        sub = ana_cnt[ana_cnt["Statut"]==stat]
                        fig_ana.add_trace(go.Bar(
                            name=stat.upper(), x=sub["Analyste"], y=sub["N"],
                            marker_color=clr, text=sub["N"], textposition="outside"))
                    fig_ana.update_layout(barmode="group",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=300, margin=dict(t=10,b=10), legend=dict(orientation="h"))
                    st.plotly_chart(fig_ana, use_container_width=True)

            # Boutons export / clear
            col_dl, col_clr = st.columns(2)
            with col_dl:
                st.download_button("⬇️  Exporter Historique CSV",
                    data=hist.to_csv(index=False).encode("utf-8"),
                    file_name="amen_bank_historique.csv", mime="text/csv",
                    key="dl_hist", use_container_width=True)
            with col_clr:
                if st.button("🗑️  Effacer l'historique", use_container_width=True, key="clr_hist"):
                    if os.path.exists(HISTORIQUE_CSV): os.remove(HISTORIQUE_CSV)
                    st.success("✅ Historique effacé.")
                    st.rerun()

    # ══════════════════════════════════════════
    #  TAB 3 — STATISTIQUES AVANCÉES
    # ══════════════════════════════════════════
    with tab3:
        section("📈 Profil Statistique du Dataset")

        NUM = ["Age","Job","Credit amount","Duration"]

        # Stats descriptives comparées good vs bad
        bad_df  = df[df["Risk"]=="bad"]
        good_df = df[df["Risk"]=="good"]

        rows_stat = []
        for col in NUM:
            rows_stat.append({
                "Variable":        col,
                "Moy GOOD":        f"{good_df[col].mean():.2f}",
                "Méd GOOD":        f"{good_df[col].median():.2f}",
                "Éc.T GOOD":       f"{good_df[col].std():.2f}",
                "Min GOOD":        f"{good_df[col].min():.0f}",
                "Max GOOD":        f"{good_df[col].max():.0f}",
                "Moy BAD":         f"{bad_df[col].mean():.2f}",
                "Méd BAD":         f"{bad_df[col].median():.2f}",
                "Éc.T BAD":        f"{bad_df[col].std():.2f}",
                "Diff %":          f"{(bad_df[col].mean()-good_df[col].mean())/good_df[col].mean()*100:+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows_stat), use_container_width=True, hide_index=True)

        # Boxplots comparatifs
        section("📦 Boxplots GOOD vs BAD")
        fig_box = make_subplots(rows=1, cols=4, subplot_titles=NUM)
        for i, col in enumerate(NUM):
            for risk, color in [("good",VERT_C),("bad",ROUGE)]:
                fig_box.add_trace(go.Box(
                    y=df[df["Risk"]==risk][col],
                    name=f"{'✅' if risk=='good' else '⚠️'} {risk.upper()}",
                    marker_color=color, showlegend=(i==0)), row=1, col=i+1)
        fig_box.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=40,b=10))
        st.plotly_chart(fig_box, use_container_width=True)

        # Taux de défaut par variable catégorielle
        section("🏷️ Taux BAD par Variable Catégorielle")
        cat_cols = ["Housing","Saving accounts","Checking account","Purpose","Sex"]
        cols_cat = st.columns(2)
        for idx, cat in enumerate(cat_cols):
            grp = df.groupby(cat)["Risk"].apply(
                lambda x: (x=="bad").sum()/len(x)*100).reset_index()
            grp.columns = [cat,"taux_bad"]
            grp = grp.sort_values("taux_bad", ascending=False)
            fig_t = go.Figure(go.Bar(
                x=grp[cat], y=grp["taux_bad"],
                marker_color=[ROUGE if v>50 else OR if v>30 else VERT_C for v in grp["taux_bad"]],
                text=[f"{v:.1f}%" for v in grp["taux_bad"]], textposition="outside"))
            fig_t.update_layout(
                title=f"Taux BAD — {cat}", yaxis_title="% BAD",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=300, margin=dict(t=40,b=10))
            with cols_cat[idx % 2]:
                st.plotly_chart(fig_t, use_container_width=True)

        # Heatmap de corrélation
        section("🔥 Matrice de Corrélation")
        corr = df[NUM].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r",
            text=np.round(corr.values,3), texttemplate="<b>%{text}</b>",
            showscale=True, zmin=-1, zmax=1))
        fig_corr.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10,b=10))
        _, mc, _ = st.columns([1,2,1])
        with mc:
            st.plotly_chart(fig_corr, use_container_width=True)

    # ══════════════════════════════════════════
    #  TAB 4 — GESTION & EXPORT
    # ══════════════════════════════════════════
    with tab4:
        section("🗃️ Gestion des Données & Exports")

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
                    gap:1rem;margin-bottom:1.5rem">

          <div style="background:{BLANC};border-radius:14px;padding:1.3rem;
                      box-shadow:0 4px 16px rgba(0,107,60,.08);border-top:4px solid {VERT}">
            <div style="font-size:1.5rem;margin-bottom:.4rem">📂</div>
            <div style="font-weight:700;color:{VERT_DARK};font-size:.95rem;margin-bottom:.3rem">
              Dataset Original</div>
            <div style="font-size:.78rem;color:#6B7280">1 000 clients · German Credit UCI</div>
          </div>

          <div style="background:{BLANC};border-radius:14px;padding:1.3rem;
                      box-shadow:0 4px 16px rgba(0,107,60,.08);border-top:4px solid {OR}">
            <div style="font-size:1.5rem;margin-bottom:.4rem">🔍</div>
            <div style="font-weight:700;color:{VERT_DARK};font-size:.95rem;margin-bottom:.3rem">
              Historique Analyses</div>
            <div style="font-size:.78rem;color:#6B7280">Log de toutes les prédictions</div>
          </div>

          <div style="background:{BLANC};border-radius:14px;padding:1.3rem;
                      box-shadow:0 4px 16px rgba(0,107,60,.08);border-top:4px solid {BLEU}">
            <div style="font-size:1.5rem;margin-bottom:.4rem">📊</div>
            <div style="font-weight:700;color:{VERT_DARK};font-size:.95rem;margin-bottom:.3rem">
              Rapport Statistiques</div>
            <div style="font-size:.78rem;color:#6B7280">Résumé complet du dataset</div>
          </div>

        </div>""", unsafe_allow_html=True)

        # Export dataset
        section("📤 Exports Disponibles")
        e1,e2,e3 = st.columns(3)
        with e1:
            st.markdown(f"""<div style="background:{VERT_BG};border-radius:10px;
                padding:.8rem;border:1px solid {VERT_C}44;margin-bottom:.6rem;
                font-size:.82rem;color:{VERT_DARK}">
                <b>📂 Dataset complet</b><br>Toutes les 1 000 lignes avec labels</div>""",
                unsafe_allow_html=True)
            st.download_button("⬇️  dataset_complet.csv",
                data=df.to_csv(index=True).encode("utf-8"),
                file_name="amen_bank_dataset_complet.csv", mime="text/csv",
                key="dl_e1", use_container_width=True)
        with e2:
            bad_only = df[df["Risk"]=="bad"]
            st.markdown(f"""<div style="background:#FEF2F2;border-radius:10px;
                padding:.8rem;border:1px solid {ROUGE}44;margin-bottom:.6rem;
                font-size:.82rem;color:#7F1D1D">
                <b>⚠️ Clients BAD uniquement</b><br>{len(bad_only)} dossiers à risque</div>""",
                unsafe_allow_html=True)
            st.download_button("⬇️  clients_bad.csv",
                data=bad_only.to_csv(index=True).encode("utf-8"),
                file_name="amen_bank_clients_bad.csv", mime="text/csv",
                key="dl_e2", use_container_width=True)
        with e3:
            hist_exp = load_historique()
            st.markdown(f"""<div style="background:#FEF9C3;border-radius:10px;
                padding:.8rem;border:1px solid {OR}44;margin-bottom:.6rem;
                font-size:.82rem;color:#78350F">
                <b>🔍 Historique analyses</b><br>
                {len(hist_exp)} analyse(s) enregistrée(s)</div>""",
                unsafe_allow_html=True)
            if len(hist_exp) > 0:
                st.download_button("⬇️  historique.csv",
                    data=hist_exp.to_csv(index=False).encode("utf-8"),
                    file_name="amen_bank_historique.csv", mime="text/csv",
                    key="dl_e3", use_container_width=True)
            else:
                st.button("⬇️  historique.csv (vide)", disabled=True,
                    key="dl_e3_empty", use_container_width=True)

        # Rapport texte complet
        section("📄 Rapport Descriptif Automatique")
        if st.button("📄  Générer le Rapport Dataset", key="btn_rapport_dataset"):
            NUM2 = ["Age","Job","Credit amount","Duration"]
            rapport = f"""AMEN BANK — RAPPORT DESCRIPTIF DATASET
============================================
Généré le : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Source     : German Credit Dataset (UCI)

DIMENSIONS
  Observations : {len(df):,}
  Variables    : {len(df.columns)}
  Labels       : good={len(df[df['Risk']=='good'])} ({len(df[df['Risk']=='good'])/len(df)*100:.1f}%) | bad={len(df[df['Risk']=='bad'])} ({len(df[df['Risk']=='bad'])/len(df)*100:.1f}%)

STATISTIQUES NUMÉRIQUES
{"".join([f'''
  {col}
    Global : moy={df[col].mean():.2f}  médiane={df[col].median():.2f}  écart-type={df[col].std():.2f}  min={df[col].min()}  max={df[col].max()}
    GOOD   : moy={df[df['Risk']=='good'][col].mean():.2f}  médiane={df[df['Risk']=='good'][col].median():.2f}
    BAD    : moy={df[df['Risk']=='bad'][col].mean():.2f}   médiane={df[df['Risk']=='bad'][col].median():.2f}
''' for col in NUM2])}
VARIABLES CATÉGORIELLES (top 3 modalités)
{"".join([f'''
  {col} :
    {"  |  ".join([f"{k}: {v} ({v/len(df)*100:.1f}%)" for k,v in df[col].value_counts().head(3).items()])}
''' for col in ['Housing','Saving accounts','Purpose','Sex']])}
Direction des Risques — Amen Bank Tunisie · v5.0
"""
            st.download_button("⬇️  Télécharger le rapport",
                data=rapport.encode("utf-8"),
                file_name=f"amen_bank_rapport_dataset_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain", key="dl_rapport_dataset")
            st.success("✅ Rapport généré — cliquez sur le bouton ci-dessus pour télécharger.")

        # Zone de danger
        section("⚠️ Zone de Gestion")
        st.warning("⚠️ Les actions ci-dessous sont irréversibles. Utiliser avec précaution.")
        hist_del = load_historique()
        dc1, dc2 = st.columns(2)
        with dc1:
            if len(hist_del) > 0:
                if st.button("🗑️  Effacer tout l'historique des analyses",
                             use_container_width=True, key="clr_hist_tab4"):
                    if os.path.exists(HISTORIQUE_CSV):
                        os.remove(HISTORIQUE_CSV)
                    st.success("✅ Historique supprimé.")
                    st.rerun()
            else:
                st.button("🗑️  Historique déjà vide", disabled=True,
                          use_container_width=True, key="clr_hist_empty")
        with dc2:
            st.markdown(f"""<div style="background:#FEF2F2;border-radius:8px;padding:.7rem;
                font-size:.8rem;color:#991B1B;border:1px solid {ROUGE}33">
                🔒 Le dataset source est en lecture seule.<br>
                Il ne peut pas être modifié depuis cette interface.</div>""",
                unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  DÉLAI DE DÉFAUT — PRÉDICTION TEMPORELLE
# ═══════════════════════════════════════════════════
def page_delai_defaut(df):
    render_header("⏳ Prédiction du Délai avant Défaut")

    st.markdown(f"""
    <div class="info-bar">
      <b>⏳ Moteur de Survie Crédit</b> — Ce module prédit <b>dans combien de mois</b> un client
      risque de basculer en défaut, en combinant l'analyse de survie (Kaplan-Meier), le score ML
      et les caractéristiques du dossier. Renseignez le profil du client ci-dessous.
    </div>""", unsafe_allow_html=True)

    X_enc,y,feat_cols,cat_map,NUM,CAT = prepare_features(df)
    with st.spinner("⚙️  Chargement des modèles…"):
        res,sc,X_tr,X_te,y_tr,y_te,feat_imp = train_models(X_enc,y,feat_cols)

    best_name = max(res, key=lambda k: res[k]["AUC-ROC"] if not np.isnan(res[k]["AUC-ROC"]) else 0)
    model     = res[best_name]["_model"]
    use_sc    = any(k in best_name for k in ["Logistique","Naive","KNN"])

    section("📋 Profil du Client à Analyser")
    f1,f2,f3 = st.columns(3)
    with f1:
        st.markdown("**👤 Données Personnelles**")
        age      = st.slider("Âge", 18, 75, 38, key="dd_age")
        sex      = st.selectbox("Sexe", cat_map["Sex"], key="dd_sex")
        job      = st.selectbox("Catégorie Emploi", [0,1,2,3],
                      format_func=lambda x:["0–Sans emploi","1–Non qualifié","2–Qualifié","3–Très qualifié"][x],
                      key="dd_job")
        housing  = st.selectbox("Logement", cat_map["Housing"], key="dd_housing")
    with f2:
        st.markdown("**💳 Situation Financière**")
        saving   = st.selectbox("Compte Épargne",  cat_map["Saving accounts"],  key="dd_sav")
        checking = st.selectbox("Compte Courant",  cat_map["Checking account"], key="dd_chk")
        credit   = st.number_input("Montant Crédit (TND)", 250, 25000, 5000, step=250, key="dd_cred")
    with f3:
        st.markdown("**📋 Détails du Crédit**")
        duration = st.slider("Durée contractuelle (mois)", 4, 72, 36, key="dd_dur")
        purpose  = st.selectbox("Objet du Crédit", cat_map["Purpose"], key="dd_purp")
        mois_ecoules = st.slider("Mois déjà écoulés depuis octroi", 0, duration, 0, key="dd_ecoules")

    st.markdown("<br>", unsafe_allow_html=True)
    lancer = st.button("⏳  Prédire le Délai avant Défaut", use_container_width=True, key="btn_delai")

    # ── Calcul et persistance dans session_state ──
    if lancer:
        client = {"Age":age,"Job":job,"Credit amount":credit,"Duration":duration,
                  "Sex":sex,"Housing":housing,"Saving accounts":saving,
                  "Checking account":checking,"Purpose":purpose}
        X_cl = encode_client(df, client, feat_cols)
        if use_sc: X_cl = sc.transform(X_cl)
        proba_bad = float(model.predict_proba(X_cl)[0][1])

        facteur_montant  = 1.0 + max(0,(credit-5000)/10000)*0.4
        facteur_epargne  = {"little":1.4,"moderate":1.1,"quite rich":0.8,"rich":0.6}.get(saving,1.2)
        facteur_emploi   = {0:1.6,1:1.3,2:1.0,3:0.75}.get(job,1.0)
        facteur_age      = 1.3 if age<25 else (0.9 if age>45 else 1.0)
        facteur_logement = {"free":1.2,"rent":1.15,"own":0.85}.get(housing,1.0)
        hazard_base = float(np.clip(
            proba_bad*facteur_montant*facteur_epargne*facteur_emploi*facteur_age*facteur_logement,
            0.005, 0.95))

        t_range_l = list(range(0, duration+1))
        survie_l  = [1.0]
        hazards_l = []
        for i_h in range(1, duration+1):
            acc = 1.0+0.015*i_h if i_h<=duration//2 else 1.0+0.015*(duration-i_h+1)
            ht  = float(np.clip(hazard_base*acc/duration, 0.001, 0.99))
            hazards_l.append(ht)
            survie_l.append(survie_l[-1]*(1-ht))

        mois_50 = next((t_range_l[i] for i,s in enumerate(survie_l) if s<0.50), duration)
        mois_25 = next((t_range_l[i] for i,s in enumerate(survie_l) if s<0.25), duration)
        mois_75 = next((t_range_l[i] for i,s in enumerate(survie_l) if s<0.75), duration)

        st.session_state["dd_result"] = {
            "proba_bad":proba_bad, "hazard_base":hazard_base,
            "t_range":t_range_l, "survie":survie_l, "hazards":hazards_l,
            "mois_50":mois_50, "mois_25":mois_25, "mois_75":mois_75,
            "age":age,"sex":sex,"job":job,"housing":housing,
            "saving":saving,"checking":checking,"credit":credit,
            "duration":duration,"purpose":purpose,"mois_ecoules":mois_ecoules,
            "facteur_montant":facteur_montant,"facteur_epargne":facteur_epargne,
            "facteur_emploi":facteur_emploi,"facteur_age":facteur_age,
            "facteur_logement":facteur_logement,
        }

    # ── Si aucun résultat encore : écran d'accueil ──
    if "dd_result" not in st.session_state:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{VERT_BG},{BLANC});border:2px dashed {VERT_C};
                    border-radius:16px;padding:3rem 2rem;text-align:center;margin-top:.5rem">
          <div style="font-size:3.5rem;margin-bottom:.8rem">⏳</div>
          <div style="font-family:'Playfair Display',serif;font-size:1.4rem;
                      color:{VERT_DARK};font-weight:700">Renseignez le profil et lancez la prédiction</div>
          <div style="color:#6B7280;font-size:.87rem;margin-top:.5rem">
            Le moteur calculera le <b>délai estimé avant défaut</b>, la courbe de survie mensuelle,
            les facteurs aggravants et les actions préventives recommandées.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Récupérer résultats persistés ──
    r = st.session_state["dd_result"]
    proba_bad        = r["proba_bad"];    hazard_base      = r["hazard_base"]
    t_range          = np.array(r["t_range"]); survie = np.array(r["survie"])
    hazards          = r["hazards"]
    mois_50          = r["mois_50"];      mois_25 = r["mois_25"]; mois_75 = r["mois_75"]
    age              = r["age"];          sex = r["sex"];         job = r["job"]
    housing          = r["housing"];      saving  = r["saving"];  checking = r["checking"]
    credit           = r["credit"];       duration = r["duration"]; purpose = r["purpose"]
    mois_ecoules     = r["mois_ecoules"]
    facteur_montant  = r["facteur_montant"]; facteur_epargne  = r["facteur_epargne"]
    facteur_emploi   = r["facteur_emploi"];  facteur_age      = r["facteur_age"]
    facteur_logement = r["facteur_logement"]

    delai_median     = max(0, mois_50-mois_ecoules)
    delai_optimiste  = max(0, mois_25-mois_ecoules)
    delai_pessimiste = max(0, mois_75-mois_ecoules)
    mois_restants    = duration-mois_ecoules

    if   delai_median<=6:  urgence,urgence_clr,urgence_bg="🔴 CRITIQUE",ROUGE,"#FEE2E2"
    elif delai_median<=18: urgence,urgence_clr,urgence_bg="🟠 ÉLEVÉ","#F97316","#FEF3C7"
    elif delai_median<=36: urgence,urgence_clr,urgence_bg="🟡 MODÉRÉ",OR,"#FFFBEB"
    else:                  urgence,urgence_clr,urgence_bg="🟢 FAIBLE",VERT_C,"#D1FAE5"

    # Résultat principal
    st.markdown(f"""
    <div style="background:{urgence_bg};border:2px solid {urgence_clr};border-radius:18px;
                padding:1.8rem 2rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.1);margin:1.2rem 0">
      <div style="font-size:2.4rem;margin-bottom:.3rem">⏳</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                  color:{urgence_clr};font-weight:700;margin-bottom:.4rem">
        Risque Temporel : {urgence}</div>
      <div style="display:flex;justify-content:center;gap:3rem;flex-wrap:wrap;margin:1rem 0">
        <div>
          <div style="font-size:3.5rem;font-weight:700;font-family:'Playfair Display',serif;
                      color:{urgence_clr};line-height:1">{delai_median}</div>
          <div style="font-size:.82rem;color:#6B7280;margin-top:4px">mois avant défaut <b>(médian)</b></div>
        </div>
        <div style="border-left:2px solid {urgence_clr}33;padding-left:3rem">
          <div style="font-size:1.1rem;font-weight:700;color:{VERT_DARK}">Fourchette</div>
          <div style="font-size:.88rem;color:#6B7280;margin-top:.3rem;line-height:1.8">
            Optimiste : <b style="color:{VERT_C}">{delai_pessimiste} mois</b><br>
            Médian : <b style="color:{urgence_clr}">{delai_median} mois</b><br>
            Pessimiste : <b style="color:{ROUGE}">{delai_optimiste} mois</b>
          </div>
        </div>
        <div style="border-left:2px solid {urgence_clr}33;padding-left:3rem">
          <div style="font-size:1.1rem;font-weight:700;color:{VERT_DARK}">PD Globale</div>
          <div style="font-size:2.4rem;font-weight:700;color:{urgence_clr};
                      font-family:'Playfair Display',serif">{proba_bad*100:.1f}%</div>
          <div style="font-size:.75rem;color:#6B7280">Probabilité de Défaut</div>
        </div>
      </div>
      <div style="font-size:.8rem;color:#6B7280">
        Crédit <b>{credit:,} TND</b> · <b>{duration} mois</b> ·
        <b>{mois_ecoules}</b> mois écoulés · <b>{mois_restants}</b> mois restants
      </div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    kpi(k1,f"{delai_median} mois","Délai Médian Défaut","⏳",
        "kpi-danger" if delai_median<=6 else "kpi-or" if delai_median<=18 else "kpi-success")
    kpi(k2,f"{proba_bad*100:.1f}%","PD Globale","📉",
        "kpi-danger" if proba_bad>0.5 else "kpi-or" if proba_bad>0.3 else "kpi-success")
    kpi(k3,f"{survie[min(mois_ecoules,len(survie)-1)]*100:.1f}%","Survie Actuelle S(t)","🛡️",
        "kpi-success" if survie[min(mois_ecoules,len(survie)-1)]>0.7 else "kpi-or")
    kpi(k4,f"{mois_restants} mois","Durée Restante","📅","kpi-blue")
    kpi(k5,f"{hazard_base*100:.1f}%","Risque Mensuel Base","⚡",
        "kpi-danger" if hazard_base>0.05 else "kpi-or")

    st.markdown("<br>", unsafe_allow_html=True)

    col_g1, col_g2 = st.columns([2,1])
    with col_g1:
        section("📈 Courbe de Survie Kaplan-Meier Estimée")
        fig_surv = go.Figure()
        fig_surv.add_hrect(y0=0, y1=50, fillcolor="rgba(220,38,38,0.04)", line_width=0)
        fig_surv.add_trace(go.Scatter(
            x=t_range, y=survie*100, mode="lines",
            name="S(t) — Survie", line=dict(color=VERT,width=3),
            fill="tozeroy", fillcolor="rgba(0,107,60,0.08)"))
        fig_surv.add_hline(y=50, line_dash="dash", line_color=ROUGE,
            annotation_text="Défaut probable (50%)", annotation_position="top right",
            annotation_font_color=ROUGE)
        fig_surv.add_hline(y=75, line_dash="dot", line_color=OR,
            annotation_text="Seuil surveillance (75%)", annotation_position="top left",
            annotation_font_color=OR)
        if mois_ecoules>0:
            fig_surv.add_vline(x=mois_ecoules, line_dash="solid", line_color=BLEU,
                annotation_text=f"Aujourd'hui (M{mois_ecoules})",
                annotation_position="top left", annotation_font_color=BLEU)
        if mois_50<=duration:
            fig_surv.add_vline(x=mois_50, line_dash="dash", line_color=ROUGE,
                annotation_text=f"Défaut médian (M{mois_50})",
                annotation_position="top right", annotation_font_color=ROUGE)
        fig_surv.update_layout(
            xaxis_title="Mois depuis l'octroi",yaxis_title="Probabilité de Survie (%)",
            yaxis_range=[0,105], xaxis_range=[0,duration],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(t=20,b=10), legend=dict(orientation="h",y=-0.15))
        st.plotly_chart(fig_surv, use_container_width=True)

    with col_g2:
        section("📊 Risque Mensuel h(t)")
        fig_haz = go.Figure(go.Bar(
            x=list(t_range[1:]), y=[h*100 for h in hazards],
            marker_color=[ROUGE if h>0.07 else OR if h>0.04 else VERT_C for h in hazards]))
        if mois_ecoules>0:
            fig_haz.add_vline(x=mois_ecoules, line_dash="solid",
                line_color=BLEU, annotation_text="Auj.")
        fig_haz.update_layout(
            xaxis_title="Mois", yaxis_title="h(t) %",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=380, margin=dict(t=20,b=10), showlegend=False)
        st.plotly_chart(fig_haz, use_container_width=True)

    section("📋 Tableau Mensuel de Probabilités")
    rows_table = []
    for i, t in enumerate(t_range):
        if t%3==0 or t==duration:
            s = survie[i]
            ht = hazards[i-1]*100 if i>0 else 0
            statut=("🔴 Défaut probable" if s<0.5 else
                    "🟠 Surveillance"    if s<0.75 else
                    "🟡 Attention"       if s<0.90 else "🟢 Sain")
            rows_table.append({"Mois":int(t),"Survie S(t)":f"{s*100:.1f}%",
                "Risque h(t)":f"{ht:.2f}%","Prob. Défaut Cum.":f"{(1-s)*100:.1f}%",
                "Statut":statut,"Mois restants":max(0,duration-t)})
    def col_stat(v):
        if "🔴" in str(v): return "background-color:#FEE2E2;color:#991B1B;font-weight:700"
        if "🟠" in str(v): return "background-color:#FEF3C7;color:#92400E;font-weight:700"
        if "🟡" in str(v): return "background-color:#FFFBEB;color:#78350F;font-weight:700"
        if "🟢" in str(v): return "background-color:#D1FAE5;color:#065F46;font-weight:700"
        return ""
    st.dataframe(pd.DataFrame(rows_table).style.applymap(col_stat,subset=["Statut"]),
                 use_container_width=True, hide_index=True)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        section("🕸️ Radar Profil de Risque")
        cats = ["Score PD","Montant","Épargne","Emploi","Âge","Logement"]
        vals = [min(proba_bad*2,1), min(facteur_montant-.5,1),
                min(facteur_epargne-.5,1), min(facteur_emploi-.5,1),
                min(facteur_age-.5,1), min(facteur_logement-.5,1)]
        vals = [max(0,v) for v in vals]
        fig_r = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor="rgba(220,38,38,0.15)",
            line=dict(color=ROUGE,width=2)))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
            paper_bgcolor="rgba(0,0,0,0)",height=320,margin=dict(t=40,b=20),showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r2:
        section("📉 Scénarios Comparatifs")
        scenarios = [
            ("Profil actuel",         proba_bad, facteur_epargne, facteur_emploi),
            ("Si épargne ↑ (moderate)",proba_bad*0.85,1.1,facteur_emploi),
            ("Si emploi ↑ (qualifié)", proba_bad*0.80,facteur_epargne,1.0),
            ("Si montant −20%",        proba_bad*0.90,facteur_epargne,facteur_emploi),
            ("Si âge +10 ans",         proba_bad*0.88,facteur_epargne,facteur_emploi),
        ]
        sc_noms=[]; sc_delais=[]
        for nom,pd_s,fe_s,fj_s in scenarios:
            hb_s=np.clip(pd_s*facteur_montant*fe_s*fj_s*facteur_age*facteur_logement,0.003,0.95)
            srv_s=[1.0]
            for ti in range(1,duration+1):
                acc=1.0+0.015*ti if ti<=duration//2 else 1.0+0.015*(duration-ti+1)
                ht_s=np.clip(hb_s*acc/duration,0.001,0.99)
                srv_s.append(srv_s[-1]*(1-ht_s))
            d50=next((ti for ti,sv in enumerate(srv_s) if sv<0.5),duration)
            sc_noms.append(nom); sc_delais.append(max(0,d50-mois_ecoules))
        colors_sc=[ROUGE if i==0 else VERT_C if d>sc_delais[0] else OR
                   for i,d in enumerate(sc_delais)]
        fig_sc=go.Figure(go.Bar(y=sc_noms,x=sc_delais,orientation="h",
            marker_color=colors_sc,
            text=[f"{d} mois" for d in sc_delais],textposition="outside"))
        fig_sc.add_vline(x=sc_delais[0],line_dash="dash",line_color=ROUGE)
        fig_sc.update_layout(xaxis_title="Délai avant défaut (mois)",
            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=320,margin=dict(t=10,b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    section("📝 Plan d'Action Préventif Amen Bank")
    if delai_median<=6:
        actions=[("🔴 URGENCE — Convocation immédiate","Planifier un entretien sous 15 jours"),
                 ("📂 Transmission au Comité des Risques","Soumettre en session extraordinaire"),
                 ("🛡️ Activation des garanties","Activer hypothèques, cautions, nantissements"),
                 ("💳 Blocage préventif des lignes","Suspendre toute nouvelle avance")]
        brd=ROUGE; bg_a="#FEF2F2"
    elif delai_median<=18:
        actions=[("🟠 Surveillance renforcée mensuelle","Suivi flux de compte mensuel"),
                 ("📞 Contact préventif client","Appel mensuel — proposer rééchelonnement"),
                 ("📄 Révision des conditions","Étudier réduction mensualité ou extension")]
        brd="#F97316"; bg_a="#FEF3C7"
    elif delai_median<=36:
        actions=[("🟡 Surveillance trimestrielle","Revue dossier tous les 3 mois"),
                 ("💡 Proposition produits épargne","Renforcer l'épargne du client"),
                 ("📊 Réévaluation annuelle du score","Recalculer le score lors du bilan")]
        brd=OR; bg_a="#FFFBEB"
    else:
        actions=[("🟢 Profil sain — Surveillance standard","Revue annuelle du dossier"),
                 ("🌟 Opportunité cross-sell","Proposer produits Amen Bank complémentaires")]
        brd=VERT_C; bg_a="#D1FAE5"

    st.markdown(f"""<div style="background:{bg_a};border:2px solid {brd};border-radius:14px;
                               padding:1.3rem 1.6rem">
      <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700;
                  color:{brd};margin-bottom:.8rem">
        Délai estimé : <b>{delai_median} mois</b> · {urgence}</div>
      {"".join([f'<div style="margin-bottom:.6rem;padding:.55rem 1rem;background:rgba(255,255,255,.65);border-radius:8px;border-left:4px solid {brd}"><div style="font-weight:700;font-size:.87rem;color:{VERT_DARK}">{t}</div><div style="font-size:.8rem;color:#6B7280;margin-top:2px">{d}</div></div>' for t,d in actions])}
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────
    #  EXPORT PDF — RAPPORT DÉLAI DE DÉFAUT
    # ─────────────────────────────────────────────────────────
    section("📄 Télécharger le Rapport PDF")

    emploi_lbl = ["Sans emploi","Non qualifié","Qualifié","Très qualifié"][job]

    # Générer le PDF en mémoire avec reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        import io as _io

        buf = _io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
            leftMargin=1.8*cm, rightMargin=1.8*cm,
            topMargin=1.5*cm, bottomMargin=2*cm)

        C_VERT    = rl_colors.HexColor("#006B3C")
        C_VERT_C  = rl_colors.HexColor("#00A651")
        C_OR      = rl_colors.HexColor("#F5A623")
        C_ROUGE   = rl_colors.HexColor("#DC2626")
        C_BLEU    = rl_colors.HexColor("#1A4FA0")
        C_GRIS    = rl_colors.HexColor("#6B7280")
        C_DARK    = rl_colors.HexColor("#1A1A1A")
        C_VERTBG  = rl_colors.HexColor("#E8F5EE")
        urgence_clr_pdf = (C_ROUGE if delai_median<=6 else
                           rl_colors.HexColor("#F97316") if delai_median<=18 else
                           C_OR if delai_median<=36 else C_VERT_C)

        sty = getSampleStyleSheet()
        s_title  = ParagraphStyle("t",  fontName="Helvetica-Bold",   fontSize=18, textColor=rl_colors.white,    leading=22)
        s_sub    = ParagraphStyle("sb", fontName="Helvetica",        fontSize=9,  textColor=rl_colors.HexColor("#CCCCCC"))
        s_h1     = ParagraphStyle("h1", fontName="Helvetica-Bold",   fontSize=13, textColor=C_VERT,    spaceBefore=12, spaceAfter=4)
        s_body   = ParagraphStyle("bo", fontName="Helvetica",        fontSize=9,  textColor=C_DARK,    spaceAfter=4,  leading=13)
        s_small  = ParagraphStyle("sm", fontName="Helvetica",        fontSize=8,  textColor=C_GRIS,    spaceAfter=2,  leading=11)
        s_center = ParagraphStyle("ce", fontName="Helvetica",        fontSize=9,  textColor=C_DARK,    alignment=TA_CENTER)
        s_kv     = ParagraphStyle("kv", fontName="Helvetica-Bold",   fontSize=15, alignment=TA_CENTER, leading=18)
        s_kl     = ParagraphStyle("kl", fontName="Helvetica",        fontSize=7.5,textColor=C_GRIS,    alignment=TA_CENTER)
        s_ok     = ParagraphStyle("ok", fontName="Helvetica",        fontSize=9,  textColor=rl_colors.HexColor("#065F46"), leading=12)
        s_warn   = ParagraphStyle("wn", fontName="Helvetica",        fontSize=9,  textColor=rl_colors.HexColor("#92400E"), leading=12)
        s_err    = ParagraphStyle("er", fontName="Helvetica",        fontSize=9,  textColor=rl_colors.HexColor("#991B1B"), leading=12)

        story = []

        # ── Bandeau titre ──
        hdr = Table([[
            Paragraph("<b>AMEN BANK</b>", s_title),
            Paragraph("Rapport — Prédiction du Délai avant Défaut", s_title),
            Paragraph("Direction des Risques", s_sub),
        ]], colWidths=[4*cm, 9*cm, 4*cm])
        hdr.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), C_VERT),
            ("TOPPADDING",(0,0),(-1,-1),14),("BOTTOMPADDING",(0,0),(-1,-1),14),
            ("LEFTPADDING",(0,0),(-1,-1),12),("LINEBELOW",(0,0),(-1,0),3,C_OR),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ]))
        story.append(hdr); story.append(Spacer(1,8))

        # ── Méta ──
        meta_tbl = Table([[
            Paragraph(f"<b>Analyste :</b> {USERS[st.session_state.username]['name']}", s_small),
            Paragraph(f"<b>Rôle :</b> {USERS[st.session_state.username]['role']}", s_small),
            Paragraph(f"<b>Date :</b> {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", s_small),
            Paragraph("<b>Système :</b> Amen Bank Risque Crédit v5.0", s_small),
        ]], colWidths=[4.25*cm]*4)
        meta_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), C_VERTBG),
            ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
            ("LEFTPADDING",(0,0),(-1,-1),8),("BOX",(0,0),(-1,-1),0.5,C_VERT_C),
        ]))
        story.append(meta_tbl); story.append(Spacer(1,10))

        # ── Niveau urgence ──
        alert_bg  = (rl_colors.HexColor("#FEE2E2") if delai_median<=6 else
                     rl_colors.HexColor("#FEF3C7") if delai_median<=18 else
                     rl_colors.HexColor("#FFFBEB") if delai_median<=36 else
                     rl_colors.HexColor("#D1FAE5"))
        alert_txt = (s_err if delai_median<=6 else s_warn if delai_median<=18 else s_body)
        urg_tbl = Table([[Paragraph(
            f"<b>Niveau de risque : {urgence}  —  Délai médian estimé : {delai_median} mois</b>",
            alert_txt)]], colWidths=[17*cm])
        urg_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1), alert_bg),
            ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
            ("LEFTPADDING",(0,0),(-1,-1),12),
            ("LINEBEFORE",(0,0),(0,-1),4, urgence_clr_pdf),
            ("BOX",(0,0),(-1,-1),0.5, urgence_clr_pdf),
        ]))
        story.append(urg_tbl); story.append(Spacer(1,10))

        # ── KPIs ──
        story.append(Paragraph("Résultats de la Prédiction", s_h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_OR, spaceAfter=6))
        kpi_items = [
            (str(delai_median)+" mois", "Délai Médian",   urgence_clr_pdf),
            (f"{proba_bad*100:.1f}%",   "PD Globale",     C_ROUGE if proba_bad>0.5 else C_OR),
            (f"{survie[min(mois_ecoules,len(survie)-1)]*100:.1f}%","Survie S(t)", C_VERT_C),
            (str(mois_restants)+" mois","Durée Restante",  C_BLEU),
            (f"{delai_pessimiste} / {delai_median} / {delai_optimiste}", "Optimiste/Médian/Pessimiste", C_GRIS),
        ]
        kpi_row  = [[
            [Paragraph(f"<b>{v}</b>", ParagraphStyle("kv2",fontName="Helvetica-Bold",
                fontSize=14,textColor=c,alignment=TA_CENTER,leading=18)),
             Paragraph(l, s_kl)]
            for v,l,c in kpi_items
        ]]
        kpi_tbl = Table(kpi_row, colWidths=[3.4*cm]*5)
        kpi_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),rl_colors.white),
            ("BOX",(0,0),(-1,-1),0.5,rl_colors.HexColor("#E5E7EB")),
            ("GRID",(0,0),(-1,-1),0.3,rl_colors.HexColor("#F3F4F6")),
            ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ]))
        story.append(kpi_tbl); story.append(Spacer(1,10))

        # ── Profil client ──
        story.append(Paragraph("Profil du Client", s_h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_OR, spaceAfter=6))
        profil_data = [
            [Paragraph("<b>Paramètre</b>", ParagraphStyle("th",fontName="Helvetica-Bold",
                fontSize=8,textColor=rl_colors.white,alignment=TA_CENTER)),
             Paragraph("<b>Valeur</b>", ParagraphStyle("th2",fontName="Helvetica-Bold",
                fontSize=8,textColor=rl_colors.white,alignment=TA_CENTER))],
            *[[Paragraph(k,s_body), Paragraph(str(v),s_body)] for k,v in [
                ("Âge", f"{age} ans"), ("Sexe", sex), ("Emploi", emploi_lbl),
                ("Logement", housing), ("Compte Épargne", saving),
                ("Compte Courant", checking), ("Montant Crédit", f"{credit:,} TND"),
                ("Durée contractuelle", f"{duration} mois"),
                ("Mois déjà écoulés", str(mois_ecoules)),
                ("Mois restants au contrat", str(mois_restants)),
                ("Objet du crédit", purpose),
            ]]
        ]
        profil_tbl = Table(profil_data, colWidths=[6*cm, 11*cm], repeatRows=1)
        profil_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), C_VERT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#F9FAFB")]),
            ("GRID",(0,0),(-1,-1),0.3,rl_colors.HexColor("#E5E7EB")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(-1,-1),8),
            ("LINEBELOW",(0,0),(-1,0),1.5,C_OR),
        ]))
        story.append(profil_tbl); story.append(Spacer(1,10))

        # ── Courbe de survie (tableau) ──
        story.append(Paragraph("Évolution de la Probabilité de Survie", s_h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_OR, spaceAfter=6))
        surv_hdrs = ["Mois","Survie S(t)","Risque h(t)","Défaut Cumulé","Statut"]
        surv_rows = [[
            Paragraph(f"<b>{k}</b>", ParagraphStyle("th3",fontName="Helvetica-Bold",
                fontSize=8,textColor=rl_colors.white,alignment=TA_CENTER))
            for k in surv_hdrs]]
        for i_s, t_s in enumerate(t_range):
            if t_s % 6 == 0 or t_s == duration:
                sv = survie[i_s]
                ht_s = hazards[i_s-1]*100 if i_s > 0 else 0
                stat = ("Défaut probable" if sv<0.5 else
                        "Surveillance"   if sv<0.75 else
                        "Attention"      if sv<0.90 else "Sain")
                surv_rows.append([
                    Paragraph(str(int(t_s)), s_center),
                    Paragraph(f"{sv*100:.1f}%", s_center),
                    Paragraph(f"{ht_s:.2f}%",   s_center),
                    Paragraph(f"{(1-sv)*100:.1f}%", s_center),
                    Paragraph(stat, s_center),
                ])
        surv_tbl = Table(surv_rows, colWidths=[2.5*cm,3.5*cm,3.5*cm,3.5*cm,4*cm], repeatRows=1)
        surv_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), C_VERT),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.white, rl_colors.HexColor("#F9FAFB")]),
            ("GRID",(0,0),(-1,-1),0.3,rl_colors.HexColor("#E5E7EB")),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LINEBELOW",(0,0),(-1,0),1.5,C_OR),
        ]))
        story.append(surv_tbl); story.append(Spacer(1,10))

        # ── Plan d'action ──
        story.append(Paragraph("Plan d'Action Préventif Amen Bank", s_h1))
        story.append(HRFlowable(width="100%", thickness=2, color=C_OR, spaceAfter=6))
        for titre_a, detail_a in actions:
            act_tbl = Table([[Paragraph(f"<b>{titre_a}</b><br/>{detail_a}", s_body)]],
                            colWidths=[17*cm])
            act_tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,-1), alert_bg),
                ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("LEFTPADDING",(0,0),(-1,-1),12),
                ("LINEBEFORE",(0,0),(0,-1),3, urgence_clr_pdf),
                ("BOX",(0,0),(-1,-1),0.5,rl_colors.HexColor("#E5E7EB")),
            ]))
            story.append(act_tbl); story.append(Spacer(1,4))

        # ── Pied de page ──
        story.append(Spacer(1,14))
        footer = Table([[
            Paragraph("© 2025 Amen Bank Tunisie — Direction des Risques — Document Confidentiel",
                ParagraphStyle("ft",fontName="Helvetica",fontSize=7.5,textColor=C_GRIS,alignment=TA_LEFT)),
            Paragraph(f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')}",
                ParagraphStyle("ftd",fontName="Helvetica",fontSize=7.5,textColor=C_GRIS,alignment=TA_RIGHT)),
        ]], colWidths=[10*cm, 7*cm])
        footer.setStyle(TableStyle([("LINEABOVE",(0,0),(-1,0),0.5,C_VERT_C),("TOPPADDING",(0,0),(-1,-1),4)]))
        story.append(footer)

        doc.build(story)
        buf.seek(0)
        pdf_bytes = buf.read()

        st.download_button(
            label="📥  Télécharger le Rapport PDF",
            data=pdf_bytes,
            file_name=f"amen_bank_delai_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
            key="dl_rapport_delai_pdf",
            use_container_width=False,
        )
        st.success("✅ Rapport PDF prêt — cliquez sur le bouton ci-dessus pour télécharger.")

    except Exception as e_pdf:
        # Fallback TXT si reportlab manque
        rapport_txt = f"""AMEN BANK — RAPPORT DÉLAI DE DÉFAUT
=========================================
Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
Analyste : {USERS[st.session_state.username]['name']}
Rôle     : {USERS[st.session_state.username]['role']}

PROFIL
  Âge              : {age} ans
  Sexe             : {sex}
  Emploi           : {emploi_lbl}
  Logement         : {housing}
  Épargne          : {saving}
  Compte courant   : {checking}
  Montant crédit   : {credit:,} TND
  Durée            : {duration} mois
  Objet            : {purpose}
  Mois écoulés     : {mois_ecoules}

RÉSULTATS
  PD Globale          : {proba_bad*100:.2f}%
  Délai médian        : {delai_median} mois
  Délai optimiste     : {delai_pessimiste} mois
  Délai pessimiste    : {delai_optimiste} mois
  Niveau urgence      : {urgence}
  Survie actuelle     : {survie[min(mois_ecoules,len(survie)-1)]*100:.1f}%

PLAN D'ACTION
{"".join([f"  - {t_}: {d_}" + chr(10) for t_,d_ in actions])}
Direction des Risques — Amen Bank Tunisie · v5.0
"""
        st.download_button(
            label="📥  Télécharger le Rapport (TXT)",
            data=rapport_txt.encode("utf-8"),
            file_name=f"amen_bank_delai_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            key="dl_rapport_delai_txt",
        )
        st.info(f"⚠️ PDF non disponible ({e_pdf}) — rapport TXT généré à la place. Installez : pip install reportlab")

# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        page_login()
        return
    df   = load_data()
    page = render_sidebar()
    if   "Tableau de Bord"       in page: page_dashboard(df)
    elif "Analyse Exploratoire"  in page: page_eda(df)
    elif "Modèles"               in page: page_models(df)
    elif "Prédiction"            in page: page_prediction(df)
    elif "Délai"                 in page: page_delai_defaut(df)
    elif "Données"               in page: page_data(df)
    st.markdown(f"""
    <div class="footer">
      © 2025 <span>Amen Bank Tunisie</span> — Tous droits réservés &nbsp;|&nbsp;
      Direction des Risques &nbsp;·&nbsp; Système Risque Crédit <b>v5.0 Professional</b>
      &nbsp;·&nbsp; XGBoost · LightGBM · Ensemble Voting · Survie · Délai Défaut
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

