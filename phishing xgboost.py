import numpy as np
import pandas as pd
import re
import joblib
import matplotlib.pyplot as plt
from urllib.parse import urlparse
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
df = pd.read_csv("C:\\Users\\Shahzadi\\Desktop\\project\\dataset1.csv")
corr = df.corr(numeric_only=True)
corr['Result'].sort_values(ascending=False).plot(kind='bar', figsize=(14,5))
plt.title("Correlation with Result")
plt.tight_layout()
plt.show()

# ── STEP 3: Drop weak columns ─────────────────────────────
df.drop(columns=['id','Abnormal_URL','HTTPS_token','Favicon','Iframe',
                 'popUpWidnow','Page_Rank','RightClick','on_mouseover'],
        inplace=True, errors='ignore')

# ── STEP 4: Convert labels — XGBoost needs 0 and 1 ───────
df['Result'] = df['Result'].replace(-1, 0)

X = df.drop('Result', axis=1)
y = df['Result']
print("X shape:", X.shape)
print("Label counts:\n", y.value_counts())
print("Feature columns:", X.columns.tolist())

# ── STEP 5: Train test split ──────────────────────────────
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── STEP 6: Train XGBoost only ────────────────────────────
# FIX: removed useless RandomForest that was wasting time
model = XGBClassifier(
    n_estimators=100,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)
model.fit(x_train, y_train)
print("\nXGBoost trained successfully!")

# ── STEP 7: Evaluate ──────────────────────────────────────
# FIX: y_train_pred instead of y_train — do NOT overwrite labels!
y_train_pred = model.predict(x_train)
y_pred       = model.predict(x_test)

print("\n" + "="*55)
print("         MODEL EVALUATION — XGBOOST")
print("="*55)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  Precision : {precision_score(y_test, y_pred)*100:.2f}%")
print(f"  Recall    : {recall_score(y_test, y_pred)*100:.2f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred)*100:.2f}%")
print("="*55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Phishing','Legitimate']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── STEP 8: Save model ────────────────────────────────────
joblib.dump(model, 'phishing_model.pkl')
print("\nModel saved as phishing_model.pkl")

# ── STEP 9: Shorteners list ───────────────────────────────
SHORTENERS = [
    'bit.ly','goo.gl','tinyurl.com','ow.ly','t.co',
    'is.gd','buff.ly','adf.ly','short.link','rebrand.ly',
    'cutt.ly','rb.gy'
]

# ── STEP 10: Feature extraction — ALL BUGS FIXED ─────────
def extract_features(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.lower().replace('www.', '')
    path   = parsed.path

    is_shortener = any(
        s == netloc or netloc.endswith('.' + s)
        for s in SHORTENERS
    )

    return {
        # FIX 1: IP address — having IP = SUSPICIOUS (-1)
        # was backwards before — now correct
        'having_IP_Address'           : -1 if re.match(
                                            r'\d+\.\d+\.\d+\.\d+',
                                            netloc) else 1,

        'URL_Length'                  : 1 if len(url) < 54 else (
                                        -1 if len(url) > 75 else 0),

        'Shortining_Service'          : -1 if is_shortener else 1,

        'having_At_Symbol'            : -1 if '@' in url else 1,

        # FIX 2: check full URL after http:// not just path
        'double_slash_redirecting'    : -1 if '//' in url[7:] else 1,

        'Prefix_Suffix'               : -1 if '-' in netloc else 1,

        'having_Sub_Domain'           : -1 if netloc.count('.') > 2
                                        else (0 if netloc.count('.') == 2
                                        else 1),

        'SSLfinal_State'              : 1 if parsed.scheme == 'https'
                                        else -1,

        # FIX 3: short netloc = newly registered suspicious domain
        'Domain_registeration_length' : -1 if len(netloc) < 12 else 1,

        'port'                        : -1 if ':' in netloc else 1,

        # FIX 4: changed hardcoded 1s to 0 (neutral/unknown)
        # these features need webpage crawling to get real values
        # hardcoding 1 was pushing everything toward LEGITIMATE
        'Request_URL'                 : 0,
        'URL_of_Anchor'               : 0,
        'Links_in_tags'               : 0,
        'SFH'                         : 0,

        'Submitting_to_email'         : -1 if 'mailto:' in url else 1,
        'Redirect'                    : 0,
        'age_of_domain'               : 0,
        'DNSRecord'                   : 0,
        'web_traffic'                 : 0,

        # FIX 5: Google Index unknown without crawling — set neutral
        'Google_Index'                : 0,

        'Links_pointing_to_page'      : 0,
        'Statistical_report'          : -1 if is_shortener else 1,
    }

# ── STEP 11: Predict function ─────────────────────────────
def predict_url(url):
    parsed   = urlparse(url)
    netloc   = parsed.netloc.lower().replace('www.', '')
    is_short = any(s == netloc or netloc.endswith('.' + s)
                   for s in SHORTENERS)

    features     = extract_features(url)
    feature_cols = X.columns.tolist()
    input_data   = pd.DataFrame([features])[feature_cols]
    prediction   = model.predict(input_data)[0]
    confidence   = model.predict_proba(input_data).max() * 100

    print("\n" + "="*55)
    print(f"  URL       : {url}")
    print("="*55)

    if is_short:
        print(f"  RESULT    : SUSPICIOUS — URL shortener detected!")
        print(f"  WARNING   : Real destination hidden by {netloc}")
        print(f"  ADVICE    : Expand this URL before visiting")

    elif confidence < 70:
        print(f"  RESULT    : SUSPICIOUS — Cannot determine safely")
        print(f"  WARNING   : Low confidence ({confidence:.1f}%)")

    elif prediction == 1:
        print(f"  RESULT    : LEGITIMATE WEBSITE")
        print(f"  CONFIDENCE: {confidence:.1f}%")

    else:
        print(f"  RESULT    : PHISHING / UNSAFE WEBSITE")
        print(f"  CONFIDENCE: {confidence:.1f}%")

    print("="*55)

    # Feature breakdown
    print("\n  Feature breakdown:")
    key_features = [
        'having_IP_Address', 'SSLfinal_State', 'URL_Length',
        'Shortining_Service', 'Prefix_Suffix', 'having_Sub_Domain',
        'having_At_Symbol', 'Domain_registeration_length',
        'double_slash_redirecting', 'port'
    ]
    for feat in key_features:
        val       = features[feat]
        meaning   = {1:"Safe", 0:"Neutral", -1:"Suspicious"}[val]
        indicator = {1:"✓", 0:"-", -1:"✗"}[val]
        print(f"  {indicator} {feat:<35} = {val:>2}  ({meaning})")
    print()

# ── STEP 12: Test with known URLs ─────────────────────────
print("\n--- BATCH TEST ---")
test_urls = [
    "http://paypal-secure-login-verify.com/account/update",
    "http://192.168.1.1/bank/login",
    "http://login.secure.verify.paypal.com.evil.ru",
    "https://www.google.com",
    "https://www.facebook.com",
    "https://www.amazon.com",
]
for url in test_urls:
    predict_url(url)

# ── STEP 13: User input ───────────────────────────────────
user_url = input("\nPaste any URL to check: ")
predict_url(user_url)