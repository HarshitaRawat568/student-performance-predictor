# ================================================================
# STUDENT PERFORMANCE PREDICTOR — Model Training Script
# Dataset: Student Performance (UCI ML Repository) ~395 rows
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings; warnings.filterwarnings('ignore')

# ── CREATE SYNTHETIC DATASET (mirrors UCI Student Performance) ──
np.random.seed(42)
n = 1000

study_hours   = np.random.normal(5, 2, n).clip(0, 12)
attendance    = np.random.normal(75, 15, n).clip(30, 100)
prev_score    = np.random.normal(60, 15, n).clip(20, 100)
sleep_hours   = np.random.normal(7, 1.5, n).clip(3, 10)
extra_classes = np.random.choice([0, 1], n, p=[0.4, 0.6])
parental_edu  = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])  # 0=none,1=school,2=grad

# Target: pass (1) if score >= 50, fail (0)
noise = np.random.normal(0, 8, n)
score = (0.35*study_hours*8 + 0.25*prev_score + 0.15*attendance*0.6
         + 0.1*sleep_hours*5 + 0.1*extra_classes*15 + 0.05*parental_edu*10 + noise)
target = (score >= 50).astype(int)

df = pd.DataFrame({
    'study_hours':   np.round(study_hours, 1),
    'attendance_pct': np.round(attendance, 1),
    'prev_score':    np.round(prev_score, 1),
    'sleep_hours':   np.round(sleep_hours, 1),
    'extra_classes': extra_classes,
    'parental_edu':  parental_edu,
    'pass':          target
})

df.to_csv('student_data.csv', index=False)
print(f"✅ Dataset created: {df.shape[0]} students, {df.shape[1]-1} features")
print(f"   Pass rate: {target.mean()*100:.1f}%\n")

# ── FEATURES & TARGET ──────────────────────────────────────────
X = df.drop('pass', axis=1)
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── TRAIN 3 MODELS ────────────────────────────────────────────
models = {
    'Logistic Regression':   LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':         RandomForestClassifier(n_estimators=150, random_state=42),
    'Gradient Boosting':     GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:,1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv      = cross_val_score(model, X_train_sc, y_train, cv=5).mean()
    results[name] = dict(model=model, y_pred=y_pred, y_proba=y_proba,
                         acc=acc, auc=auc, cv=cv)
    print(f"{'─'*40}")
    print(f"📌 {name}")
    print(f"   Accuracy : {acc*100:.2f}%  |  AUC: {auc:.4f}  |  CV: {cv*100:.2f}%")

# ── PLOTS ──────────────────────────────────────────────────────
os.makedirs('plots', exist_ok=True)

# 1. Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#4361EE', '#3A86FF', '#06D6A0']
names  = list(results.keys())

axes[0].bar(names, [results[m]['acc']*100 for m in names],
            color=colors, edgecolor='black', alpha=0.88)
axes[0].set_ylim(70, 100); axes[0].set_title('Accuracy Comparison', fontweight='bold')
axes[0].set_ylabel('Accuracy (%)'); axes[0].tick_params(axis='x', rotation=12)
for i, m in enumerate(names):
    axes[0].text(i, results[m]['acc']*100+0.3,
                 f"{results[m]['acc']*100:.1f}%", ha='center', fontweight='bold')

for m, c in zip(names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[m]['y_proba'])
    axes[1].plot(fpr, tpr, color=c, lw=2,
                 label=f"{m.split()[0]} (AUC={results[m]['auc']:.3f})")
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_title('ROC Curves', fontweight='bold')
axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

best = max(results, key=lambda m: results[m]['acc'])
cm = confusion_matrix(y_test, results[best]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
            xticklabels=['Fail','Pass'], yticklabels=['Fail','Pass'])
axes[2].set_title(f'Confusion Matrix\n({best})', fontweight='bold')
axes[2].set_ylabel('Actual'); axes[2].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('plots/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close(); print("\n✅ model_comparison.png saved")

# 2. Feature Importance
rf = results['Random Forest']['model']
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
feat_labels = {
    'study_hours':'Study Hours/Day',
    'attendance_pct':'Attendance %',
    'prev_score':'Previous Score',
    'sleep_hours':'Sleep Hours',
    'extra_classes':'Extra Classes',
    'parental_edu':'Parental Education'
}
plt.figure(figsize=(9,5))
bars = plt.barh([feat_labels[c] for c in imp.index], imp.values,
                color='#4361EE', edgecolor='black', alpha=0.85)
for bar, val in zip(bars, imp.values):
    plt.text(val+0.003, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=10)
plt.title('Feature Importance — Random Forest', fontweight='bold', fontsize=13)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ feature_importance.png saved")

# 3. EDA
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
feat_list = ['study_hours','attendance_pct','prev_score',
             'sleep_hours','extra_classes','parental_edu']
for ax, feat in zip(axes.flat, feat_list):
    ax.hist(df[df['pass']==0][feat], bins=20, alpha=0.6,
            label='Fail', color='#EF233C')
    ax.hist(df[df['pass']==1][feat], bins=20, alpha=0.6,
            label='Pass', color='#4361EE')
    ax.set_title(feat_labels[feat], fontweight='bold')
    ax.legend()
plt.suptitle('Feature Distributions by Outcome', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/eda.png', dpi=150, bbox_inches='tight')
plt.close(); print("✅ eda.png saved")

# ── SAVE MODEL & SCALER ────────────────────────────────────────
best_model = results[best]['model']
with open('model.pkl',  'wb') as f: pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)

print(f"\n🏆 Best Model : {best}")
print(f"   Accuracy   : {results[best]['acc']*100:.2f}%")
print(f"   ROC-AUC    : {results[best]['auc']:.4f}")
print(f"   CV Acc     : {results[best]['cv']*100:.2f}%")
print("\n✅ model.pkl & scaler.pkl saved — ready for web app!")
