from flask import Flask, render_template_string, request, send_file
from waitress import serve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
import base64
from io import BytesIO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------- Helper ----------
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


# ---------- ML Pipeline ----------
def train_and_visualize(features_path, classes_path):
    features = pd.read_csv(features_path, index_col=0)
    classes = pd.read_csv(classes_path, index_col=0)

    df = features.merge(classes, left_index=True, right_index=True, how="left")
    df = df[df['class'].notnull()]

    mapping = {'1': 1, '2': 0, 1: 1, 2: 0, 'Illicit': 1, 'Licit': 0}
    df['label'] = df['class'].map(mapping)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    drop_cols = ['class', 'label']
    if 'time_step' in df.columns:
        drop_cols.append('time_step')
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_res_scaled, y_res)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    cm = confusion_matrix(y_test, y_pred)

    pred_df = pd.DataFrame({
        "transaction_id": X_test.index,
        "predicted_label": y_pred
    })
    pred_df["prediction_type"] = pred_df["predicted_label"].map({0: "Normal", 1: "Fraudulent"})
    fraud_txns = pred_df[pred_df["predicted_label"] == 1]
    normal_txns = pred_df[pred_df["predicted_label"] == 0]

    fraud_path = os.path.join(UPLOAD_FOLDER, "fraudulent_transactions.csv")
    normal_path = os.path.join(UPLOAD_FOLDER, "normal_transactions.csv")
    fraud_txns.to_csv(fraud_path, index=False)
    normal_txns.to_csv(normal_path, index=False)

    # Graphs
    fig1, ax1 = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    cm_img = fig_to_base64(fig1)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig2, ax2 = plt.subplots(figsize=(4,3))
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0,1],[0,1],'k--')
    ax2.legend()
    ax2.set_title("ROC Curve")
    roc_img = fig_to_base64(fig2)

    fig3, ax3 = plt.subplots(figsize=(4,3))
    sns.countplot(x=y, ax=ax3)
    ax3.set_title("Class Distribution (Before SMOTE)")
    dist_img = fig_to_base64(fig3)

    return {
        "fraud_count": len(fraud_txns),
        "normal_count": len(normal_txns),
        "total": len(pred_df),
        "acc": round(acc, 4),
        "prec": round(prec, 4),
        "rec": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "cm_img": cm_img,
        "roc_img": roc_img,
        "dist_img": dist_img,
        "fraud_path": fraud_path,
        "normal_path": normal_path
    }


# ---------- Flask Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        features_file = request.files["features"]
        classes_file = request.files["classes"]

        f_path = os.path.join(UPLOAD_FOLDER, features_file.filename)
        c_path = os.path.join(UPLOAD_FOLDER, classes_file.filename)
        features_file.save(f_path)
        classes_file.save(c_path)

        results = train_and_visualize(f_path, c_path)

        html = """
        <html>
        <head>
            <title>Blockchain Fraud Detection Dashboard</title>
            <style>
                body { font-family: Arial; background: #f4f6fc; margin: 20px; }
                .container { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 0 10px #ccc; }
                h1 { background: #007BFF; color: white; padding: 10px; border-radius: 8px; text-align: center; }
                .metrics { display: flex; justify-content: space-around; margin-top: 20px; flex-wrap: wrap; }
                .card { background: #e9f1ff; padding: 15px; border-radius: 8px; text-align: center; width: 18%; margin: 10px; }
                img { width: 320px; border-radius: 10px; box-shadow: 0 0 6px #bbb; margin: 10px; }
                .graph-container { display: flex; justify-content: space-around; flex-wrap: wrap; }
                .download { text-align: center; margin-top: 20px; }
                a { background: #007BFF; color: white; padding: 10px 15px; text-decoration: none; border-radius: 6px; margin: 5px; }
                a:hover { background: #0056b3; }
                h2 { text-align: center; background: #007BFF; color: white; padding: 8px; border-radius: 6px; margin-top: 40px; }
                table { width: 60%; margin: 0 auto; border-collapse: collapse; background: #eef4ff; }
                th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }
                th { background: #007BFF; color: white; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1> Blockchain Fraud Detection Dashboard</h1>
                
                <div class="metrics">
                    <div class="card"><b>Total Transactions</b><br>{{total}}</div>
                    <div class="card"><b>Fraudulent</b><br>{{fraud_count}}</div>
                    <div class="card"><b>Normal</b><br>{{normal_count}}</div>
                    <div class="card"><b>Accuracy</b><br>{{acc}}</div>
                    <div class="card"><b>ROC-AUC</b><br>{{roc_auc}}</div>
                </div>

                <div class="graph-container">
                    <div><h3>Confusion Matrix</h3><img src="data:image/png;base64,{{cm_img}}"></div>
                    <div><h3>ROC Curve</h3><img src="data:image/png;base64,{{roc_img}}"></div>
                    <div><h3>Class Distribution</h3><img src="data:image/png;base64,{{dist_img}}"></div>
                </div>

                <h2>📊 Perfomance Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Score</th></tr>
                    <tr><td>Accuracy</td><td>{{acc}}</td></tr>
                    <tr><td>Precision</td><td>{{prec}}</td></tr>
                    <tr><td>Recall</td><td>{{rec}}</td></tr>
                    <tr><td>F1-Score</td><td>{{f1}}</td></tr>
                    <tr><td>ROC-AUC</td><td>{{roc_auc}}</td></tr>
                </table>

                <div class="download">
                    <a href="/download/fraud">Download Fraudulent CSV</a>
                    <a href="/download/normal">Download Normal CSV</a>
                    <br><br>
                    <a href="/">🔁 Upload New Dataset</a>
                </div>
            </div>
        </body>
        </html>
        """
        return render_template_string(html, **results)

    # Upload page
    upload_html = """
    <html>
    <head>
        <title>Upload Datasets - Blockchain Fraud Detection</title>
        <style>
            body { font-family: Arial; background: #eef3fa; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .upload-box { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 12px #bbb; text-align: center; }
            input { margin: 10px; padding: 10px; }
            button { background: #007BFF; color: white; border: none; padding: 10px 15px; border-radius: 6px; cursor: pointer; }
            button:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="upload-box">
            <h1> Upload Dataset Files</h1>
            <form method="POST" enctype="multipart/form-data">
                <p><b>Features CSV:</b> <input type="file" name="features" required></p>
                <p><b>Classes CSV:</b> <input type="file" name="classes" required></p>
                <button type="submit">Start Fraud Detection</button>
            </form>
        </div>
    </body>
    </html>
    """
    return upload_html


@app.route("/download/<string:which>")
def download(which):
    if which == "fraud":
        return send_file(os.path.join(UPLOAD_FOLDER, "fraudulent_transactions.csv"), as_attachment=True)
    else:
        return send_file(os.path.join(UPLOAD_FOLDER, "normal_transactions.csv"), as_attachment=True)


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5000)
