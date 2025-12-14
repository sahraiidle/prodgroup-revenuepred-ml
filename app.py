from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import joblib
import os
from forms import ProductPredictForm
from utils import prepare_features_from_json, prepare_regression_features_from_json, signed_expm1
from business_interpretation import KMEANS_BUSINESS, DBSCAN_BUSINESS

app = Flask(__name__)
CORS(app)
app.secret_key = "dev-secret-key-please-change"
app.config.setdefault('WTF_CSRF_ENABLED', False)

# Models loaded from disk
MODELS = {
    "kmeans": joblib.load("models/kmeans_model.joblib"),
    "dbscan": joblib.load("models/dbscan_model.joblib"),
    "random_forest": joblib.load("models/random_forest_regressor.joblib"),
    "xgboost": joblib.load("models/xgboost_regressor.joblib"),
}


@app.route("/", methods=["GET"])
def home():
    return redirect(url_for('ui_predict'))


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "message": "Product Group & Revenue Prediction API is running.",
        "endpoints": {
            "POST /predict_group?model=kmeans|dbscan": {
                "expect_json": {
                    "NetRevenue": "float",
                    "NetQuantity": "float",
                    "NumTransactions": "int",
                    "NumUniqueCustomers": "int"
                }
            },
            "POST /predict_revenue?model=random_forest|xgboost": {
                "expect_json": {
                    'NetRevenue': 'float',
                    'NetRevenue_LastMonth': 'float',
                    'NetRevenue_MA3': 'float',
                    'Month': 'int',
                    'ProductFrequency': 'int'
                }
            },
            "POST /predict_all?group_model=kmeans|dbscan&rev_model=random_forest|xgboost": {
                "expect_json": {
                    "NetRevenue": "float",
                    "NetQuantity": "float",
                    "NumTransactions": "int",
                    "NumUniqueCustomers": "int",
                    'NetRevenue_LastMonth': 'float',
                    'NetRevenue_MA3': 'float',
                    'Month': 'int',
                    'ProductFrequency': 'int'
                }
            }
        }
    })


@app.route("/predict_group", methods=["POST"])
def predict_group():
    choice = request.args.get("model", "kmeans")
    if choice not in MODELS:
        return jsonify({"error": "Invalid model. Use model='kmeans' or model='dbscan'."}), 400

    model = MODELS[choice]
    data = request.get_json() or {}
    required = {"NetQuantity", "NetRevenue", "NumTransactions", "NumUniqueCustomers"}
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        X_scaled = prepare_features_from_json(data)
        if choice == "kmeans":
            cluster = int(model.predict(X_scaled)[0])
            business = KMEANS_BUSINESS.get(cluster, {"cluster_name": "unknown product cluster", "description": ""})
            model_used = "KMeans"
        else:
            cluster = int(model.fit_predict(X_scaled)[0])
            business = DBSCAN_BUSINESS.get(cluster, {"cluster_name": "unknown product cluster", "description": ""})
            model_used = "DBSCAN"
    except Exception as e:
        return jsonify({"error": f"Failed to prepare/predict: {str(e)}"}), 500

    return jsonify({
        "model_used": model_used,
        "predicted_group": int(cluster),
        "group_info": business,
        "input_data": data
    })


@app.route("/predict_revenue", methods=["POST"])
def predict_revenue():
    choice = request.args.get("model", "xgboost")
    if choice not in ("random_forest", "xgboost"):
        return jsonify({"error": "Invalid model. Use model='xgboost' or model='random_forest'."}), 400
    model = MODELS[choice]
    data = request.get_json() or {}
    required = ['NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3', 'Month', 'ProductFrequency']
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    try:
        X_scaled = prepare_regression_features_from_json(data)
        pred_revenue = float(model.predict(X_scaled)[0])
        try:
            pred_revenue = float(signed_expm1(pred_revenue))
        except Exception:
            pass
        pred_formatted = f"${pred_revenue:,.2f}"
    except Exception as e:
        return jsonify({"error": f"Failed to prepare/predict: {str(e)}"}), 500
    return jsonify({
        "model_used": "XGBoost" if choice == "xgboost" else "Random Forest",
        "input_data": data,
        "next_month_revenue": pred_formatted,
    })


@app.route("/predict_all", methods=["POST"])
def predict_all():
    group_choice = request.args.get("group_model", "kmeans")
    rev_choice = request.args.get("rev_model", "xgboost")

    if group_choice not in ("kmeans", "dbscan"):
        return jsonify({"error": "Invalid group_model. Use group_model='kmeans' or 'dbscan'."}), 400
    if rev_choice not in ("random_forest", "xgboost"):
        return jsonify({"error": "Invalid rev_model. Use rev_model='xgboost' or 'random_forest'."}), 400

    data = request.get_json() or {}
    required = {"NetRevenue", "NetQuantity", "NumTransactions", "NumUniqueCustomers",
                "NetRevenue_LastMonth", "NetRevenue_MA3", "Month", "ProductFrequency"}
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        Xc = prepare_features_from_json(data)
        if group_choice == "kmeans":
            Group = int(MODELS["kmeans"].predict(Xc)[0])
            Group_info = KMEANS_BUSINESS.get(Group, {"cluster_name": "unknown product cluster", "description": "", "recommended_action": ""})
            group_model = "KMeans"
        else:
            Group = int(MODELS["dbscan"].fit_predict(Xc)[0])
            Group_info = DBSCAN_BUSINESS.get(Group, {"cluster_name": "unknown product cluster", "description": "", "recommended_action": ""})
            group_model = "DBSCAN"

        Xr = prepare_regression_features_from_json(data)
        rev_model = MODELS[rev_choice]
        pred = float(rev_model.predict(Xr)[0])
        try:
            pred = float(signed_expm1(pred))
        except Exception:
            pass
        pred_formatted = f"${pred:,.2f}"
        model_name = "XGBoost" if rev_choice == "xgboost" else "Random Forest"
    except Exception as e:
        return jsonify({"error": f"Failed to prepare/predict: {str(e)}"}), 500

    resp = {
       
        "product_group": Group_info.get("cluster_name"),
        "business": Group_info.get("description"),
        "recommended_action": Group_info.get("recommended_action"),
        "next_month_revenue": pred_formatted,
      
        "Models": [group_model, model_name],
        "Product Group": Group_info.get("cluster_name"),
        "Description": Group_info.get("description"),
        "Recommendation": Group_info.get("recommended_action"),
        "Next Month Revenue": pred_formatted,

        "cluster_name": Group_info.get("cluster_name"),
        "description": Group_info.get("description"),
        "next_month_revenue_formatted": pred_formatted
    }

    if request.args.get('debug') in ('1', 'true', 'True'):
        try:
            features = ['NetRevenue', 'NetRevenue_LastMonth', 'NetRevenue_MA3', 'Month', 'ProductFrequency']
            raw_vals = {f: data.get(f) for f in features}
            scaled = prepare_regression_features_from_json(data).tolist()
            resp['debug'] = {'raw_regression_input': raw_vals, 'scaled_regression_input': scaled, 'raw_prediction': pred}
        except Exception as e:
            resp['debug_error'] = str(e)

    return jsonify(resp)


@app.route("/ui", methods=["GET", "POST"])
def ui_predict():
    form = ProductPredictForm()
    group_models = [("kmeans", "Predictor-1 (KMeans)"), ("dbscan", "Predictor-2 (DBSCAN)")]
    rev_models = [("xgboost", "Predictor-1 (XGBoost)"), ("random_forest", "Predictor-2 (Random Forest)")]
    return render_template("forms.html", form=form, group_models=group_models, rev_models=rev_models)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7000))
    app.run(host="0.0.0.0", port=port)
