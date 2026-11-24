"""SchemeMatch dashboard and JSON API. Run with python app.py."""
from functools import lru_cache
import os
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response
from plotly.offline import get_plotlyjs
from schemematch.charts import figures
from werkzeug.exceptions import RequestEntityTooLarge
from schemematch.data import prepare
from schemematch.model import analyze, assign
from schemematch.schemes import QUESTIONS, SCHEMES, screen

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
MODEL = Path(__file__).parent / "models" / "households.joblib"


def view(tab, **context):
    if "report" in context:
        context["plots"] = figures(context["report"])
    return render_template("dashboard.html", tab=tab, schemes=SCHEMES,
                           questions=QUESTIONS, available=MODEL.exists(), **context)


@lru_cache(maxsize=1)
def benchmark():
    data = pd.read_csv(Path(__file__).parent / "data" / "engel.csv")
    return analyze(data)["report"], {"received": len(data), "complete": len(data), "excluded": 0}


@app.get("/benchmark")
def benchmark_view():
    report, audit = benchmark()
    return view("benchmark", report=report, audit=audit)


@app.get("/assets/plotly.js")
def plotly_asset():
    return Response(get_plotlyjs(), mimetype="application/javascript",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/")
def home():
    return view("directory")


@app.route("/matcher", methods=["GET", "POST"])
def matcher():
    if request.method == "GET":
        return view("matcher")
    answers = {key: request.form.get(key) for key in QUESTIONS}
    if any(v not in ("yes", "no") for v in answers.values()):
        return view("matcher", error="Please answer every question.", answers=answers), 400
    matches = screen({key: value == "yes" for key, value in answers.items()})
    return view("matcher", matches=matches, answers=answers)


def upload():
    file = request.files.get("survey")
    if file is None:
        raise ValueError("Choose a CSV to analyze.")
    frame = pd.read_csv(file, nrows=2001)
    if len(frame) > 2000:
        raise ValueError("The web lab supports 2,000 rows. Use train.py for larger surveys.")
    data, audit = prepare(frame)
    return analyze(data)["report"], audit


@app.route("/lab", methods=["GET", "POST"])
def lab():
    if request.method == "POST":
        report, audit = upload()
        return view("lab", report=report, audit=audit)
    return view("lab")


@app.get("/model")
def model_view():
    if not MODEL.exists():
        return view("model")
    bundle = joblib.load(MODEL)
    return view("model", report=bundle["report"], audit=bundle["audit"])


@app.post("/api/analyze")
def analyze_api():
    report, audit = upload()
    return jsonify(report=report, audit=audit)


@app.post("/api/match")
def match_api():
    return jsonify(matches=screen(request.get_json(silent=True)))


@app.post("/api/predict")
def predict_api():
    if not MODEL.exists():
        return jsonify(error="Train a model first using train.py."), 409
    row = request.get_json(silent=True)
    if not isinstance(row, dict) or any(isinstance(v, (list, dict, bool)) for v in row.values()):
        raise ValueError("Send one numeric household as a JSON object.")
    data, _ = prepare(pd.DataFrame([row]))
    return jsonify(assign(joblib.load(MODEL), data))


@app.errorhandler(ValueError)
@app.errorhandler(RequestEntityTooLarge)
def invalid(error):
    status = 413 if isinstance(error, RequestEntityTooLarge) else 400
    message = "The upload exceeds 2 MB." if status == 413 else str(error)
    if request.path.startswith("/api/"):
        return jsonify(error=message), status
    return view("lab", error=message), status


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 7860)))
