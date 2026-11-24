"""Compare cluster models on a shared validation split, then refit the winner."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer

SEED = 42


def estimator(name, k):
    if name == "K-means":
        return KMeans(n_clusters=k, n_init=10, random_state=SEED)
    return GaussianMixture(n_components=k, covariance_type="diag", n_init=3,
                           reg_covar=1e-4, random_state=SEED)


def analyze(data, max_k=6):
    if not 40 <= len(data) <= 100_000:
        raise ValueError("Use 40 to 100,000 complete households for training.")
    train, validation = train_test_split(data, test_size=0.25, random_state=SEED)
    # Fit all transformations on training households before validation.
    income = StandardScaler()
    income_column = data.columns[0]
    if train[income_column].nunique() > 2:
        income = PowerTransformer(method="yeo-johnson")
    preprocess = ColumnTransformer([
        ("income", income, [income_column]),
        ("other", StandardScaler(), list(data.columns[1:]))
    ])
    train_x = preprocess.fit_transform(train)
    valid_x = preprocess.transform(validation.sample(n=min(800, len(validation)), random_state=SEED))
    trials = []
    distinct = len(np.unique(train_x, axis=0))
    for k in range(2, min(max_k, distinct) + 1):
        for name in ("K-means", "Gaussian mixture"):
            model = estimator(name, k).fit(train_x)
            labels = model.predict(valid_x)
            if len(np.unique(labels)) != k or k >= len(labels):
                continue
            trials.append({"algorithm": name, "k": k,
                           "silhouette": float(silhouette_score(valid_x, labels)),
                           "davies_bouldin": float(davies_bouldin_score(valid_x, labels))})
    if not trials:
        raise ValueError("Too little variation for a supported validation result. Try a broader cohort.")
    trials.sort(key=lambda t: (-t["silhouette"], t["davies_bouldin"], t["k"]))
    best = trials[0]
    pipeline = make_pipeline(clone(preprocess), estimator(best["algorithm"], best["k"]))
    pipeline.fit(data)
    labels = pipeline.predict(data)
    grouped = data.assign(group=labels + 1).groupby("group")
    profiles = grouped.mean().round(2)
    profiles.insert(0, "households", grouped.size())
    transformed = pipeline[0].transform(data)
    pca = PCA(n_components=2).fit(transformed)
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(data), min(500, len(data)), replace=False)
    points = pca.transform(transformed[indices])
    report = {"trials": trials, "profiles": profiles.reset_index().to_dict("records"),
              "features": list(data.columns), "training_rows": len(train),
              "validation_rows_scored": len(valid_x),
              "variance": pca.explained_variance_ratio_.tolist(),
              "points": points.tolist(), "point_groups": (labels[indices] + 1).tolist()}
    return {"pipeline": pipeline, "features": list(data.columns), "report": report}


def assign(bundle, data):
    if list(data.columns) != bundle["features"] or len(data) != 1:
        raise ValueError("Supply one complete household using the trained model's feature set.")
    pipeline = bundle["pipeline"]
    group = int(pipeline.predict(data)[0])
    model = pipeline[-1]
    centers = model.cluster_centers_ if hasattr(model, "cluster_centers_") else model.means_
    distance = cdist(pipeline[0].transform(data), centers)[0, group]
    return {"group": group + 1, "distance_to_center": round(float(distance), 3)}
