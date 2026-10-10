from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "house_data.csv"

app = Flask(__name__)

LOCATIONS = [
    "Kozhikode",
    "Kochi",
    "Thiruvananthapuram",
    "Malappuram",
    "Bengaluru",
    "Thrissur",
    "Kannur",
]

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Age"]
FEATURES = ["Location"] + NUMERIC_FEATURES
TARGET = "Price"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find house_data.csv at {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    df["Location"] = df["Location"].astype(str).str.strip()

    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=required).copy()

    df = df[
        df["Location"].isin(LOCATIONS)
        & (df["Area"] > 0)
        & (df["Bedrooms"] >= 1)
        & (df["Bathrooms"] >= 1)
        & (df["Age"] >= 0)
        & (df["Price"] > 0)
    ]

    if len(df) < 5:
        raise ValueError(
            "Please provide at least 5 valid rows in house_data.csv."
        )

    return df


def build_model(df):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "location",
                OneHotEncoder(handle_unknown="ignore"),
                ["Location"],
            ),
            (
                "numbers",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(df[FEATURES], df[TARGET])
    return model


# Load the dataset and train the prediction model.
df = load_data()
model = build_model(df)


def empty_form():
    return {
        "location": LOCATIONS[0],
        "area": "",
        "bedrooms": "",
        "bathrooms": "",
        "age": "",
    }


@app.route("/", methods=["GET", "POST"])
def home():
    form_data = empty_form()
    prediction = None
    details = None
    error = None

    if request.method == "POST":
        action = request.form.get("action", "predict")

        # Clear all inputs and remove the previous prediction.
        if action == "clear":
            return render_template(
                "index.html",
                locations=LOCATIONS,
                form_data=form_data,
                prediction=None,
                details=None,
                error=None,
            )

        form_data = {
            "location": request.form.get("location", "").strip(),
            "area": request.form.get("area", "").strip(),
            "bedrooms": request.form.get("bedrooms", "").strip(),
            "bathrooms": request.form.get("bathrooms", "").strip(),
            "age": request.form.get("age", "").strip(),
        }

        try:
            location = form_data["location"]

            if location not in LOCATIONS:
                raise ValueError("Please select a valid location.")

            area = float(form_data["area"])
            bedrooms = int(form_data["bedrooms"])
            bathrooms = int(form_data["bathrooms"])
            age = int(form_data["age"])

            if area <= 0:
                raise ValueError("Area must be greater than zero.")

            if bedrooms < 1 or bathrooms < 1:
                raise ValueError(
                    "Bedrooms and bathrooms must be at least 1."
                )

            if age < 0:
                raise ValueError("House age cannot be negative.")

            if not area < float("inf"):
                raise ValueError("Please enter a valid area.")

            property_data = pd.DataFrame(
                [{
                    "Location": location,
                    "Area": area,
                    "Bedrooms": bedrooms,
                    "Bathrooms": bathrooms,
                    "Age": age,
                }],
                columns=FEATURES,
            )

            prediction = max(
                0,
                float(model.predict(property_data)[0])
            )

            details = {
                "location": location,
                "area": f"{area:,.0f}",
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age": age,
            }

        except (ValueError, TypeError, OverflowError):
            error = (
                "Please enter valid property details. "
                "Area must be positive, bedrooms and bathrooms "
                "must be whole numbers of at least 1, and age "
                "must be a non-negative whole number."
            )

    return render_template(
        "index.html",
        locations=LOCATIONS,
        form_data=form_data,
        prediction=prediction,
        details=details,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)