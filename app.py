"""
Used Car Price Prediction - Streamlit App
Egyptian Market (Hatla2ee data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------------------------------------
# Page config
# ----------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor | Egypt",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        text-align: center;
        color: #4a5568;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    .prediction-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .feature-box {
        background-color: #f7fafc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# Constants & helpers
# ----------------------------------------------------------
CURRENT_YEAR = 2024

# Popular brands in the Egyptian market (from the notebook data)
POPULAR_MAKES = [
    "Toyota", "Hyundai", "Kia", "Nissan", "Chevrolet", "Renault",
    "Fiat", "BMW", "Mercedes", "Peugeot", "Mitsubishi", "Honda",
    "Suzuki", "Volkswagen", "Ford", "Opel", "Skoda", "Chery",
    "MG", "Geely", "BYD", "Citroen", "Seat", "Mazda", "Audi",
    "Jeep", "Land Rover", "Volvo", "Subaru", "Daihatsu",
]

# Common colors seen in the dataset
COLORS = [
    "White", "Black", "Silver", "Gray", "Grey", "Dark grey",
    "Red", "Blue", "Petroleum", "Bronze", "Brown", "Beige",
    "Green", "Gold", "Orange", "Yellow", "Purple", "Maroon",
    "Champagne", "Titanium", "Navy", "Other",
]

YES_NO = ["Yes", "No"]


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline and brand statistics."""
    model_path = Path("xgb_pipeline.joblib")
    stats_path = Path("brand_stats.joblib")

    if not model_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    stats = joblib.load(stats_path) if stats_path.exists() else None
    return model, stats.get("brand_medians") if stats else None, stats.get("global_median") if stats else None


def compute_derived_features(
    year: int,
    mileage: float,
    auto: str,
    ac: str,
    ps: str,
    rc: str,
    make: str,
    brand_medians: dict | None,
    global_median: float | None,
) -> dict:
    """Recreate the same feature engineering used in training."""
    car_age = CURRENT_YEAR - year
    mile_per_year = mileage / (car_age + 1) if (car_age + 1) > 0 else mileage

    options_score = sum(
        1 for v in [auto, ac, ps, rc] if str(v).strip().lower() == "yes"
    )

    if brand_medians is not None and global_median is not None and global_median > 0:
        brand_med = brand_medians.get(make, global_median)
        luxury_score = round(brand_med / global_median, 2)
    else:
        # Fallback neutral score when stats are missing
        luxury_score = 1.0

    return {
        "car_age": float(car_age),
        "milePyear": float(mile_per_year),
        "Options_Score": int(options_score),
        "Luxury_Score": float(luxury_score),
    }


def format_price(price: float) -> str:
    """Format price with thousands separators and EGP."""
    return f"{price:,.0f} EGP"


# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown('<p class="main-header">🚗 Used Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Egyptian Market • Powered by XGBoost • Hatla2ee Data</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
model, brand_medians, global_median = load_artifacts()

if model is None:
    st.error(
        """
        **Model not found.**  
        Please train the model in the notebook and save it with:

        ```python
        import joblib

        # After training (best_model = results['XGBoost']['model'])
        joblib.dump(best_model, "xgb_pipeline.joblib")

        # Also save the brand statistics used for Luxury_Score
        joblib.dump(
            {
                "brand_medians": brand_medians.to_dict(),
                "global_median": global_median,
            },
            "brand_stats.joblib",
        )
        ```

        Place both files next to this `app.py` and restart the app.
        """
    )
    st.stop()

# ----------------------------------------------------------
# Input form
# ----------------------------------------------------------
st.subheader("Enter Car Details")

col1, col2, col3 = st.columns(3)

with col1:
    make = st.selectbox("Brand (Make)", options=POPULAR_MAKES, index=0)
    model_name = st.text_input(
        "Model",
        value="Corolla",
        help="e.g. Sportage, Tucson, Tipo, X1, Arrizo 5...",
    )
    color = st.selectbox("Color", options=COLORS, index=0)

with col2:
    year = st.number_input(
        "Year of Manufacture",
        min_value=1963,
        max_value=CURRENT_YEAR,
        value=2018,
        step=1,
    )
    mileage = st.number_input(
        "Mileage (Km)",
        min_value=0,
        max_value=2_500_000,
        value=80_000,
        step=1000,
    )

with col3:
    auto_trans = st.selectbox("Automatic Transmission", YES_NO, index=0)
    air_cond = st.selectbox("Air Conditioner", YES_NO, index=0)
    power_steer = st.selectbox("Power Steering", YES_NO, index=0)
    remote_ctrl = st.selectbox("Remote Control", YES_NO, index=0)

# ----------------------------------------------------------
# Derived features preview
# ----------------------------------------------------------
derived = compute_derived_features(
    year=year,
    mileage=mileage,
    auto=auto_trans,
    ac=air_cond,
    ps=power_steer,
    rc=remote_ctrl,
    make=make,
    brand_medians=brand_medians,
    global_median=global_median,
)

with st.expander("🔎 Derived Features (used by the model)", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Car Age (years)", f"{derived['car_age']:.0f}")
    c2.metric("Km per Year", f"{derived['milePyear']:,.0f}")
    c3.metric("Options Score", f"{derived['Options_Score']} / 4")
    c4.metric("Luxury Score", f"{derived['Luxury_Score']:.2f}")

# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------
st.markdown("---")

if st.button("🔮 Predict Price", use_container_width=True):
    # Build the exact feature row the pipeline expects
    input_df = pd.DataFrame(
        [
            {
                "Color": color,
                "Mileage": float(mileage),
                "Make": make,
                "Model": model_name,
                "Automatic Transmission": auto_trans,
                "Air Conditioner": air_cond,
                "Power Steering": power_steer,
                "Remote Control": remote_ctrl,
                "car_age": derived["car_age"],
                "milePyear": derived["milePyear"],
                "Options_Score": derived["Options_Score"],
                "Luxury_Score": derived["Luxury_Score"],
            }
        ]
    )

    try:
        prediction = model.predict(input_df)[0]
        prediction = max(0, float(prediction))  # safety floor

        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size:1.1rem; opacity:0.9;">Estimated Market Price</div>
                <div class="prediction-value">{format_price(prediction)}</div>
                <div style="font-size:0.95rem; opacity:0.85;">
                    {make} {model_name} • {year} • {mileage:,} km
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Simple confidence / context note
        st.info(
            "💡 This estimate is based on historical listings from Hatla2ee. "
            "Actual selling price can vary depending on condition, location, "
            "negotiations and market fluctuations."
        )

        # Feature summary
        st.subheader("Input Summary")
        summary = input_df.T.reset_index()
        summary.columns = ["Feature", "Value"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/car.png",
        width=80,
    )
    st.title("About")
    st.markdown(
        """
        This app predicts **used car prices in Egypt** using an 
        **XGBoost** model trained on scraped listings from Hatla2ee.

        **Key features used:**
        - Brand & Model
        - Year / Age
        - Mileage & km/year
        - Color
        - Options (AT, AC, PS, Remote)
        - Luxury Score (brand relative price)

        The model was trained with log-transformed target 
        (`TransformedTargetRegressor`) for better handling of 
        the wide price range.
        """
    )

    st.markdown("---")
    st.caption("Data source: Hatla2ee.com • Model: XGBoost")
    st.caption("For educational / demo purposes only.")
