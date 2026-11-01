# 🚗 Used Car Price Prediction – Egyptian Market

**Predict used car prices in Egypt using machine learning on real listings from [Hatla2ee](https://eg.hatla2ee.com).**

This project scrapes / uses publicly available used-car listings, performs thorough feature engineering, trains multiple regression models, and deploys the best model (XGBoost) as an interactive **Streamlit** web application.

---

## 📊 Project Highlights

| Metric              | Value              |
|---------------------|--------------------|
| **Best Model**      | XGBoost            |
| **Test R²**         | **0.840**          |
| **Test MAE**        | **152,934 EGP**    |
| **Test RMSE**       | 361,629 EGP        |
| **Training samples**| ~17,500            |
| **Target**          | Price (EGP)        |

---

## ✨ Features

- **Data cleaning & preprocessing**
  - Removal of duplicates and missing prices
  - Parsing of price, mileage, and year from raw text
- **Feature engineering**
  - `car_age` = current year − manufacturing year
  - `milePyear` = mileage / (age + 1)
  - `Options_Score` = count of available options (AT, AC, Power Steering, Remote)
  - `Luxury_Score` = brand median price / global median price
- **Models compared**
  - Ridge Regression
  - Random Forest
  - **XGBoost** (winner)
  - CatBoost
- **Target transformation** using `TransformedTargetRegressor` (`log1p` / `expm1`) for better handling of the wide price range
- **Interactive Streamlit app** for real-time price prediction

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Data**: pandas, numpy
- **ML**: scikit-learn, XGBoost, CatBoost, category-encoders
- **Visualization**: matplotlib, seaborn, plotly
- **Deployment**: Streamlit + joblib

---

## 📁 Project Structure

```
├── used_car_prices.ipynb      # Full EDA + modeling notebook
├── app.py                     # Streamlit web application
├── requirements.txt           # Dependencies
├── xgb_pipeline.joblib        # Trained pipeline (generate this)
├── brand_stats.joblib         # Brand medians for Luxury_Score
├── hatla2ee_scraped_data.csv  # Raw dataset (not included in repo)
└── README.md
```

> **Note**: The raw CSV is large and is not committed. You need to place `hatla2ee_scraped_data.csv` in the same folder as the notebook before running it.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/used-car-price-prediction-egypt.git
cd used-car-price-prediction-egypt
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (optional – if you don’t have the `.joblib` files)

1. Place `hatla2ee_scraped_data.csv` next to the notebook.
2. Open and run `used_car_prices.ipynb` end-to-end.
3. At the end of the notebook, save the artifacts:

```python
import joblib

best_model = results["XGBoost"]["model"]
joblib.dump(best_model, "xgb_pipeline.joblib")

joblib.dump(
    {
        "brand_medians": brand_medians.to_dict(),
        "global_median": float(global_median),
    },
    "brand_stats.joblib",
)
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser. Enter car details (brand, model, year, mileage, options…) and get an instant price estimate in Egyptian Pounds (EGP).

---

## 📈 Model Performance Summary

| Model          | MAE (EGP) | RMSE (EGP) | R²    |
|----------------|-----------|------------|-------|
| Ridge          | 270,515   | 785,516    | 0.245 |
| Random Forest  | 157,251   | 409,875    | 0.795 |
| **XGBoost**    | **152,934** | **361,629** | **0.840** |
| CatBoost       | 157,189   | 380,307    | 0.823 |

**Overfitting check (XGBoost)**  
- Train R²: 0.947  
- Test R²: 0.840  
- Gap: ~0.11 (acceptable)

---

## 🧠 How the Prediction Works

1. User inputs basic car attributes.
2. The app computes the same engineered features used during training.
3. The full scikit-learn pipeline (imputation → encoding → scaling → XGBoost) produces a price prediction on the original EGP scale thanks to `TransformedTargetRegressor`.

---

## 📦 Requirements

See `requirements.txt` for the full list. Main packages:

```
streamlit
pandas
numpy
scikit-learn
xgboost
category-encoders
joblib
```

---


## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.  
Predicted prices are estimates based on historical listings and should **not** be considered professional valuation advice. Actual market prices depend on vehicle condition, location, negotiation, and current market dynamics.

---

## 🙌 Acknowledgments

- Data source: [Hatla2ee](https://eg.hatla2ee.com)
- Libraries: scikit-learn, XGBoost, Streamlit, and the open-source community

---

## 📄 License

This project is released under the MIT License. Feel free to use, modify, and share.

---

**Made with ❤️ for the Egyptian used-car market**
