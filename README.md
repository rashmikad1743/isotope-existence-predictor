# 🧪 Isotope Existence Predictor

A machine learning project for predicting the existence probability and half-life of isotopes based on their physical and chemical properties. Includes model training, data preprocessing, and a Streamlit web interface for interactive predictions.

---

## 📁 Project Structure

```
isotope-existence-predictor/
├── app.py                       # Streamlit web app
├── train.py                     # Model training script
├── final_corrected_isotope_data.csv  # Input dataset
├── requirements.txt             # Python dependencies
├── artifact/
│   └── model.pkl                # Trained model file
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### 1. Clone the repository
```sh
git clone https://github.com/rashmikad1743/isotope-existence-predictor.git
cd isotope-existence-predictor
```

### 2. Add your dataset
Place your `final_corrected_isotope_data.csv` in the project folder.

### 3. Create a virtual environment (recommended)
```sh
python -m venv venv
```
Activate on Windows:
```sh
venv\Scripts\activate
```

### 4. Install dependencies
```sh
pip install -r requirements.txt
```

### 5. Train the model
```sh
python train.py
```
This will clean and preprocess your data, train a Gradient Boosting Regression model, perform cross-validation, and save the trained model to `artifact/model.pkl`.

### 6. Run the Streamlit app
```sh
streamlit run app.py
```
Open the displayed localhost URL (e.g. http://localhost:8501/) in your browser.

---

## 💡 Features

| Feature            | Description                                         |
|--------------------|-----------------------------------------------------|
| Regression Model   | Predicts continuous isotope existence probabilities |
| Auto Preprocessing | Handles numeric and categorical data automatically  |
| Cross-Validation   | Uses K-Fold evaluation for stable accuracy          |
| Error Handling     | Cleans inf, NaN, and extreme outlier values         |
| Streamlit UI       | Drag-and-drop CSV upload, live predictions          |
| Reusable Model     | Model saved as `.pkl` for easy reuse                |

---

## 📊 Example Output

After training:
```
[INFO] Dataset loaded successfully. Shape: (400, 12)
[INFO] Cross-validation R² scores: [0.82, 0.79, 0.84, 0.81, 0.83]
[INFO] Mean R²: 0.8180
[INFO] Model saved as artifact/model.pkl
```

---

## 🧠 Tech Stack

- Python 3.10+
- Scikit-learn
- Pandas / NumPy
- Streamlit (UI)
- Joblib / Pickle (model persistence)

---

## 🔍 Future Enhancements

- Feature importance visualization
- Deploy Streamlit app on Hugging Face or Streamlit Cloud
- Integrate LLM-based reasoning (explain predictions)

---

## 👨‍💻 Author

**Rashmika** [@rashmikad1743]  
📧 Building intelligent AI tools for science and education

⭐ If you find this project useful, please star the repo!