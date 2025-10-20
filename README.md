🧪 Isotope Existence Predictor

An AI-powered regression model that predicts the existence probability of isotopes based on their physical and chemical properties.
Includes a Streamlit web interface for easy data upload, prediction visualization, and interactive exploration.

📁 Project Structure
📦 isotope-existence-predictor
├── app.py                         # Streamlit web app (for user interface)
├── train.py                       # Model training script
├── final_corrected_isotope_data.csv  # Input dataset
├── requirements.txt               # Dependencies
├── artifact/
│   └── model.pkl                  # Saved trained model
└── README.md                      # Project documentation

🚀 Quick Start (Local Setup)
1️⃣ Clone this repository
git clone https://github.com/<your-username>/isotope-existence-predictor.git
cd isotope-existence-predictor

2️⃣ Add your dataset

Place your dataset file as:

final_corrected_isotope_data.csv

3️⃣ Create a Virtual Environment
python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt


Your requirements.txt should include:

pandas
numpy
scikit-learn
joblib
streamlit

5️⃣ Train the Model
python train.py


✅ This will:

Clean and preprocess your dataset

Train a Gradient Boosting Regression model

Perform 5-fold cross-validation

Save the trained model to artifact/model.pkl

6️⃣ Run the Streamlit App
streamlit run app.py


Then open the displayed localhost URL (e.g. http://localhost:8501/
) to use your model interactively.

💡 Features
Feature	Description
Regression Model	Predicts continuous isotope existence probabilities
Auto Preprocessing	Handles numeric + categorical data automatically
Cross-Validation	Uses K-Fold evaluation for stable model accuracy
Error Handling	Cleans inf, NaN, and extreme outlier values
Streamlit UI	Simple drag-and-drop CSV upload + live predictions
Reusable Model	Model is saved as .pkl for easy reuse in any app
📊 Example Output (Console)

After training:

[INFO] Dataset loaded successfully. Shape: (400, 12)
[INFO] Cross-validation R² scores: [0.82, 0.79, 0.84, 0.81, 0.83]
[INFO] Mean R²: 0.8180
[INFO] Model saved as artifact/model.pkl

🧠 Tech Stack

Python 3.10+

Scikit-learn

Pandas / NumPy

Streamlit (for UI)

Joblib (for model persistence)

🔍 Future Enhancements

Add feature importance visualization

Deploy Streamlit app on Hugging Face or Streamlit Cloud

Integrate LLM-based reasoning (explain why a certain isotope exists)

👨‍💻 Author

Rashmika  [@rashmikad1743]
📧 “Building intelligent AI tools for science and education”
⭐ Don’t forget to star the repo if you find it useful!