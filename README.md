# 🩺 Diabetes Prediction App

A machine learning-powered web application that predicts the likelihood of diabetes based on key medical indicators. Built with Streamlit and scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Gender-Aware Input**: Smart form that adapts based on gender selection
- **Real-Time Predictions**: Instant diabetes risk assessment
- **Confidence Score**: Shows prediction probability and risk level
- **Health Recommendations**: Personalized advice based on results
- **User-Friendly UI**: Clean, intuitive interface with helpful tooltips
- **HuggingFace Integration**: Authenticated with HuggingFace Hub

## 🎯 Input Parameters

The app uses 8 key medical indicators for prediction:

| Parameter | Description | Range |
|-----------|-------------|-------|
| Gender | Patient's gender | Male/Female |
| Pregnancies | Number of pregnancies (females only) | 0-20 |
| Glucose | Plasma glucose concentration (mg/dL) | 0-200 |
| Blood Pressure | Diastolic blood pressure (mm Hg) | 0-130 |
| Skin Thickness | Triceps skin fold thickness (mm) | 0-100 |
| Insulin | 2-Hour serum insulin (mu U/ml) | 0-900 |
| BMI | Body Mass Index (kg/m²) | 10-70 |
| Diabetes Pedigree Function | Genetic influence score | 0.0-2.5 |
| Age | Age in years | 21-100 |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vznu7/diabeties-prediction.git
   cd diabeties-prediction
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   HF_KEY=your_huggingface_api_key_here
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 🌐 Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vznu7-diabeties-prediction.streamlit.app)

## 📦 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Model**: Random Forest Classifier ([scikit-learn](https://scikit-learn.org/))
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Storage**: [Joblib](https://joblib.readthedocs.io/)
- **API Integration**: [HuggingFace Hub](https://huggingface.co/)

## 🗂️ Project Structure

```
diabeties-prediction/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (not tracked)
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

## 🧠 Model Information

- **Algorithm**: Random Forest Classifier
- **Dataset**: Pima Indians Diabetes Database
- **Features**: 8 medical indicators
- **Accuracy**: ~77%

The model is trained on the Pima Indians Diabetes Dataset, which contains diagnostic measurements for female patients of Pima Indian heritage.

## ☁️ Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Add secrets in Settings → Secrets:
   ```toml
   HF_KEY = "your_huggingface_api_key"
   ```
6. Deploy!

## ⚠️ Disclaimer

This application is for **educational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Vznu7** - [GitHub Profile](https://github.com/Vznu7)

Project Link: [https://github.com/Vznu7/diabeties-prediction](https://github.com/Vznu7/diabeties-prediction)

---

<p align="center">
  Made with ❤️ using Streamlit
</p>
