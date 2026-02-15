# AIF360 Bias Detection Tool

A comprehensive Streamlit application for detecting and analyzing bias in datasets and machine learning models using AIF360 fairness metrics.

## 🎯 Features

### Dual Detection Modes
- **Dataset Bias Detection**: Analyze bias in raw data before training
- **Model Bias Detection**: Analyze bias in model predictions

### 14 Fairness Metrics
- **Dataset Metrics (4)**: Statistical Parity Difference, Disparate Impact, Consistency, Base Rate
- **Classification Metrics (10)**: Equal Opportunity, Average Odds, FPR/FNR/FDR/FOR Differences, Error Rate, Theil Index

### Interactive UI
- CSV file upload with validation
- Protected attribute configuration
- Metric selection with expandable explanations
- Real-time bias calculation
- Interactive visualizations (gauge charts, distributions, comparisons)
- Downloadable reports (text and JSON)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Select Detection Type
Choose between Dataset Bias Detection or Model Bias Detection

### Step 2: Upload Data
- Upload your dataset (CSV format)
- For model detection, also upload predictions dataset
- Ensure CSV has: features, protected attribute, label column

### Step 3: Configure Attributes
- Select protected attribute (e.g., gender, race)
- Select label column (target variable)
- Define privileged and unprivileged group values

### Step 4: Select Metrics
Choose from available fairness metrics. Each metric has:
- Clear description
- Mathematical formula
- Ideal value
- Interpretation guidelines

### Step 5: Generate Report
Click "Detect Bias & Generate Report" to:
- Calculate selected metrics
- View detailed analysis with interpretations
- Explore interactive visualizations
- Download comprehensive reports

## 📊 Dataset Format

Your CSV should follow this structure:

```csv
age,education,gender,income
25,12,0,0
45,16,1,1
35,14,0,0
50,18,1,1
```

**Requirements:**
- Protected attribute column (binary: 0/1 or categorical)
- Label column (binary: 0/1 for classification)
- Additional feature columns (numeric or categorical)

## 📈 Metrics Explained

### Dataset Metrics

**Statistical Parity Difference**
- Measures selection rate differences between groups
- Ideal value: 0 (no difference)
- Threshold: ±0.1

**Disparate Impact**
- Ratio of selection rates between groups
- Ideal value: 1.0
- Threshold: 0.8-1.25 (80% rule)

**Consistency**
- Individual fairness measure
- Ideal value: 1.0 (perfect consistency)
- Threshold: ≥0.8

**Base Rate Difference**
- Difference in positive outcome rates
- Ideal value: 0 (similar rates)
- Threshold: ±0.1

### Classification Metrics

**Equal Opportunity Difference**
- TPR difference between groups
- Ensures equal chance of favorable outcomes

**Average Odds Difference**
- Average of TPR and FPR differences
- Comprehensive prediction fairness measure

**False Positive/Negative Rate Differences**
- Ensures fairness in error rates
- Measures false accusations and missed opportunities

**Theil Index**
- Generalized entropy measure
- Quantifies overall inequality

## 🏗️ Project Structure

```
aif360-bias-detector/
├── app.py                      # Main Streamlit application
├── bias_calculator.py          # Core bias calculation engine
├── metrics_info.py             # Metric definitions and explanations
├── report_generator.py         # Report generation utilities
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── sample_dataset.csv          # Test data
└── sample_predictions.csv      # Test predictions
```

## 🧪 Testing

Use the provided sample data:

1. **Dataset Bias Detection Test**:
   - Upload: `sample_dataset.csv`
   - Protected Attribute: `gender`
   - Label: `income`
   - Privileged: `1` (Male), Unprivileged: `0` (Female)

2. **Model Bias Detection Test**:
   - Upload both `sample_dataset.csv` and `sample_predictions.csv`
   - Same configuration as above
   - Observe bias amplification

## 🔧 Customization

### Adding New Metrics

1. Define metric in `metrics_info.py`:
```python
'new_metric': {
    'name': 'New Metric Name',
    'description': 'What it measures',
    'formula': 'Mathematical formula',
    'ideal_value': 'Target value',
    'interpretation': 'How to interpret',
    'threshold': {'min': x, 'max': y}
}
```

2. Implement calculation in `bias_calculator.py`:
```python
def _new_metric(self) -> float:
    # Calculation logic
    return metric_value
```

3. Add to dispatch in `_calculate_dataset_metric()` or `_calculate_classification_metric()`

### Adjusting Thresholds

Edit threshold values in `metrics_info.py` to match your requirements:
```python
'threshold': {'min': -0.05, 'max': 0.05}  # Stricter
```

## 🐛 Troubleshooting

### Module Import Errors
```bash
# Ensure all files are in the same directory
ls -la *.py
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Calculation Errors
- Verify dataset has binary labels (0/1)
- Check protected attribute has expected values
- Ensure predictions dataset matches original structure

## 📚 References

- [AIF360 Documentation](https://aif360.readthedocs.io/)
- [AIF360 GitHub Repository](https://github.com/Trusted-AI/AIF360)
- [Fairness Definitions Explained](https://fairmlbook.org/)

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Multi-class classification support
- Multiple protected attributes
- Integration with official AIF360 library
- PDF report generation
- Historical bias tracking

## 📄 License

MIT License - Feel free to use and modify for your projects.

## 🎓 Learning Resources

This tool demonstrates:
- AIF360 fairness metrics implementation
- Streamlit application development
- Interactive data visualization with Plotly
- Modular Python project architecture
- Bias detection best practices

## 💡 Tips

- Start with dataset metrics to understand inherent bias
- Use multiple metrics for comprehensive assessment
- Compare dataset vs model metrics to detect bias amplification
- Document threshold decisions for your domain
- Share reports with stakeholders for transparency

## 🏆 Acknowledgments

Built with inspiration from:
- IBM's AI Fairness 360 toolkit
- Fairness, Accountability, and Transparency in ML community
- Open-source bias detection research

---

**Happy bias detecting! For questions or issues, please open a GitHub issue.**
