# Insurance Claims Analysis

## 🎯 Project Overview
Comprehensive analysis of insurance claims data to identify patterns, detect fraud, and provide business insights for risk management and product optimization.

## 💼 Business Questions Addressed
1. What are the most common types of insurance claims?
2. How do claim amounts vary by customer demographics?
3. What is the fraud rate and how can we detect fraudulent claims?
4. Are there seasonal or temporal patterns in claims?
5. Which customer segments have the highest claim amounts?

## 🛠️ Technologies Used
- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **Statistical Analysis** - Descriptive statistics and pattern recognition

## 📊 Dataset
- **Source**: Kaggle / Auto Insurance Claims Dataset
- **Size**: 1,000 claims records
- **Features**: 40+ attributes including customer demographics, policy details, claim information
- **Key Variables**: claim_type, claim_amount, customer_age, fraud_reported, claim_status

## 📈 Key Findings

### Claims Distribution
- Analyzed 1,000+ insurance claims across multiple categories
- Average claim amount: $[X,XXX]
- Most common claim type: [Type]

### Fraud Detection
- Fraud rate: [X.X]%
- Fraudulent claims tend to have [specific pattern]
- High-risk indicators identified: [factors]

### Customer Insights
- Age group [XX-XX] has highest average claim amounts
- [State/Region] shows highest claim frequency
- Customer tenure correlates with [finding]

## 📁 Project Structure
```
insurance-claims-analysis/
│
├── insurance_claims.csv          # Dataset
├── claims_analysis.py            # Main analysis script
├── create_sample_data.py         # Sample data generator
├── README.md                     # Project documentation
│
└── outputs/                      # Generated visualizations
    ├── claims_by_type.png
    ├── claim_amount_distribution.png
    ├── fraud_analysis.png
    └── claims_by_age.png
```

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn
```

### Execution
```bash
# Download dataset from Kaggle and place in project folder
# Run analysis
python claims_analysis.py
```

## 💡 Business Impact
This analysis helps insurance companies:
- **Identify fraud patterns** - Reduce losses from fraudulent claims
- **Optimize pricing** - Better understand claim patterns by demographics
- **Improve risk assessment** - Target high-risk customer segments
- **Enhance customer service** - Understand claim processing efficiency

## 🔮 Future Enhancements
- [ ] Build predictive model for claim amounts
- [ ] Implement fraud detection ML algorithm
- [ ] Add time-series analysis for seasonal patterns
- [ ] Create interactive dashboard with Plotly/Streamlit
- [ ] Perform customer segmentation analysis

## 👤 Author
**Munkhkhishig Banzragch**
- Master's in Data Science, Western Michigan University
- Domain Expertise: Insurance Analytics, Risk Management
- LinkedIn: [linkedin.com/in/munkh-banz](https://www.linkedin.com/in/munkh-banz/)
- Email: munkh.mn@gmail.com

## 📝 License
This project is for educational and portfolio purposes.