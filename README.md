# 🏨 AI-Powered Payment Anomaly Detection
### Sunrise Budget Inn — Full Year 2024 Transaction Monitoring

> Built by **Ved Shastri** — Business & Financial Data Analyst | Banking · Payments · AI Systems

---

## 🔗 Live Demo
**[Click here to try the live dashboard →](YOUR_STREAMLIT_URL_HERE)**

---

## 💼 Business Problem

Payment anomalies — duplicate charges, inflated amounts, suspicious refunds, odd-hour transactions — cost hospitality businesses thousands in undetected revenue leakage and fraud every year. Manual review of thousands of transactions is slow, inconsistent, and expensive.

This system automates that process using machine learning, flagging suspicious transactions in real time and giving analysts a clear risk score and explanation for every flag — enabling faster, more accurate decisions with less manual effort.

---

## 📊 What It Does

| Feature | Description |
|---|---|
| 📈 Executive Dashboard | KPIs, monthly trends, anomaly breakdown by type and channel |
| 🚨 Flagged Transactions | Filterable table of all flagged transactions with risk scores |
| 📉 Trend Analysis | Revenue by room type, staff risk monitor, channel analysis |
| 🔍 Live Check | Enter any transaction and get an instant AI risk assessment |

---

## 🚨 Anomaly Types Detected

- **Duplicate Charges** — same booking charged more than once
- **Odd-Hour Transactions** — charges processed between 1am–4am
- **Unusually High Amounts** — transaction far above expected room rate
- **Refund Spikes** — large refunds with no matching original charge
- **Large Cash Payments** — high-value cash transactions flagged for AML review
- **No-show Fee Anomalies** — no-show fees exceeding room rate

---

## 📈 Results

- **8,311 transactions** analysed across full year 2024
- **165 anomalies** detected (2% anomaly rate)
- **$1.37M total revenue** monitored
- Real-time prediction on new transactions in under 1 second

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Scikit-learn | Isolation Forest anomaly detection model |
| Pandas / NumPy | Data processing and feature engineering |
| Streamlit | Interactive web dashboard |
| Plotly | Data visualisations |

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/payment-anomaly-detection.git
cd payment-anomaly-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the data file
# Place hotel_transactions.csv inside the /data folder

# 4. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
payment-anomaly-detection/
│
├── data/
│   └── hotel_transactions.csv    ← Transaction dataset
├── app.py                        ← Streamlit dashboard
├── model.py                      ← Isolation Forest ML model
├── requirements.txt              ← Dependencies
└── README.md
```

---

## 👤 About the Author

**Ved Shastri** — Business & Financial Data Analyst with expertise in banking, payments, and AI-integrated analytics systems. Currently based in Richmond, VA.

- 🔗 [LinkedIn](https://www.linkedin.com/in/vedshastri-7a309b172)
- 🌐 [Portfolio](https://ved-shastri-3lpysch.gamma.site/)
- 📧 shastrived45@gmail.com

---

*Model: Isolation Forest | Domain: Hospitality · Payments · Fraud Detection*
