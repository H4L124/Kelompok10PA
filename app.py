import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_curve, auc
from sklearn.cluster import KMeans
from joblib import load
from sklearn.preprocessing import StandardScaler

# Initialize dummy scaler and model
svm_scaler = StandardScaler()
svm_model = SVC()

# Example scaled data for initialization
svm_scaler.fit([[0, 0, 0], [30000, 86400*365, 365]])

# Dummy model fit for demonstration purposes
svm_model.fit(svm_scaler.transform([[0, 0, 0], [30000, 86400*365, 365]]), [0, 1])

def convert_days_to_seconds(days):
    return days * 86400

def convert_seconds_to_days(seconds):
    return seconds / 86400
    # Callback function when days input changes
def update_days():
    st.session_state.second = convert_days_to_seconds(st.session_state.days)

# Callback function when seconds input changes
def update_seconds():
    st.session_state.days = convert_seconds_to_days(st.session_state.second)

# Set page config
st.set_page_config(page_title="Dashboard Klasifikasi SVM dan KMeans SVM")

# Cache the data loading function to avoid reloading the data on each rerun
@st.cache_resource
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name='data')
    train_data = pd.read_excel(file_path, sheet_name='oversample.train')
    test_data = pd.read_excel(file_path, sheet_name='test')
    return data, train_data, test_data

# Load data initially
data, train_data, test_data = load_data('data.xlsx')

# Load the pre-trained models
svm_scaler = load('svm_scaler.joblib')
svm_model = load('svm_model.joblib')

kmeans_scaler = load('kmeans_scaler.joblib')
kmeans = load('kmeans_model.joblib')

# Standardize the test data
cluster = test_data['cluster'] 
X_test_svm = svm_scaler.transform(test_data[['amount', 'second', 'days']])
y_test_svm = test_data['fraud']

X_test_ksvm_prep = kmeans_scaler.transform(test_data[['amount', 'second', 'days']])
X_test_ksvm_prep =pd.DataFrame(X_test_ksvm_prep, columns=['amount', 'second', 'days']) 
X_test_ksvm = pd.concat([cluster, X_test_ksvm_prep], axis=1)
y_test_ksvm = test_data['fraud']

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih halaman", ["Deskripsi Data", "Prediksi SVM", "Prediksi KMeans SVM", "Perbandingan Model", "Prediksi Baru"])

# Descriptive Statistics Page
if page == "Deskripsi Data":
    st.title("Statistika Deskriptif")
    
    # Descriptive statistics for each variable
    def descriptive_stats(variable):
        stats = data.groupby('fraud')[variable].agg(['mean', 'std', 'min', 'median', 'max']).reset_index()
        stats['variable'] = variable
        return stats

    amount_stats = descriptive_stats('amount')
    second_stats = descriptive_stats('second')
    days_stats = descriptive_stats('days')

    # Concatenate all stats into a single DataFrame
    desc_stats = pd.concat([amount_stats, second_stats, days_stats], ignore_index=True)

    # Display the descriptive statistics
    st.subheader("Tabel Statistika Deskriptif")
    st.table(desc_stats)

    # Pie chart for fraud variable
    st.subheader("Pie Chart Variabel Fraud")
    fraud_counts = data['fraud'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(3, 3))
    ax1.pie(fraud_counts, labels=['Sah', 'Penipuan'], autopct='%1.1f%%', startangle=140)
    st.pyplot(fig1)
    
    # Boxplot for variables based on fraud category
    st.subheader("Boxplot Berdasarkan Kategori Fraud")
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(x='fraud', y='amount', data=data, ax=axes[0])
    axes[0].set_title('Amount')
    sns.boxplot(x='fraud', y='second', data=data, ax=axes[1])
    axes[1].set_title('Second')
    sns.boxplot(x='fraud', y='days', data=data, ax=axes[2])
    axes[2].set_title('Days')
    st.pyplot(fig2)

# Predictions and evaluations for SVM
y_pred_svm = svm_model.predict(X_test_svm)
y_pred_svm_proba = svm_model.decision_function(X_test_svm)
cm_svm = confusion_matrix(y_test_svm, y_pred_svm)

# Calculate accuracy, sensitivity, and specificity manually for SVM
TP_svm = cm_svm[1, 1]
FN_svm = cm_svm[1, 0]
FP_svm = cm_svm[0, 1]
TN_svm = cm_svm[0, 0]

accuracy_svm = (TP_svm + TN_svm) / (TP_svm + TN_svm + FP_svm + FN_svm)
recall_svm = TP_svm / (TP_svm + FN_svm)
precision_svm = TN_svm / (TN_svm + FP_svm)

# Predictions and evaluations for KMeans SVM
y_pred_cluster_svm = cluster_svm_model.predict(X_test_ksvm)
y_pred_cluster_svm_proba = kmeans.decision_function(X_test_ksvm)
y_pred_cluster_svm = kmeans.predict(X_test_svm)
cm_cluster_svm = confusion_matrix(y_test_ksvm, y_pred_cluster_svm)

# Calculate accuracy, sensitivity, and specificity manually for KMeans SVM
TP_cluster_svm = cm_cluster_svm[1, 1]
FN_cluster_svm = cm_cluster_svm[1, 0]
FP_cluster_svm = cm_cluster_svm[0, 1]
TN_cluster_svm = cm_cluster_svm[0, 0]

accuracy_cluster_svm = (TP_cluster_svm + TN_cluster_svm) / (TP_cluster_svm + TN_cluster_svm + FP_cluster_svm + FN_cluster_svm)
recall_cluster_svm = TP_cluster_svm / (TP_cluster_svm + FN_cluster_svm)
precision_cluster_svm = TN_cluster_svm / (TN_cluster_svm + FP_cluster_svm)

# Calculate ROC curve and AUC
fpr_svm, tpr_svm, _ = roc_curve(y_test_svm, y_pred_svm_proba)
roc_auc_svm = auc(fpr_svm, tpr_svm)

fpr_ksvm, tpr_ksvm, _ = roc_curve(y_test_ksvm, y_pred_cluster_svm_proba)
roc_auc_ksvm = auc(fpr_ksvm, tpr_ksvm)

# SVM Predictions Page
if page == "Prediksi SVM":
    st.title("Prediksi Menggunakan SVM")
    
    st.subheader("Confusion Matrix")
    st.table(cm_svm)
    
    st.subheader("Evaluasi Model")
    st.write(f"Akurasi: {accuracy_svm:.5f}")
    st.write(f"Sensitivitas: {recall_svm:.5f}")
    st.write(f"Spesifisitas: {precision_svm:.5f}")

# KMeans SVM Predictions Page
elif page == "Prediksi KMeans SVM":
    st.title("Prediksi Menggunakan KMeans SVM")
    
    st.subheader("Confusion Matrix")
    st.table(cm_cluster_svm)
    
    st.subheader("Evaluasi Model")
    st.write(f"Akurasi: {accuracy_cluster_svm:.5f}")
    st.write(f"Sensitivitas: {recall_cluster_svm:.5f}")
    st.write(f"Spesifisitas: {precision_cluster_svm:.5f}")

# Model Comparison Page
elif page == "Perbandingan Model":
    st.title("Perbandingan Model SVM dan KMeans SVM")
    st.subheader("Evaluasi Model SVM")
    st.write(f"Confusion Matrix SVM:\n{cm_svm}")
    st.write(f"Akurasi: {accuracy_svm:.5f}")
    st.write(f"Sensitivitas: {recall_svm:.5f}")
    st.write(f"Spesifisitas: {precision_svm:.5f}")
    
    st.subheader("Evaluasi Model KMeans SVM")
    st.write(f"Akurasi: {accuracy_cluster_svm:.5f}")
    st.write(f"Sensitivitas: {recall_cluster_svm:.5f}")
    st.write(f"Spesifisitas: {precision_cluster_svm:.5f}")
    
    # Compare accuracy and display message based on comparison
    if accuracy_svm > accuracy_cluster_svm:
        st.write("Metode SVM lebih baik dalam memprediksi penipuan transaksi kartu kredit.")
    elif accuracy_svm < accuracy_cluster_svm:
        st.write("Metode KMeans SVM lebih baik dalam memprediksi penipuan transaksi kartu kredit.")
    else:
        st.write("Metode SVM dan KMeans SVM memiliki performa prediksi yang sama untuk penipuan transaksi kartu kredit.")

    st.subheader("Kurva ROC Perbandingan Metode")
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM ROC curve (area = {roc_auc_svm:.2f})')
    ax3.plot(fpr_ksvm, tpr_ksvm, color='red', lw=2, label=f'KMeans SVM ROC curve (area = {roc_auc_ksvm:.2f})')
    ax3.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Kurva ROC')
    ax3.legend(loc="lower right")
    st.pyplot(fig3)

if page == "Prediksi Baru":
    st.title("Prediksi Menggunakan Model SVM")

    # Input fields for amount, days, and seconds
    amount = st.number_input("Amount", min_value=0.0, max_value=30000.0)
    
    # Input field for days (remove value=st.session_state.days)
    days = st.number_input("Days", min_value=0.0, step=1.0, key='days', on_change=update_days)

    # Input field for seconds (remove value=st.session_state.second)
    second = st.number_input("Second", min_value=0.0, step=1.0, key='second', on_change=update_seconds)

    if st.button("Prediksi"):
        input_data = np.array([[amount, second, days]])
        standardized_input = svm_scaler.transform(input_data)
        prediction = svm_model.predict(standardized_input)
        st.write(f"Hasil Prediksi: {'Transaksi kartu kredit ini adalah Penipuan' if prediction[0] == 1 else 'Transaksi kartu kredit ini adalah Sah'}")
