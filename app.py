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
svm_model = SVC(probability=True)  # Ensure probability=True to get probability estimates

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
    data = pd.read_excel(file_path, sheet_name='data', engine='openpyxl')
    train_data = pd.read_excel(file_path, sheet_name='oversample.train')
    test_data = pd.read_excel(file_path, sheet_name='test')
    train_ksvm = pd.read_excel(file_path, sheet_name='train_ksvm')
    test_ksvm = pd.read_excel(file_path, sheet_name='test_ksvm')
    return data, train_data, test_data, train_ksvm, test_ksvm

# Load data initially
data, train_data, test_data, train_ksvm, test_ksvm = load_data('data.xlsx')

# Load the pre-trained models
svm_scaler = load('svm_scaler.joblib')
svm_model = load('svm_model.joblib')
kmeans = load('ksvm_model.joblib')

# Standardize the test data
X_test_svm = svm_scaler.transform(test_data[['amount', 'second', 'days']])
y_test_svm = test_data['fraud']

X_test_ksvm = test_ksvm[['amount', 'second', 'days', 'cluster']]
y_test_ksvm = test_ksvm['fraud']

# Sidebar for navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Data", "Karakteristik Data", "Single Classifier: SVM", "Hybrid Classifier: KMeans SVM", "Pemilihan Metode Terbaik", "Prediksi Data"], key='navigation')
# Load data initially
data = pd.read_excel('data.xlsx', sheet_name='data')
def descriptive_stats(variable):
    stats = data.groupby('fraud')[variable].agg(['mean', 'std', 'min', 'median', 'max']).reset_index()
    
    # Mapping nama variabel
    variable_names = {
        'fraud': 'Jenis Transaksi',
        'amount': 'Nilai Transaksi',
        'second': 'Jeda Detik',
        'days': 'Jeda Hari',
        'variable': 'Variabel'
    }
    stats['variable'] = variable_names.get(variable, variable)
    
    stats = stats.rename(columns={
        'fraud': 'Jenis Transaksi',  # Rename kolom 'fraud' menjadi 'Jenis Transaksi'
        'mean': 'Rata-rata',
        'std': 'Standar Deviasi', 
        'min': 'Nilai Minimum',
        'median': 'Median',
        'max': 'Nilai Maksimum'
    })
    stats[['Rata-rata', 'Standar Deviasi', 'Nilai Minimum', 'Median', 'Nilai Maksimum']] = stats[['Rata-rata', 'Standar Deviasi', 'Nilai Minimum', 'Median', 'Nilai Maksimum']].applymap(lambda x: f"{x:.2f}")
    return stats
# Customizing fraud categories
data['fraud'] = data['fraud'].replace({0: 'Sah', 1: 'Penipuan'})

# Data Page
if page == "Data":
    st.title("Data Transaksi Kartu Kredit")
    st.subheader("Tabel Data Transaksi Kartu Kredit")
    
    # Rename columns
    data_renamed = data.rename(columns={
        'amount': 'Nilai Transaksi',
        'second': 'Jeda Detik',
        'days': 'Jeda Hari',
        'fraud': 'Jenis Transaksi'
    })
    
    # Selection for number of rows to display
    num_rows = st.slider('Pilih jumlah baris yang akan ditampilkan:', min_value=1, max_value=len(data_renamed), value=10)
    
    st.dataframe(data_renamed[['Nilai Transaksi', 'Jeda Detik', 'Jeda Hari', 'Jenis Transaksi']].head(num_rows))
    
# Descriptive Statistics Page
elif page == "Karakteristik Data":
    st.title("Karakteristik Data Penipuan Kartu Kredit")
    
    st.subheader("Tabel Statistika Deskriptif")
      # Calculate descriptive statistics for each variable
    amount_stats = descriptive_stats('amount')
    second_stats = descriptive_stats('second')
    days_stats = descriptive_stats('days')

    # Concatenate all stats into a single DataFrame
    desc_stats = pd.concat([amount_stats, second_stats, days_stats], ignore_index=True)

    # Display the descriptive statistics
    st.table(desc_stats)

    st.markdown("Tabel ini menampilkan statistik deskriptif untuk setiap variabel")

    # Visualization options
    st.subheader("Pilih Visualisasi")
    show_pie_chart = st.checkbox("Pie Chart Variabel Jenis Transaksi")
    show_boxplot_amount = st.checkbox("Boxplot Amount Berdasarkan Kategori Jenis Transaksi")
    show_boxplot_second = st.checkbox("Boxplot Second Berdasarkan Kategori Jenis Transaksi")
    show_boxplot_days = st.checkbox("Boxplot Days Berdasarkan Kategori Jenis Transaksi")

    # Pie chart for fraud variable
    if show_pie_chart:
        st.subheader("Pie Chart Variabel Fraud")
        fraud_counts = data['fraud'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(fraud_counts, labels=['Sah', 'Penipuan'], autopct='%1.1f%%', startangle=140)
        st.pyplot(fig1)
        st.markdown("Karakteristik data persentase jumlah transaksi sah dan tidak sah sangat timpang atau tidak seimbang, dengan data transaksi dengan kategori penipuan lebih sedikit dibanding transaksi sah")

    # Boxplot for variables based on fraud category
    if show_boxplot_amount or show_boxplot_second or show_boxplot_days:
        st.subheader("Boxplot Berdasarkan Kategori Fraud")
        if show_boxplot_amount:
            fig_amount, ax_amount = plt.subplots()
            sns.boxplot(x='fraud', y='amount', data=data, ax=ax_amount)
            ax_amount.set_title('Amount')
            st.pyplot(fig_amount)
            st.markdown("Distribusi nilai transaksi kategori penipuan lebih beragam dibanding kategori sah")
        if show_boxplot_second:
            fig_second, ax_second = plt.subplots()
            sns.boxplot(x='fraud', y='second', data=data, ax=ax_second)
            ax_second.set_title('Second')
            st.pyplot(fig_second)
            st.markdown("Distribusi jeda detik transaksi kategori penipuan dan sah pada variabel jeda detik cenderung seragam")
        if show_boxplot_days:
            fig_days, ax_days = plt.subplots()
            sns.boxplot(x='fraud', y='days', data=data, ax=ax_days)
            ax_days.set_title('Days')
            st.pyplot(fig_days)
            st.markdown("Distribusi jeda hari transaksi kategori penipuan dan sah pada variabel jeda detik cenderung seragam")


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
y_pred_cluster_svm = kmeans.predict(X_test_ksvm)
y_pred_cluster_svm_proba = kmeans.decision_function(X_test_ksvm)
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
fpr_svm, tpr_svm, _ = roc_curve(y_test_svm, y_pred_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

fpr_ksvm, tpr_ksvm, _ = roc_curve(y_test_ksvm, y_pred_cluster_svm)
roc_auc_ksvm = auc(fpr_ksvm, tpr_ksvm)

# SVM Predictions Page
if page == "Single Classifier: SVM":
    st.title("Prediksi Menggunakan SVM")
    st.write(f"Hasil prediksi menggunakan metode SVM dengan kernel Linear dan Cost =1")
    st.subheader("Confusion Matrix")
    cm_svm_df = pd.DataFrame(cm_svm, index=['Sah', 'Penipuan'], columns=['Prediksi Sah', 'Prediksi Penipuan'])
    # Menampilkan tabel confusion matrix dengan Streamlit
    st.table(cm_svm_df)
    
    st.subheader("Evaluasi Model")
    st.write(f"Akurasi: {accuracy_svm:.5f}")
    st.write(f"Sensitivitas: {recall_svm:.5f}")
    st.write(f"Spesifisitas: {precision_svm:.5f}")
    st.write(f"Hasil prediksi menggunakan metode SVM dengan kernel Linear dan Cost = 1, menghasilkan akurasi model yang memiliki performa yang sangat baik")

# KMeans SVM Predictions Page
elif page == "Hybrid Classifier: KMeans SVM":
    st.title("Prediksi Menggunakan KMeans SVM")
    st.write(f"Hasil prediksi menggunakan metode K-Means SVM dengan kernel Linear dan Cost =100")
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(cm_cluster_svm, index=['Sah', 'Penipuan'], columns=['Prediksi Sah', 'Prediksi Penipuan'])
    # Menampilkan tabel confusion matrix dengan Streamlit
    st.table(cm_df)
    
    st.subheader("Evaluasi Model")
    st.write(f"Akurasi: {accuracy_cluster_svm:.5f}")
    st.write(f"Sensitivitas: {recall_cluster_svm:.5f}")
    st.write(f"Spesifisitas: {precision_cluster_svm:.5f}")
    st.write(f"Hasil prediksi menggunakan metode K-Means SVM dengan kernel Linear dan Cost =100, menghasilkan akurasi model yang memiliki performa yang cukup baik tetapi masih memiliki ruang untuk perbaikan")

# Model Comparison Page
elif page == "Pemilihan Metode Terbaik":
    st.title("Perbandingan Model SVM dan KMeans SVM")
    # Membuat dataframe untuk SVM
    svm_metrics = {
    "Model": ["SVM"],
    "Akurasi": [accuracy_svm],
    "Sensitivitas": [recall_svm],
    "Spesifisitas": [precision_svm]
    }
    svm_df = pd.DataFrame(svm_metrics)
    # Membuat dataframe untuk K-Means SVM
    cluster_svm_metrics = {
    "Model": ["K-Means SVM"],
    "Akurasi": [accuracy_cluster_svm],
    "Sensitivitas": [recall_cluster_svm],
    "Spesifisitas": [precision_cluster_svm]
    }
    cluster_svm_df = pd.DataFrame(cluster_svm_metrics)
    combined_df = pd.concat([svm_df, cluster_svm_df], ignore_index=True)
    # Menampilkan tabel gabungan untuk membandingkan SVM dan K-Means SVM
    st.subheader("Perbandingan Evaluasi Model SVM dan K-Means SVM")
    st.dataframe(combined_df.style.format({"Akurasi": "{:.5f}", "Sensitivitas": "{:.5f}", "Spesifisitas": "{:.5f}"}))
    st.subheader("Kurva ROC Perbandingan Metode")
    fig, ax = plt.subplots()
    ax.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'ROC curve SVM (area = {roc_auc_svm:.2f})')
    ax.plot(fpr_ksvm, tpr_ksvm, color='red', lw=2, label=f'ROC curve KMeans SVM (area = {roc_auc_ksvm:.2f})')
    ax.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')  # Garis diagonal
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    # Compare accuracy and display message based on comparison
    if accuracy_svm > accuracy_cluster_svm:
        st.write("Metode SVM lebih baik dalam memprediksi penipuan transaksi kartu kredit.")
    elif accuracy_svm < accuracy_cluster_svm:
        st.write("Metode KMeans SVM lebih baik dalam memprediksi penipuan transaksi kartu kredit.")
    else:
        st.write("Metode SVM dan KMeans SVM memiliki performa prediksi yang sama untuk penipuan transaksi kartu kredit.")
# New Predictions Page
if page == "Prediksi Data":
    st.title("Prediksi Menggunakan Metode SVM")
    st.write("Masukkan nilai trasaksi, salah satu diantara jeda hari/jeda detik untuk memprediksi apakah transaksi kartu kredit yang terjadi terindikasi penipuan")
    # Input fields for amount, days, and seconds
    amount = st.number_input("Nilai Transaksi (Dalam US Dollar)", min_value=0.0)
    days = st.number_input("Jeda Hari (Isi salah satu antara Jeda Hari dan Jeda Detik)", min_value=0.0, step=1.0, key='days', on_change=update_days)
    second = st.number_input("Jeda Detik (Isi salah satu antara Jeda Hari dan Jeda Detik)", min_value=0.0, step=1.0, key='second', on_change=update_seconds)

    if st.button("Prediksi"):
        input_data = np.array([[amount, second, days]])
        standardized_input = svm_scaler.transform(input_data)
        prediction = svm_model.predict(standardized_input)
        st.write(f"Hasil Prediksi: {'Transaksi kartu kredit ini adalah Penipuan' if prediction[0] == 1 else 'Transaksi kartu kredit ini adalah Sah'}")
