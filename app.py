"""
Aplikasi Prediksi Status Mahasiswa - Jaya Jaya Institut
Prototype Machine Learning untuk Deteksi Dini Dropout
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Status Mahasiswa | Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #546e7a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }
    .result-graduate {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 12px; padding: 1.5rem;
        color: white; text-align: center; font-size: 1.4rem; font-weight: 700;
    }
    .result-dropout {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 12px; padding: 1.5rem;
        color: white; text-align: center; font-size: 1.4rem; font-weight: 700;
    }
    .result-enrolled {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        border-radius: 12px; padding: 1.5rem;
        color: white; text-align: center; font-size: 1.4rem; font-weight: 700;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a237e;
        border-left: 4px solid #667eea;
        padding-left: 0.6rem;
        margin: 1rem 0 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102,126,234,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Model ────────────────────────────────────────────────────────────────
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, "model/model.joblib"))
    le    = joblib.load(os.path.join(BASE_DIR, "model/label_encoder.joblib"))
    with open(os.path.join(BASE_DIR, "model/features.json")) as f:
        features = json.load(f)
    return model, le, features

try:
    model, le, feature_names = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"❌ Gagal memuat model: {e}")

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎓 Sistem Prediksi Status Mahasiswa</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Jaya Jaya Institut — Deteksi Risiko Dropout Berbasis Machine Learning</div>', unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
    st.markdown("## 📋 Panduan Penggunaan")
    st.info("""
    **Langkah Prediksi:**
    1. Isi data mahasiswa di form utama
    2. Klik tombol **Prediksi Status**
    3. Lihat hasil prediksi & probabilitas
    4. Gunakan rekomendasi untuk intervensi
    """)
    st.markdown("---")
    st.markdown("**ℹ️ Tentang Model**")
    st.markdown("""
    - **Algoritma:** XGBoost Classifier
    - **Akurasi:** ~78%
    - **Kelas Target:** Graduate, Dropout, Enrolled
    - **Fitur:** 41 variabel
    """)
    st.markdown("---")
    st.markdown("**🏫 Jaya Jaya Institut**")
    st.caption("Sistem ini membantu staf akademik mengidentifikasi mahasiswa berisiko tinggi dropout")

# ─── Main Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi Individual", "📊 Prediksi Batch (CSV)", "📖 Info & Panduan"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Individual Prediction
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Masukkan Data Mahasiswa")

    with st.form("prediction_form"):
        # ── Informasi Demografis ──────────────────────────────
        st.markdown('<div class="section-header">👤 Informasi Demografis</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            marital_status = st.selectbox(
                "Status Pernikahan",
                options=[1,2,3,4,5,6],
                format_func=lambda x: {1:"Single",2:"Menikah",3:"Duda/Janda",4:"Cerai",5:"Kohabitasi",6:"Pisah Legal"}.get(x,str(x))
            )
            gender = st.selectbox("Jenis Kelamin", [0,1], format_func=lambda x: "Wanita" if x==0 else "Pria")
        with col2:
            age_at_enrollment = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
            nationality = st.selectbox(
                "Kode Kebangsaan",
                options=[1,2,6,11,13,14,17,21,22,24,25,26,32,41,62,100,101,103,105,108,109],
                format_func=lambda x: {1:"1 - Portuguese",2:"2 - German",6:"6 - Spanish",11:"11 - Italian",13:"13 - Dutch",14:"14 - English",17:"17 - Lithuanian",21:"21 - Angolan",
                                       22:"22 - Cape Verdean",24:"24 - Guinean",25:"25 - Mozambican",26:"26 - Santomean",32:"32 - Turkish",41:"41 - Brazilian",62:"62 - Romanian",
                                        100:"100 - Moldova (Republic of)",101:"101 - Mexican",103:"103 - Ukrainian",105:"105 - Russian",108:"108 - Cuban",109:"109 - Colombian"
                                       }.get(x,str(x))
            )
        with col3:
            displaced = st.selectbox("Status Pengungsi (Displaced)", [0,1], format_func=lambda x:"Tidak" if x==0 else "Ya")
            international = st.selectbox("Mahasiswa Internasional", [0,1], format_func=lambda x:"Tidak" if x==0 else "Ya")

        # ── Informasi Akademik ────────────────────────────────
        st.markdown('<div class="section-header">📚 Latar Belakang Akademik</div>', unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3)
        with col4:
            application_mode = st.selectbox("Mode Pendaftaran", [1,2,5,7,10,15,16,17,18,26,27,39,42,43,44,51,53,57])
            application_order = st.number_input("Urutan Pilihan (0–9)", min_value=0, max_value=9, value=1)
        with col5:
            course = st.selectbox("Program Studi", [33,171,8014,9003,9070,9085,9119,9130,9147,9238,9254,9500,9556,9670,9773,9853,9991],
                                  help="Kode program studi")
            daytime_evening = st.selectbox("Waktu Kuliah", [1,0], format_func=lambda x:"Siang" if x==1 else "Malam")
        with col6:
            prev_qual = st.selectbox("Kualifikasi Sebelumnya",
                        options=[1,2,3,4,5,6,9,10,12,14,15,19,38,39,42,43],
                        format_func=lambda x: {1:"1 - Secondary education",2:"2 - Higher education - bachelor's degree",3:"3 - Higher education - degree",4:"4 - Higher education - master's",5:"5 - Higher education - doctorate",
                                               6:"6 - Frequency of higher education",9:"9 - 12th year of schooling - not completed",10:"10 - 11th year of schooling - not completed",12:"12 - Other - 11th year of schooling",
                                               14:"14 - 10th year of schooling",15:"15 - 10th year of schooling - not completed",19:"19 - Basic education 3rd cycle (9th/10th/11th year) or equiv.",38:"38 - Basic education 2nd cycle (6th/7th/8th year) or equiv.",
                                               39:"39 - Technological specialization course 40 - Higher education - degree (1st cycle)",42:"42 - Professional higher technical course",43:"43 - Higher education - master (2nd cycle)"
                                       }.get(x,str(x))
            )
            prev_qual_grade = st.slider("Nilai Kualifikasi Sebelumnya", 95.0, 190.0, 130.0, 0.5)

        admission_grade = st.slider("Nilai Masuk (Admission Grade)", 95.0, 190.0, 127.0, 0.5)

        # ── Latar Belakang Orang Tua ──────────────────────────
        st.markdown('<div class="section-header">👨‍👩‍👦 Latar Belakang Orang Tua</div>', unsafe_allow_html=True)
        col7, col8 = st.columns(2)
        with col7:
            mothers_qual = st.selectbox("Pendidikan Ibu",
                            options=[1,2,3,4,5,6,9,10,11,12,14,18,19,22,26,27,29,30,34,35,36,37,38,39,40,41,42,43,44],
                            format_func=lambda x: {1:"1 - Secondary Education - 12th Year of Schooling or Eq.",2:"2 - Higher Education - Bachelor's Degree",3:"3 - Higher Education - Degree",
                                                   4:"4 - Higher Education - Master's",5:"5 - Higher Education - Doctorate",6:"6 - Frequency of Higher Education",9:"9 - 12th Year of Schooling - Not Completed",
                                                   10:"10 - 11th Year of Schooling - Not Completed",11:"11 - 7th Year (Old)",12:"12 - Other - 11th Year of Schooling",14:"14 - 10th Year of Schooling",18:"18 - General commerce course",
                                                   19:"19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",22:"22 - Technical-professional course",26:"26 - 7th year of schooling",27:"27 - 2nd cycle of the general high school course",
                                                   29:"29 - 9th Year of Schooling - Not Completed",30:"30 - 8th year of schooling",34:"34 - Unknown",35:"35 - Can't read or write",36:"36 - Can read without having a 4th year of schooling",
                                                   37:"37 - Basic education 1st cycle (4th/5th year) or equiv.",38:"38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",39:"39 - Technological specialization course",40:"40 - Higher education - degree (1st cycle)",
                                                   41:"41 - Specialized higher studies course",42:"42 - Professional higher technical course",43:"43 - Higher Education - Master (2nd cycle)",44:"44 - Higher Education - Doctorate (3rd cycle)"	
                                       }.get(x,str(x))            
            )
            mothers_occ = st.selectbox("Pekerjaan Ibu",
                            options=[0,1,2,3,4,5,6,7,8,9,10,90,99,122,123,125,131,132,134,141,143,144,151,152,153,171,173,175,191,192,193,194],
                            format_func=lambda x: {0:"0 - Student",1:"1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",2:"2 - Specialists in Intellectual and Scientific Activities",3:"3 - Intermediate Level Technicians and Professions",4:"4 - Administrative staff",5:"5 - Personal Services, Security and Safety Workers and Sellers",
                                                   6:"6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",7:"7 - Skilled Workers in Industry, Construction and Craftsmen",8:"8 - Installation and Machine Operators and Assembly Workers",9:"9 - Unskilled Workers",10:"10 - Armed Forces Professions",90:"90 - Other Situation",99:"99 - (blank)",122:"122 - Health professionals",
                                                   123:"123 - teachers",125:"125 - Specialists in information and communication technologies (ICT)",131:"131 - Intermediate level science and engineering technicians and professions",132:"132 - Technicians and professionals, of intermediate level of health",134:"134 - Intermediate level technicians from legal, social, sports, cultural and similar services",
                                                   141:"141 - Office workers, secretaries in general and data processing operators",143:"143 - Data, accounting, statistical, financial services and registry-related operators",144:"144 - Other administrative support staff",151:"151 - personal service workers",152:"152 - sellers",153:"153 - Personal care workers and the like",171:"171 - Skilled construction workers and the like, except electricians",
                                                   173:"173 - Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like",175:"175 - Workers in food processing, woodworking, clothing and other industries and crafts",191:"191 - cleaning workers",192:"192 - Unskilled workers in agriculture, animal production, fisheries and forestry",
                                                   193:"193 - Unskilled workers in extractive industry, construction, manufacturing and transport",194:"194 - Meal preparation assistants"
                                       }.get(x,str(x))       
            )
        with col8:
            fathers_qual = st.selectbox("Pendidikan Ayah",
                            options=[1,2,3,4,5,6,9,10,11,12,13,14,18,19,20,22,25,26,27,29,30,31,33,34,35,36,37,38,39,40,41,42,43,44],
                            format_func=lambda x: {1:"1 - Secondary Education - 12th Year of Schooling or Eq.",2:"2 - Higher Education - Bachelor's Degree",3:"3 - Higher Education - Degree",
                                                   4:"4 - Higher Education - Master's",5:"5 - Higher Education - Doctorate",6:"6 - Frequency of Higher Education",9:"9 - 12th Year of Schooling - Not Completed",
                                                   10:"10 - 11th Year of Schooling - Not Completed",11:"11 - 7th Year (Old)",12:"12 - Other - 11th Year of Schooling",13:"13 - 2nd year complementary high school course",14:"14 - 10th Year of Schooling",18:"18 - General commerce course",
                                                   19:"19 - Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.",20:"20 - Complementary High School Course",22:"22 - Technical-professional course",25:"25 - Complementary High School Course - not concluded",26:"26 - 7th year of schooling",27:"27 - 2nd cycle of the general high school course",
                                                   29:"29 - 9th Year of Schooling - Not Completed",30:"30 - 8th year of schooling",31:"31 - General Course of Administration and Commerce",33:"33 - Supplementary Accounting and Administration",34:"34 - Unknown",35:"35 - Can't read or write",36:"36 - Can read without having a 4th year of schooling",
                                                   37:"37 - Basic education 1st cycle (4th/5th year) or equiv.",38:"38 - Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.",39:"39 - Technological specialization course",40:"40 - Higher education - degree (1st cycle)",
                                                   41:"41 - Specialized higher studies course",42:"42 - Professional higher technical course",43:"43 - Higher Education - Master (2nd cycle)",44:"44 - Higher Education - Doctorate (3rd cycle)"	
                                       }.get(x,str(x))     
            )
            fathers_occ = st.selectbox("Pekerjaan Ayah",
                            options=[0,1,2,3,4,5,6,7,8,9,10,90,99,101,102,103,112,114,121,122,123,124,131,132,134,141,143,144,151,152,153,154,161,163,171,172,174,175,181,182,183,192,193,194,195],
                            format_func=lambda x: {0:"0 - Student",1:"1 - Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers",2:"2 - Specialists in Intellectual and Scientific Activities",3:"3 - Intermediate Level Technicians and Professions",4:"4 - Administrative staff",5:"5 - Personal Services, Security and Safety Workers and Sellers",
                                                   6:"6 - Farmers and Skilled Workers in Agriculture, Fisheries and Forestry",7:"7 - Skilled Workers in Industry, Construction and Craftsmen",8:"8 - Installation and Machine Operators and Assembly Workers",9:"9 - Unskilled Workers",10:"10 - Armed Forces Professions",90:"90 - Other Situation",99:"99 - (blank)",
                                                   101:"101 - Armed Forces Officers",102:"102 - Armed Forces Sergeants",103:"103 - Other Armed Forces personnel",112:"112 - Directors of administrative and commercial services",114:"114 - Hotel, catering, trade and other services directors",121:"121 - Specialists in the physical sciences, mathematics, engineering and related techniques",
                                                   122:"122 - Health professionals",123:"123 - teachers",124:"124 - Specialists in finance, accounting, administrative organization, public and commercial relations",131:"131 - Intermediate level science and engineering technicians and professions",132:"132 - Technicians and professionals, of intermediate level of health",134:"134 - Intermediate level technicians from legal, social, sports, cultural and similar services",
                                                   141:"141 - Office workers, secretaries in general and data processing operators",143:"143 - Data, accounting, statistical, financial services and registry-related operators",144:"144 - Other administrative support staff",151:"151 - personal service workers",152:"152 - sellers",153:"153 - Personal care workers and the like",
                                                   154:"154 - Protection and security services personnel",161:"161 - Market-oriented farmers and skilled agricultural and animal production workers",163:"163 - Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence",
                                                   171:"171 - Skilled construction workers and the like, except electricians",172:"172 - Skilled workers in metallurgy, metalworking and similar",174:"174 - Skilled workers in electricity and electronics",175:"175 - Workers in food processing, woodworking, clothing and other industries and crafts",
                                                   181:"181 - Fixed plant and machine operators",182:"182 - assembly workers",183:"183 - Vehicle drivers and mobile equipment operators", 192:"192 - Unskilled workers in agriculture, animal production, fisheries and forestry",
                                                   193:"193 - Unskilled workers in extractive industry, construction, manufacturing and transport",194:"194 - Meal preparation assistants",195:"195 - Street vendors (except food) and street service providers"
                                       }.get(x,str(x))       
            )

        # ── Informasi Finansial ───────────────────────────────
        st.markdown('<div class="section-header">💰 Informasi Finansial</div>', unsafe_allow_html=True)
        col9, col10, col11 = st.columns(3)
        with col9:
            tuition_up_to_date = st.selectbox("Biaya Kuliah Terbayar?", [1,0], format_func=lambda x:"Ya" if x==1 else "Tidak")
            scholarship = st.selectbox("Penerima Beasiswa?", [0,1], format_func=lambda x:"Tidak" if x==0 else "Ya")
        with col10:
            debtor = st.selectbox("Status Debtor?", [0,1], format_func=lambda x:"Tidak" if x==0 else "Ya")
            educational_special = st.selectbox("Kebutuhan Khusus?", [0,1], format_func=lambda x:"Tidak" if x==0 else "Ya")
        with col11:
            unemployment_rate = st.number_input("Tingkat Pengangguran (%)", 7.0, 17.0, 11.1, 0.1)
            inflation_rate = st.number_input("Tingkat Inflasi (%)", -1.0, 4.0, 1.4, 0.1)
            gdp = st.number_input("Pertumbuhan GDP", -4.5, 3.6, 0.32, 0.01)

        # ── Performa Semester 1 ───────────────────────────────
        st.markdown('<div class="section-header">📈 Performa Semester 1</div>', unsafe_allow_html=True)
        col12, col13, col14 = st.columns(3)
        with col12:
            cu1_credited  = st.number_input("Unit Dikreditkan S1", 0, 20, 0)
            cu1_enrolled  = st.number_input("Unit Diambil S1", 0, 26, 6)
        with col13:
            cu1_evaluations = st.number_input("Evaluasi S1", 0, 45, 6)
            cu1_approved    = st.number_input("Unit Disetujui S1", 0, 26, 5)
        with col14:
            cu1_grade    = st.number_input("Nilai Rata-rata S1", 0.0, 20.0, 12.0, 0.1)
            cu1_no_eval  = st.number_input("Tanpa Evaluasi S1", 0, 12, 0)

        # ── Performa Semester 2 ───────────────────────────────
        st.markdown('<div class="section-header">📈 Performa Semester 2</div>', unsafe_allow_html=True)
        col15, col16, col17 = st.columns(3)
        with col15:
            cu2_credited  = st.number_input("Unit Dikreditkan S2", 0, 19, 0)
            cu2_enrolled  = st.number_input("Unit Diambil S2", 0, 23, 6)
        with col16:
            cu2_evaluations = st.number_input("Evaluasi S2", 0, 33, 6)
            cu2_approved    = st.number_input("Unit Disetujui S2", 0, 20, 5)
        with col17:
            cu2_grade    = st.number_input("Nilai Rata-rata S2", 0.0, 20.0, 12.0, 0.1)
            cu2_no_eval  = st.number_input("Tanpa Evaluasi S2", 0, 12, 0)

        submitted = st.form_submit_button("🔮 Prediksi Status Mahasiswa")

    # ── Hasil Prediksi ────────────────────────────────────────
    if submitted and model_loaded:
        avg_approved = (cu1_approved + cu2_approved) / 2
        avg_grade = (cu1_grade + cu2_grade) / 2
        total_enr = cu1_enrolled + cu2_enrolled
        total_app = cu1_approved + cu2_approved
        approval_rate = total_app / total_enr if total_enr > 0 else 0
        financial_risk = (1 - tuition_up_to_date) + debtor - scholarship

        input_data = {
            'Marital_status': marital_status,
            'Application_mode': application_mode,
            'Application_order': application_order,
            'Course': course,
            'Daytime_evening_attendance': daytime_evening,
            'Previous_qualification': prev_qual,
            'Previous_qualification_grade': prev_qual_grade,
            'Nacionality': nationality,
            'Mothers_qualification': mothers_qual,
            'Fathers_qualification': fathers_qual,
            'Mothers_occupation': mothers_occ,
            'Fathers_occupation': fathers_occ,
            'Admission_grade': admission_grade,
            'Displaced': displaced,
            'Educational_special_needs': educational_special,
            'Debtor': debtor,
            'Tuition_fees_up_to_date': tuition_up_to_date,
            'Gender': gender,
            'Scholarship_holder': scholarship,
            'Age_at_enrollment': age_at_enrollment,
            'International': international,
            'Curricular_units_1st_sem_credited': cu1_credited,
            'Curricular_units_1st_sem_enrolled': cu1_enrolled,
            'Curricular_units_1st_sem_evaluations': cu1_evaluations,
            'Curricular_units_1st_sem_approved': cu1_approved,
            'Curricular_units_1st_sem_grade': cu1_grade,
            'Curricular_units_1st_sem_without_evaluations': cu1_no_eval,
            'Curricular_units_2nd_sem_credited': cu2_credited,
            'Curricular_units_2nd_sem_enrolled': cu2_enrolled,
            'Curricular_units_2nd_sem_evaluations': cu2_evaluations,
            'Curricular_units_2nd_sem_approved': cu2_approved,
            'Curricular_units_2nd_sem_grade': cu2_grade,
            'Curricular_units_2nd_sem_without_evaluations': cu2_no_eval,
            'Unemployment_rate': unemployment_rate,
            'Inflation_rate': inflation_rate,
            'GDP': gdp,
            'Avg_approved': avg_approved,
            'Avg_grade': avg_grade,
            'Approval_rate': approval_rate,
            'Financial_risk': financial_risk,
        }

        input_df = pd.DataFrame([input_data])
        
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        pred_encoded = model.predict(input_df)[0]
        pred_label   = le.inverse_transform([pred_encoded])[0]
        pred_prob    = model.predict_proba(input_df)[0]
        prob_dict    = dict(zip(le.classes_, pred_prob))

        st.markdown("## 🎯 Hasil Prediksi")

        col_r1, col_r2 = st.columns([1, 1.5])

        with col_r1:
            if pred_label == "Graduate":
                st.markdown(f'<div class="result-graduate">✅ GRADUATE<br><small>Mahasiswa diprediksi akan lulus</small></div>', unsafe_allow_html=True)
            elif pred_label == "Dropout":
                st.markdown(f'<div class="result-dropout">⚠️ DROPOUT<br><small>Mahasiswa berisiko tinggi dropout</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-enrolled">📚 ENROLLED<br><small>Mahasiswa masih aktif berkuliah</small></div>', unsafe_allow_html=True)

            st.markdown("#### Probabilitas Prediksi")
            for status, color in [("Graduate","#2ecc71"), ("Dropout","#e74c3c"), ("Enrolled","#f39c12")]:
                prob_val = prob_dict.get(status, 0)
                st.markdown(f"**{status}**")
                st.progress(float(prob_val))
                st.caption(f"{prob_val*100:.1f}%")

        with col_r2:
            # Pie chart probabilitas
            fig, ax = plt.subplots(figsize=(5, 4))
            colors_pie = ['#2ecc71', '#e74c3c', '#f39c12']
            labels_pie = ["Graduate", "Dropout", "Enrolled"]
            values_pie = [prob_dict.get(l, 0) for l in labels_pie]
            wedges, texts, autotexts = ax.pie(
                values_pie, labels=labels_pie, colors=colors_pie,
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2},
                textprops={'fontsize': 11}
            )
            ax.set_title("Distribusi Probabilitas Prediksi", fontweight='bold')
            st.pyplot(fig)
            plt.close()

        # Rekomendasi action
        st.markdown("### 💡 Rekomendasi Action")
        if pred_label == "Dropout":
            st.error("🚨 **Mahasiswa ini memiliki risiko TINGGI untuk dropout. Diperlukan intervensi segera!**")
            risk_factors = []
            if tuition_up_to_date == 0:
                risk_factors.append("❌ Biaya kuliah belum terbayar — hubungi bagian keuangan untuk pembayaran cicilan")
            if debtor == 1:
                risk_factors.append("❌ Mahasiswa memiliki hutang — pertimbangkan program keringanan finansial")
            if cu2_approved < 3:
                risk_factors.append("❌ Unit disetujui semester 2 sangat rendah — perlu bimbingan akademik intensif")
            if cu2_grade < 10:
                risk_factors.append("❌ Nilai rata-rata semester 2 rendah — rekomendasikan tutoring")
            if age_at_enrollment > 30:
                risk_factors.append("⚠️ Usia pendaftar > 30 — sediakan layanan konseling khusus mahasiswa non-tradisional")

            if risk_factors:
                st.markdown("**Faktor Risiko yang Terdeteksi:**")
                for f in risk_factors:
                    st.markdown(f)
            else:
                st.markdown("- Lakukan konsultasi tatap muka dengan mahasiswa\n- Evaluasi beban perkuliahan\n- Tawarkan program pendampingan akademik")
        elif pred_label == "Graduate":
            st.success("✅ **Mahasiswa diprediksi akan lulus dengan baik. Pertahankan performa ini!**")
            st.markdown("- Pastikan mahasiswa terdaftar pada mata kuliah semester berikutnya\n- Tawarkan program pengembangan karir\n- Pertimbangkan program mentor/tutor sebaya")
        else:
            st.warning("📊 **Mahasiswa masih aktif. Lakukan monitoring berkala untuk mencegah risiko dropout.**")
            st.markdown("- Monitor perkembangan akademik semester ini\n- Pastikan biaya kuliah terbayar tepat waktu\n- Tawarkan bimbingan akademik bila nilai menurun")


# ═══════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📂 Prediksi Batch Menggunakan File CSV")
    st.info("Upload file CSV berisi data mahasiswa (format sama dengan dataset training). Sistem akan memprediksi status setiap mahasiswa.")

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file and model_loaded:
        try:
            df_batch = pd.read_csv(uploaded_file, sep=';')
            st.success(f"✅ File berhasil dimuat! {df_batch.shape[0]} mahasiswa, {df_batch.shape[1]} kolom")
            st.dataframe(df_batch.head(5))

            if st.button("🔮 Jalankan Prediksi Batch"):
                # Feature engineering
                df_batch['Avg_approved'] = (df_batch.get('Curricular_units_1st_sem_approved', 0) +
                                             df_batch.get('Curricular_units_2nd_sem_approved', 0)) / 2
                df_batch['Avg_grade'] = (df_batch.get('Curricular_units_1st_sem_grade', 0) +
                                          df_batch.get('Curricular_units_2nd_sem_grade', 0)) / 2
                enr = (df_batch.get('Curricular_units_1st_sem_enrolled', 0) +
                       df_batch.get('Curricular_units_2nd_sem_enrolled', 0))
                app = (df_batch.get('Curricular_units_1st_sem_approved', 0) +
                       df_batch.get('Curricular_units_2nd_sem_approved', 0))
                df_batch['Approval_rate'] = np.where(enr > 0, app / enr, 0)
                df_batch['Financial_risk'] = (
                    (1 - df_batch.get('Tuition_fees_up_to_date', 1)) +
                    df_batch.get('Debtor', 0) -
                    df_batch.get('Scholarship_holder', 0)
                )

                for col in feature_names:
                    if col not in df_batch.columns:
                        df_batch[col] = 0

                X_batch = df_batch[feature_names]
                preds = le.inverse_transform(model.predict(X_batch))
                probs = model.predict_proba(X_batch)

                df_result = df_batch.copy()
                df_result['Predicted_Status'] = preds
                df_result['P_Dropout']  = probs[:, list(le.classes_).index('Dropout')].round(3)
                df_result['P_Enrolled'] = probs[:, list(le.classes_).index('Enrolled')].round(3)
                df_result['P_Graduate'] = probs[:, list(le.classes_).index('Graduate')].round(3)
                df_result['Risk_Level'] = pd.cut(
                    df_result['P_Dropout'], bins=[0, 0.3, 0.6, 1.0],
                    labels=['Rendah', 'Sedang', 'Tinggi']
                )

                st.markdown("#### Hasil Prediksi Batch")
                st.dataframe(df_result[['Predicted_Status','P_Dropout','P_Enrolled','P_Graduate','Risk_Level']].head(20))

                # Summary stats
                col_b1, col_b2, col_b3 = st.columns(3)
                vc = pd.Series(preds).value_counts()
                col_b1.metric("🎓 Graduate", vc.get('Graduate', 0))
                col_b2.metric("⚠️ Dropout", vc.get('Dropout', 0))
                col_b3.metric("📚 Enrolled", vc.get('Enrolled', 0))

                csv_out = df_result.to_csv(index=False)
                st.download_button("📥 Download Hasil Prediksi", csv_out,
                                   file_name="prediksi_status_mahasiswa.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error saat memproses file: {e}")

# ═══════════════════════════════════════════════════════════
# TAB 3 — Info
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📖 Dokumentasi Sistem")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        #### 🎯 Tentang Sistem Ini
        Sistem ini menggunakan model **XGBoost Classifier** yang telah dilatih
        dengan data 4.424 mahasiswa dari Jaya Jaya Institut untuk memprediksi
        apakah seorang mahasiswa akan:
        - ✅ **Graduate** — Berhasil menyelesaikan studi
        - ⚠️ **Dropout** — Tidak menyelesaikan studi
        - 📚 **Enrolled** — Masih aktif berkuliah

        #### 📊 Performa Model
        | Metrik | Nilai |
        |--------|-------|
        | Accuracy | 78.19% |
        | F1-Score (Weighted) | 77.59% |
        | F1-Score (Macro) | 72.04% |

        #### 🔑 Fitur Paling Berpengaruh
        1. Rasio SKS yang Disetujui
        2. SKS Semester 2 yang Disetujui
        3. Pembayaran Kuliah
        4. Rerata SKS yang Disetujui
        5. Risiko Finansial
        """)

    with col_i2:
        st.markdown("""
        #### 💡 Interpretasi Hasil

        **Skala Risiko Dropout:**
        - 🟢 **P(Dropout) < 30%** → Risiko Rendah
        - 🟡 **P(Dropout) 30–60%** → Risiko Sedang, perlu pemantauan
        - 🔴 **P(Dropout) > 60%** → Risiko Tinggi, perlu intervensi segera

        #### 🏃 Rekomendasi Intervensi Cepat
        Untuk mahasiswa dengan risiko tinggi:
        1. **Finansial**: Program cicilan biaya kuliah, beasiswa darurat
        2. **Akademik**: Tutoring intensif, konseling studi
        3. **Psikologis**: Layanan konseling mahasiswa
        4. **Monitoring**: Evaluasi bulanan oleh wali akademik

        #### ⚠️ Keterbatasan Model
        - Model dilatih pada data historis; kondisi terkini mungkin berbeda
        - Akurasi kelas "Enrolled" lebih rendah (~50%) karena data terbatas
        - Gunakan prediksi sebagai *decision support*, bukan keputusan final
        """)