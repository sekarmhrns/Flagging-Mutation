# DNA Mutation Flagging

## Repository Outline

1. READ.md - Penjelasan gambaran umum project
2. P1M2_Sekar_M.ipynb - Notebook yang berisi project utama, pengolahan data dan modelling.
3. P1M2_Sekar_M_dataset.csv - File CSV yang berisi dataset murni dari kaggle
4. P1M2_Sekar_M_inf.csv - Notebook untuk keperluan Inferencing model 
5. P1M2_Sekar_M_conceptual.txt - File text berisi jawaban dari pertanyaan contextual
6. final_model.pkl - Model akhir yang terpilih diasumsikan menjadi 'Best Model'
7. Flagging_Mutation - Folder berisi keperluan Model Deployment


## Problem Background
DNA sequence analysis merupakan salah satu bidang penting dalam bioinformatika untuk memahami struktur genetik, mendeteksi mutasi, dan memprediksi risiko penyakit. Adanya mutasi pada DNA dapat memengaruhi fungsi biologis suatu organisme, sehingga identifikasi mutasi secara cepat dan akurat sangat dibutuhkan. Dengan memanfaatkan machine learning, kita dapat membangun model klasifikasi untuk memprediksi apakah suatu DNA sequence mengandung mutasi atau tidak, berdasarkan berbagai fitur biologis (GC/AT content, panjang DNA, distribusi basa, dan pola k-mer).

## Project Output
Output project ini yaitu dapat menghasilkan Model Machine Learning yang dapat membantu DNA Laboratory dalam flagging mutasi dalam DNA.

## Data
Dataset ini berisi 3.000 sampel DNA dari berbagai organisme (manusia, bakteri, virus, dan tumbuhan). Setiap sampel dilengkapi dengan informasi biologis seperti komposisi basa, panjang DNA, pola k-mer, serta indikator mutasi. Data ini digunakan untuk menganalisis hubungan karakteristik DNA dengan mutasi, sekaligus membangun model machine learning untuk memprediksi risiko mutasi.

## Method
Machine learning menggunakan metode model supervised learning dengan Classification menggunakan berbagai macam model guna menemukan Model terbaik untuk kasus ini.

## Stacks
| **Kategori**             | **Library / Module**                                | **Fungsi Singkat**                             |
| ------------------------ | --------------------------------------------------- | ---------------------------------------------- |
| Data manipulation & math | `pandas`, `numpy`                                   | Manipulasi data, operasi numerik               |
| Data visualization       | `matplotlib`, `seaborn`                             | Membuat plot, visualisasi statistik            |
| Statistics               | `scipy.stats.kendalltau`                            | Menghitung korelasi Kendall Tau antar variabel |
| Data preprocessing       | `train_test_split`, `cross_validate`                | Membagi dataset, evaluasi model                |
|                          | `StandardScaler`, `OneHotEncoder`, `OrdinalEncoder` | Standarisasi numerik dan encoding kategori     |
| Modeling                 | `ColumnTransformer`, `make_pipeline`                | Pipeline preprocessing + model                 |
|                          | `DecisionTreeClassifier`, `RandomForestClassifier`  | Model berbasis pohon                           |
|                          | `KNeighborsClassifier`                              | Model berbasis jarak                           |
|                          | `SVC`                                               | Support Vector Machine                         |
|                          | `XGBClassifier`                                     | Gradient Boosting (XGBoost)                    |
| Evaluation               | `classification_report`, `accuracy_score`           | Metrik performa klasifikasi                    |
|                          | `confusion_matrix`, `roc_auc_score`                 | Evaluasi model                                 |
| Hyperparameter tuning    | `RandomizedSearchCV`, `randint`, `uniform`          | Pencarian hyperparameter secara acak           |
| Model saving             | `pickle`                                            | Menyimpan dan memuat model ke/dari file        |


## Reference
- [Dataset URL](https://www.kaggle.com/datasets/miadul/dna-classification-dataset?resource=download)
- [Model URL](https://drive.google.com/drive/folders/1biCosTK5Xu2RMD9XMWfi-HfsaZ18N1RJ?usp=drive_link)
- [Deployment URL](https://huggingface.co/spaces/sekarmeuw/Flagging_Mutation)