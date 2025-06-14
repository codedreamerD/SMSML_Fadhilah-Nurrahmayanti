import pandas as pd

def preprocess_and_classify_student_data(input_filepath, output_filepath):
    # Membaca dataset dari file input
    student_data = pd.read_csv(input_filepath)

    # Menghitung skor rata-rata dari tiga mata pelajaran
    student_data['average_score'] = student_data[['math score', 'reading score', 'writing score']].mean(axis=1)

    # Membuat kolom klasifikasi performa berdasarkan skor rata-rata
    def assign_performance_level(score):
        if score >= 85:
            return 'A'
        elif score >= 75:
            return 'B'
        elif score >= 60:
            return 'C'
        elif score >= 50:
            return 'D'
        else:
            return 'E'

    student_data['performance_level'] = student_data['average_score'].apply(assign_performance_level)

    # Melakukan one-hot encoding untuk kolom-kolom kategori
    encoded_data = pd.get_dummies(student_data, columns=[
        'gender',
        'race/ethnicity',
        'parental level of education',
        'lunch',
        'test preparation course'
    ])

    # Deteksi dan hapus outlier langsung tanpa fungsi terpisah
    numeric_columns = encoded_data.select_dtypes(include=['int64', 'float64']).columns

    # Iterasi untuk setiap kolom numerik
    outlier_indices = []
    for column in numeric_columns:
        Q1 = encoded_data[column].quantile(0.25)
        Q3 = encoded_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Temukan dan simpan indeks outlier
        outlier_indices.extend(encoded_data[(encoded_data[column] < lower_bound) | (encoded_data[column] > upper_bound)].index)

    # Menghapus outlier
    encoded_data = encoded_data.drop(index=set(outlier_indices))

    # Menyimpan data yang telah diproses ke file output
    encoded_data.to_csv(output_filepath, index=False)
    print(f"Preprocessed data has been saved to: {output_filepath}")

if __name__ == "__main__":
    # Tentukan path file input dan output
    input_csv_path = "Eksperimen_SML_Fadhilah-Nurrahmayanti/dataset_raw/StudentsPerformance_raw.csv"
    output_csv_path = "Eksperimen_SML_Fadhilah-Nurrahmayanti/preprocessing/dataset_preprocessed/processed_student_automated.csv"

    preprocess_and_classify_student_data(input_csv_path, output_csv_path)
