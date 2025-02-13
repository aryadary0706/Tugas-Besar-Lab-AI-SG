import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import json
import os

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

with open("tomato_diseases_description.json", "r") as f:
    disease_info = json.load(f)
dir_model = r"cnn-tomato_disease-20250205.pth"
dir_images = r"informasi_penyakit_image"
model = SimpleCNN(num_classes=10)
checkpoint = torch.load(dir_model, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image_size = (224, 224)
test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['bacterial_spot', 'early_blight', 'late_blight', 'leaf_mold', 'septoria_leaf_spot', 
               'spider_mites', 'target_spot', 'yellow_leaf_curl_virus', 'mosaic_virus', 'healthy']

st.sidebar.title("Dashboard")
page = st.sidebar.radio("Pilih Menu", ["Welcome", "Klasifikasi", "Informasi Penyakit", "About"])

if page == "Welcome":
    st.title("Scanomato 🔍🍅")
    st.header("👋 Selamat Datang di Aplikasi Pendeteksi Penyakit Tanaman Tomat!")
    st.write("Aplikasi yang bertujuan untuk melakukan klasifikasi jenis penyakit yang ada pada suatu tanaman tomat berdasarkan gambar daunnya.")
    st.write("Contoh:")
    st.image(r"display_images\display_perbandingan.png", width=3000)
    st.header("Model yang digunakan!")
    st.write("Aplikasi ini menggunakan model CNN sederhana dengan arsitektur:")
    st.image(r"display_images\cnn_model.png")
    st.write("Silakan pilih menu di samping untuk melakukan klasifikasi, melihat informasi penyakit yang dapat dideteksi, dan informasi tentang aplikasi ini.")
    
elif page == "Klasifikasi":
    st.title("🔍 Klasifikasi Penyakit!")
    classification_type = st.selectbox("Pilih Jenis Pendeteksi", ["Pilih Jenis Klasifikasi", "Satuan", "Banyak"])

    if classification_type == "Pilih Jenis Klasifikasi":
        st.write("Silakan pilih jenis klasifikasi terlebih dahulu.")
    elif classification_type == "Satuan":
        uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang Diunggah", width=300)
            
            if image.size != (224, 224):
                image = test_transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = model(image)
                _, predicted_idx = torch.max(output, 1)
                predicted_label = class_names[predicted_idx.item()]
                confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx.item()].item()
            
            st.write(f"**Hasil Diagnosa:** {predicted_label.replace('_', ' ').title()}")
            st.write(f"**Probabilitas Prediksi:** {confidence * 100:.2f}%")
            
            info = disease_info.get(predicted_label, {"nama_lain": "Tidak ada informasi","deskripsi": "Informasi tidak tersedia", "treatment": "Tidak ada rekomendasi"})
            st.subheader("Informasi Penyakit:")
            st.write(f"📌 **Nama Lain:** {info['nama_lain']}")
            st.write(f"📌 **Deskripsi:** {info['deskripsi']}")
            st.write(f"💊 **Pengobatan:** {info['penanganan']}")

            if confidence < 0.6:
                st.error("Kepercayaan model kurang dari 60%. Silakan unggah ulang gambar!")

    else:
        uploaded_files = st.file_uploader("Upload Gambar (Batch)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
        
        if uploaded_files:
            results = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                if image.size != (224, 224):
                    image = test_transform(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(image)
                    _, predicted_idx = torch.max(output, 1)
                    predicted_label = class_names[predicted_idx.item()]
                    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_idx.item()].item()
                
                results.append([uploaded_file.name, predicted_label, confidence])
            
            import pandas as pd
            df = pd.DataFrame(results, columns=["Nama File", "Label", "Confidence"])
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "classification_results.csv", "text/csv", key="download-csv")

elif page == "Informasi Penyakit":
    st.title("🩺 Informasi Penyakit")
    st.write("Berikut adalah penjelasan dari tiap hasil klasifikasi yang mungkin dari aplikasi ini:")
    
    for key, value in disease_info.items():
        image_path = os.path.join(dir_images, f"{key}.jpg")

        if not os.path.exists(image_path):
            image_path = "https://via.placeholder.com/150"

        st.header(f"{key.replace('_', ' ').title()}")
        st.image(image_path, caption=key.replace('_', ' ').title(), width=250)

        st.markdown(f"""
        **{key.replace('_', ' ').title()}**  
        - **Nama Lain:** {value.get("nama_lain", "Tidak tersedia")}  
        - **Deskripsi:** {value.get("deskripsi", "Tidak tersedia")}  
        - **Penanganan:** {value.get("penanganan", "Tidak tersedia")}  
        """)

elif page == "About":
    st.title("📌 About the App")
    st.markdown("""
    Aplikasi ini dibuat untuk memenuhi tugas besar **Lab AI: Study Group** di **Telkom University**.
    """)
    
    st.write("Stomato merupakan hasil dari tugas besar kelompok 2, diawali dengan tujuan memenuhi Sustainable Development Goals, yaitu:")
    st.markdown("""
    - **SDG 2**: Zero Hunger
    - **SDG 8**: Decent Work and Economic Growth
    - **SDG 12**: Responsible Consumption and Production
    """)
 
    st.header("👥 Kelompok 2 - Developers")
    developers = [
        "M. Rifqi Dzaky Azhad",
        "Hauzan Rafi Attallah",
        "Fransiskus Harris Berliandu",
        "Kemas M Aryadary Rasyad",
        "Yoanes Bosko Galih Suripramesthi"
    ]

    for name in developers:
        st.markdown(f"- {name}")

    st.markdown("---")
    st.write("🔗 [GitHub](https://github.com/mrda)")