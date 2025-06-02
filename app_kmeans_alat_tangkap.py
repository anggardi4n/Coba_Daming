
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# Data dummy
data = {
    'Wilayah': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
    'Volume_Ikan': [1200, 850, 950, 1500, 700, 400, 1300, 600],
    'Jenis_Alat': ['Jaring Insang', 'Pancing', 'Jaring Insang', 'Jaring Insang', 'Bubu', 'Bubu', 'Pancing', 'Pancing'],
    'Jenis_Ikan': ['Pelagis', 'Demersal', 'Pelagis', 'Pelagis', 'Demersal', 'Campuran', 'Demersal', 'Campuran']
}

df = pd.DataFrame(data)
df['Jenis_Alat_Encoded'] = df['Jenis_Alat'].map({'Jaring Insang':1, 'Pancing':2, 'Bubu':3})
df['Jenis_Ikan_Encoded'] = df['Jenis_Ikan'].map({'Pelagis':1, 'Demersal':2, 'Campuran':3})

# Standarisasi
features = ['Volume_Ikan', 'Jenis_Alat_Encoded', 'Jenis_Ikan_Encoded']
X_scaled = StandardScaler().fit_transform(df[features])

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
score = silhouette_score(X_scaled, df['Cluster'])

# Streamlit UI
st.title("📊 Dashboard K-Means Alat Tangkap Perikanan")
st.markdown("Silakan eksplorasi hasil clustering berdasarkan volume ikan, jenis alat, dan jenis ikan.")

st.subheader("📈 Silhouette Score")
st.write(f"Silhouette Score: **{score:.3f}** (Semakin mendekati 1, semakin baik)")

st.subheader("🧾 Data Cluster")
st.dataframe(df[['Wilayah', 'Volume_Ikan', 'Jenis_Alat', 'Jenis_Ikan', 'Cluster']])

# Visualisasi 2D
st.subheader("🔍 Visualisasi 2D Clustering")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Volume_Ikan', y='Jenis_Alat_Encoded', hue='Cluster', palette='Set1', s=100, ax=ax)
ax.set_ylabel("Jenis Alat (Encoded)")
st.pyplot(fig)

# Visualisasi 3D
st.subheader("📌 Visualisasi 3D Clustering")
fig3d = plt.figure(figsize=(8,6))
ax3d = fig3d.add_subplot(111, projection='3d')
sc = ax3d.scatter(
    df['Volume_Ikan'],
    df['Jenis_Alat_Encoded'],
    df['Jenis_Ikan_Encoded'],
    c=df['Cluster'],
    cmap='Set1',
    s=100
)
ax3d.set_xlabel("Volume Ikan")
ax3d.set_ylabel("Jenis Alat (Encoded)")
ax3d.set_zlabel("Jenis Ikan (Encoded)")
st.pyplot(fig3d)

# Interpretasi
st.subheader("🧠 Ringkasan Setiap Cluster")
for i in range(3):
    cluster_data = df[df['Cluster'] == i]
    st.markdown(f"**Cluster {i}:**")
    st.dataframe(cluster_data[['Wilayah', 'Volume_Ikan', 'Jenis_Alat', 'Jenis_Ikan']])
