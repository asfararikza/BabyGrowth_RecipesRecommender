from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url_dataset='https://raw.githubusercontent.com/asfararikza/BabyGrowth_RecipesRecommender/main/dataset_rekomen.csv'
new_df = pd.read_csv(url_dataset)

# Convert data dalam bentuk vector
cv = CountVectorizer()
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

def recommend_recipe(id_resep, top_n=5):
    # Memastikan id resep ada di DataFrame
    if id_resep not in new_df['id_resep'].values:
        return "Resep tidak ditemukan"

    # Mengambil index resep
    resep_idx = new_df.index[new_df['id_resep'] == id_resep].tolist()
    if not resep_idx:
        return "Index resep tidak ditemukan"

    resep_idx = int(resep_idx[0])  # Convert index to integer and handle the first item in case of duplicates

    # Check if the index is within the bounds of the similarity matrix
    if resep_idx >= len(similarity):
        return "Movie index is out of bounds in the similarity matrix."

    # Get the similarity scores for the selected recipe with all recipes
    scores = list(enumerate(similarity[resep_idx]))

    # Sort the recipes based on the similarity scores
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar recipes (excluding itself)
    top_recipe_scores = sorted_scores[1:top_n+1]

    # Retrieve the id recipe using their index
    recipe_indices = [i[0] for i in top_recipe_scores]
    recommended_recipe = df_merged.iloc[recipe_indices][['id_resep','nama_resep','kategori']]

    # Return the recommended id recipe along with their similarity scores
    recommendations = recommended_recipe.copy()
    recommendations['similarity_score'] = [round(score[1], 3) for score in top_recipe_scores]
    return recommendations

# Inisialisasi Flask
app = Flask(__name__)

# Endpoint untuk mendapatkan rekomendasi
@app.route('/recommend', methods=['GET'])
def recommend():
    id_resep = request.args.get('id_resep')
    if not id_resep:
        return jsonify({'error': 'Id resep tidak diberikan'}), 400

    if id_resep not in new_df['id_resep'].values:
        return jsonify({'error': 'Id resep tidak ditemukan'}), 404

    rekomendasi = recommend_recipe(id_resep)
    return jsonify({'rekomendasi': rekomendasi})

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)