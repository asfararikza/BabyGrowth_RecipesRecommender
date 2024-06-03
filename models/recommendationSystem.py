import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
url_dataset = 'https://raw.githubusercontent.com/asfararikza/BabyGrowth_RecipesRecommender/main/Data/dataset_rekomen.csv'
new_df = pd.read_csv(url_dataset)

url_complete_dataset = 'https://raw.githubusercontent.com/asfararikza/BabyGrowth_RecipesRecommender/main/Data/recipe_with_nutritions.csv'
complete_df = pd.read_csv(url_complete_dataset)

# Convert data into vectors
cv = CountVectorizer()
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

def recommend_recipe(id_resep, top_n=5):
    # Ensure the recipe ID exists in the DataFrame
    if id_resep not in new_df['id_resep'].values:
        return "Resep tidak ditemukan"

    # Get the recipe index
    resep_idx = new_df.index[new_df['id_resep'] == id_resep].tolist()
    if not resep_idx:
        return "Index resep tidak ditemukan"

    resep_idx = int(resep_idx[0])

    # Check if the index is within the bounds of the similarity matrix
    if resep_idx >= len(similarity):
        return "Movie index is out of bounds in the similarity matrix."

    # Get the similarity scores for the selected recipe with all recipes
    scores = list(enumerate(similarity[resep_idx]))

    # Sort the recipes based on the similarity scores
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar recipes (excluding itself)
    top_recipe_scores = sorted_scores[1:top_n+1]

    # Retrieve the recipe IDs using their index
    recipe_indices = [i[0] for i in top_recipe_scores]
    recommended_recipe = complete_df.iloc[recipe_indices][['id_resep', 'nama_resep','gambar','porsi','kategori','kalori','protein','lemak','karbo']]

    # Return the recommended recipe IDs along with their similarity scores
    recommendations = recommended_recipe.copy()
    recommendations['similarity_score'] = [round(score[1], 3) for score in top_recipe_scores]
    return recommendations


