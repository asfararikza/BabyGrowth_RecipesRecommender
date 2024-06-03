from flask import Flask, request, jsonify
from recommendationSystem import new_df, recommend_recipe

# Initialize Flask
app = Flask(__name__)

# Endpoint for getting recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    id_resep = request.args.get('id_resep')
    if not id_resep:
        return jsonify({'error': 'Id resep tidak diberikan'}), 400

    if id_resep not in new_df['id_resep'].values:
        return jsonify({'error': 'Id resep tidak ditemukan'}), 404

    rekomendasi = recommend_recipe(id_resep)
    if isinstance(rekomendasi, str):
        return jsonify({'error': rekomendasi}), 404
    
    # Convert DataFrame to dictionary
    rekomendasi_dict = rekomendasi.to_dict(orient='records')
    return jsonify({'rekomendasi': rekomendasi_dict})

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
