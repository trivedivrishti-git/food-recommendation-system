import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


def load_model_data():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


model_data = load_model_data()


@app.route('/reload', methods=['GET', 'POST'])
def reload_model():
    global model_data
    try:
        model_data = load_model_data()
        return jsonify({"status": "success", "message": "Model reloaded successfully in Flask API!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        dish_name = data.get('dish_name', '')
        diet_filter = data.get('diet', 'All')
        course_filter = data.get('course', 'All')
        max_price = data.get('max_price', 99999)
        top_n = data.get('top_n', 10)

        df = model_data["df"]
        cosine_sim = model_data.get("cosine_sim")

        if cosine_sim is None:
            cosine_sim = model_data.get("similarity_matrix")

        if cosine_sim is None:
            tfidf_mat = model_data.get("tfidf_matrix")
            if tfidf_mat is None and "tfidf_vectorizer" in model_data:
                tags = df['ingredients'].fillna('') + ' ' + df['diet'].fillna('') + ' ' + df['course'].fillna('')
                tfidf_mat = model_data["tfidf_vectorizer"].transform(tags)
            cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)

        if dish_name not in df['name'].values:
            return jsonify({"recommendations": []}), 200

        idx = df[df['name'] == dish_name].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i, score in sim_scores:
            if i == idx:
                continue

            row = df.iloc[i]

            if diet_filter != 'All' and str(row['diet']).lower() != str(diet_filter).lower():
                continue
            if course_filter != 'All' and str(row['course']).lower() != str(course_filter).lower():
                continue
            if float(row['price']) > float(max_price):
                continue

            recommendations.append({
                "name": str(row['name']),
                "diet": str(row['diet']),
                "course": str(row['course']),
                "price": float(row['price']),
                "ingredients": str(row['ingredients']),
                "img_url": str(row['img_url'])
            })

            if len(recommendations) >= top_n:
                break

        return jsonify({"recommendations": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/recommend_history', methods=['POST'])
def recommend_history():
    try:
        data = request.json
        history_dishes = data.get('history_dishes', [])
        top_n = data.get('top_n', 20)

        df = model_data["df"]

        tfidf_mat = model_data.get("tfidf_matrix")
        if tfidf_mat is None and "tfidf_vectorizer" in model_data:
            tags = df['ingredients'].fillna('') + ' ' + df['diet'].fillna('') + ' ' + df['course'].fillna('')
            tfidf_mat = model_data["tfidf_vectorizer"].transform(tags)

        valid_dishes = [d for d in history_dishes if d in df['name'].values]
        if not valid_dishes:
            return jsonify({"recommendations": []}), 200

        indices = df[df['name'].isin(valid_dishes)].index
        user_profile = np.asarray(tfidf_mat[indices].mean(axis=0))

        sim_scores = cosine_similarity(user_profile, tfidf_mat)[0]
        sim_indices = sim_scores.argsort()[::-1]

        recommendations = []
        for i in sim_indices:
            row = df.iloc[i]
            if row['name'] in valid_dishes:
                continue

            recommendations.append({
                "name": str(row['name']),
                "diet": str(row['diet']),
                "course": str(row['course']),
                "price": float(row['price']),
                "ingredients": str(row['ingredients']),
                "img_url": str(row['img_url'])
            })

            if len(recommendations) >= top_n:
                break

        return jsonify({"recommendations": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)