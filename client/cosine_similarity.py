import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # Use scikit-learn for efficient cosine similarity calculation

def calculate_similarity(query_embedding, database_embeddings):
    similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), database_embeddings) # query_embedding [128,] -> [1, 128]
    return similarity_scores.flatten()  # Flatten to a 1D array

def find_best_matches(similarity_scores, image_names, threshold=0.8, top_k=5):
    """
    Finds the best matches based on similarity scores and a threshold.

    Args:
        similarity_scores: Cosine similarity scores between the query image and database images.
        image_names: List of image file names corresponding to the database.
        threshold (float, optional): Cosine similarity threshold for a match. Defaults to 0.8.
        top_k (int, optional): Number of top matches to return (even if below threshold). Defaults to 5.

    Returns:
        tuple: (List of tuples (image name, score) of the best matches, bool indicating if match is found)
    """
    best_matches_indices = np.argsort(similarity_scores)[::-1]
    best_matches_indices = best_matches_indices[:top_k]

    best_matches = [(image_names[i], similarity_scores[i]) for i in best_matches_indices if similarity_scores[i] >= threshold]

    if not best_matches:
        best_matches = [(image_names[i], similarity_scores[i]) for i in best_matches_indices]


    match_found = any(score >= threshold for _, score in best_matches)
    return best_matches, match_found