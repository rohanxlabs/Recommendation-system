import numpy as np
def recommend_items(user_id,user_item_matrix,model,n=3):
    user_idx = user_item_matrix.index.get_loc(user_id)
    user_vector = model.transform(user_item_matrix)
    scores = np.dot(user_vector[user_idx],model.components_)
    recommendations = scores.argsort()[::-1][:n]
    return user_item_matrix.columns[recommendations].tolist()