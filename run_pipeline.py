from src.data.preprocess import preprocess_data
from src.features.build_features import create_user_item_matrix
from src.models.train import train_model , save_model
from src.utils.config import DATA_PATH, MODEL_PATH

train,_=preprocess_data(DATA_PATH)
matrix=create_user_item_matrix(train)
model,_=train_model(matrix)
save_model(model,MODEL_PATH)

print("Pipeline executed successfully")