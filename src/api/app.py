from fastapi import FastAPI, HTTPException
import pickle 
import pandas as pd
from src.models.recommend import recommend_items

app = FastAPI()

model = pickle.load(open("artifacts/model.pkl","rb"))
data = pd.read_csv("data/processed/train.csv")
matrix = data.pivot_table(index="user_id",columns="item_id",values="rating").fillna(0)

@app.get("/recommend/{user_id}")
def recommend(user_id:int):
    if user_id not in matrix.index:
        raise HTTPException(status_code=404, detail="User not found")
    items = recommend_items(user_id,matrix,model)
    return{"recommended_items":items}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)