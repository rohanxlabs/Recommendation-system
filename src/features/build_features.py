import pandas as pd
def create_user_item_matrix(df):
    return df.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating"
    ).fillna(0)