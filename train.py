from learning import matrix_factorization
from learning import non_negative_matrix_factorization
import numpy as np
import pandas as pd
from collections import Counter

# 入力 データのpath, 各種パラメータ
tPath="./data/ml-100k/"
# データロード
df = pd.read_table(tPath + "u.data")
# print(df)
user_ids = list(Counter(df["user_id"]).keys())
user_ids.sort()
item_ids = list(Counter(df["item_id"]).keys())
item_ids.sort()

tX_Frame = pd.DataFrame(columns=item_ids, index=user_ids)


for i, r in df.iterrows():
    tX_Frame.at[r["user_id"],r["item_id"]] = r["rating"]

tX_Frame= tX_Frame.fillna(np.nan)
tX = tX_Frame.values
# print(tX)

# 値が埋まっていないところを1(値が入っているところは小さい値)とする行列作成
tOnes_X = np.copy(tX) * -10000
# print(tOnes_X)
tOnes_X[np.isnan(tOnes_X)] = 1
# print(tOnes_X)

# 行数，列数
tC = tX.shape[0]
tR = tX.shape[1]
tMin = min(tC, tR)

# k=40(仮)
tMF = matrix_factorization.MF(k=40)
# 学習
tLearned = tMF.fit(tX)
# 値が埋まっている所を小さな値にしておく
tPredicted_X = tOnes_X * tLearned

# 出力：推薦されるアイテム
tColumns = tX_Frame.columns.values
tRows = tX_Frame.index.values
for i,j in zip(tPredicted_X.argmax(1),range(tC)):
    print("User["+str(tRows[j])+"]:"+"item["+str(tColumns[i])+"]")


