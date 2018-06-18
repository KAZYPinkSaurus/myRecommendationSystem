from learning import matrix_factorization
from learning import non_negative_matrix_factorization
import numpy as np
import pandas as pd
import sys
import time
from collections import Counter



if __name__ == '__main__':
    args = sys.argv
    # 入力 データのpath, 各種パラメータ
    
    #学習するデータのpath
    tDataPath = args[1]
    # tDataPath="./data/ml-100k/u.data"

    # 行列分解する際のパラメタ(defaultは40)
    tK= 40
    tK = int(args[2])

    tMF = True
    if args[3] == 'NMF':
        tMF = False

    # データロード
    df = pd.read_table(tDataPath)
    # print(df)
    user_ids = list(Counter(df["user_id"]).keys())
    user_ids.sort()
    item_ids = list(Counter(df["item_id"]).keys())
    item_ids.sort()

    tX_Frame = pd.DataFrame(columns=item_ids, index=user_ids)
    tTest = {}
    tNum_Test_Data =0
    for i, r in df.iterrows():
        # 1割はテスト用にとっておく
        if (np.random.rand() < 0.1):
            tNum_Test_Data += 1
            if r["user_id"] in tTest.keys():
                tTest[r["user_id"]][r["item_id"]] = r["rating"]
            else:
                tTest[r["user_id"]] = {r["item_id"]: r["rating"]}
            tX_Frame.at[r["user_id"], r["item_id"]] = np.nan
        else:
            tX_Frame.at[r["user_id"], r["item_id"]] = r["rating"]


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

    if tMF == True:

        tMF = matrix_factorization.MF(k=tK,tol=1e-3)
        # 学習
        tStart_mf=time.time()
        tLearned_mf=tMF.fit(tX)
        tEnd_mf=time.time()
        print('MF学習時間:'+str(tEnd_mf-tStart_mf))

        tLearned_mf_Frame=pd.DataFrame(tLearned_mf, columns=item_ids, index=user_ids)

        # テスト誤差計算
        # 辞書に保存しておいた値にアクセス
        tTest_error_mf=0

        for k_u in tTest.keys():
            for k_i in tTest[k_u].keys():
                tTest_error_mf += (tLearned_mf_Frame.at[k_u, k_i] - tTest[k_u][k_i])**2            
        print("test error(mf):" + str(tTest_error_mf))
        print('RMSE(mf):'+str(np.sqrt(tTest_error_mf/tNum_Test_Data)))
        # 値が埋まっている所を小さな値にしておく
        tPredicted_X = tOnes_X * tLearned_mf
    else:
        tNMF = non_negative_matrix_factorization.NMF(k=tK,tol=1e-2)
        # 学習
        tStart_nmf=time.time()
        tLearned_nmf=tNMF.fit(tX)
        tEnd_nmf=time.time()
        print('NMF学習時間:' + str(tEnd_nmf - tStart_nmf))

        tLearned_nmf_Frame=pd.DataFrame(tLearned_nmf, columns=item_ids, index=user_ids)
        # テスト誤差計算
        tTest_error_nmf=0
        for k_u in tTest.keys():
            for k_i in tTest[k_u].keys():
                tTest_error_nmf += (tLearned_nmf_Frame.at[k_u, k_i] - tTest[k_u][k_i])** 2

        print("test error(nmf):" + str(tTest_error_nmf))
        print('RMSE(nmf):' + str(np.sqrt(tTest_error_nmf / tNum_Test_Data)))
        # 値が埋まっている所を小さな値にしておく
        tPredicted_X = tOnes_X * tLearned_nmf
        
    print('テストデータ数:' + str(tNum_Test_Data))
    
    # 出力：推薦されるアイテム
    tColumns = tX_Frame.columns.values
    tRows = tX_Frame.index.values
    for i,j in zip(tPredicted_X.argmax(1),range(tC)):
        print("User["+str(tRows[j])+"]:"+"item["+str(tColumns[i])+"]")


