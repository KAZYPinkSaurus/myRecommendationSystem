import numpy as np
import random
class MF():
    def __init__(self, k=10, tol=1e-4,max_iter=500,alpha=0.002,gamma=0.0001):
        self.k=k #行列分解幅
        self.tol=tol #収束の基準(誤差の減少率)
        self.max_iter=max_iter #重み更新の最大回数
        self.alpha=alpha #正則化パラメタ
        self.error = 1000000000000 #誤差(初期値は大きな値ならなんでも良い)
        self.ex_error  =100000000000 #前回のワンステップ前の誤差保存用
        self.gamma = gamma #勾配法のステップ幅

    def fit(self, aX):
        if (min(aX.shape[0], aX.shape[1]) < self.k):
            print("size k error")
            exit()
        # Q,Pは学習する行列(初期値は乱数)
        self.X = np.copy(aX)
        self.P = np.random.rand(self.k, self.X.shape[0])
        self.Q = np.random.rand(self.k, self.X.shape[1])
        
        
        # 学習部分
        for i in range(self.max_iter):
            self.__error()
            # 二乗誤差出力
            # print("step:" + str(i) + " 誤差減少率:" + str(1 - (self.error / self.ex_error)))    
            # 誤差の減少率がself.tol以下になったら収束とみなす
            if (1 - self.error/self.ex_error < self.tol):
                print("収束，"+"二乗誤差:"+str(self.error))
                return np.dot(self.P.T, self.Q)
            self.__update()
            self.ex_error = self.error

        print("iter終了，"+"二乗誤差:"+str(self.error))
        return np.dot(self.P.T, self.Q)
    
    # 誤差計算
    def __error(self):
        tPtQ = np.dot(self.P.T, self.Q)
        tEr = self.X - tPtQ
        tEr[np.isnan(tEr)] = 0
        # 誤差
        tEr = np.linalg.norm(tEr)** 2
        # 正則化項
        tReg = self.alpha * (np.linalg.norm(self.P)** 2 + np.linalg.norm(self.Q)** 2)
        
        self.error = tEr + tReg    
        return self.error
        
    # p,qの更新(サイクリックに勾配降下法)
    def __update(self):
        tPtQ = np.dot(self.P.T, self.Q)
        tE = self.X - tPtQ
        for u in range(self.P.shape[1]):
            for i in range(self.Q.shape[1]):
                if(not np.isnan(tE[u,i])):
                    self.P[:, u] += self.gamma * (tE[u, i] * self.Q[:, i] - self.alpha * self.P[:, u])
                    self.Q[:, i] += self.gamma*(tE[u, i] * self.P[:, u] - self.alpha * self.Q[:, i])

