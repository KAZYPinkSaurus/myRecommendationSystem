import numpy as np
import random
class MF():
    def __init__(self, k=10, tol=1e-3,max_iter=100,alpha=0.002,gamma=0.0001):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
        self.alpha=alpha
        self.error = 100000 #大きな値ならなんでも良い
        self.gamma = gamma

    def fit(self,aX):
        # Q,Pは学習する行列(初期値は乱数)
        self.X =   np.copy(aX)
        self.P = np.random.rand(self.k, self.X.shape[0])
        self.Q = np.random.rand(self.k, self.X.shape[1])
        
        # 学習部分
        for i in range(self.max_iter):
            self.__error()
            # 二乗誤差出力
            # print("step:"+str(i)+" error:"+str(self.error))
            if (self.error < self.tol):
                print("収束，"+"二乗誤差:"+str(self.error))
                return np.dot(self.P.T, self.Q)
            self.__update()

        print("iter終了，"+"二乗誤差:"+str(self.error))
        return np.dot(self.P.T, self.Q)
    
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
        
    # p,qの更新
    def __update(self):
        tPtQ = np.dot(self.P.T, self.Q)
        tE = self.X - tPtQ
        for u in range(self.P.shape[1]):
            for i in range(self.Q.shape[1]):
                if(not np.isnan(tE[u,i])):
                    self.P[:, u] += self.gamma * (tE[u, i] * self.Q[:, i] - self.alpha * self.P[:, u])
                    self.Q[:, i] += self.gamma*(tE[u, i] * self.P[:, u] - self.alpha * self.Q[:, i])

