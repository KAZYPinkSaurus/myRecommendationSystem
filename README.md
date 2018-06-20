# myRecommendationSystem
- matrix factorization または　non-negative matrix factorizationを用いた推薦システム

## install(mf,nmf)
- pip install learning==0.0.dev0

## use
- 基本
  - python train.py [filepath] [number]
- NMFを使いたい時
  - python train.py [filepath] [number] 'MMF'
  
- データについて
  -ヘッダ付きtsvファイルでお願いします
- [number]は行列分解の際のパラメータ
