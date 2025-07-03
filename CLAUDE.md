評価によって得られる
comp_time_ms: 102   makespan: 61 (lb=53, ub=1.16)   sum_of_costs: 15536 (lb=8500, ub=1.83)  sum_of_loss: 13532
の値を改善するようにアルゴリズムの修正を行ってください．
計算時間(comp_time_ms)の減少，目的関数(sum_of_costs)の減少が両立する良いアルゴリズムの発見が目的です．

現在lacam/src/local_guide.cppの実装を行っています．以下の点を改善する必要があります．

・現在は毎ステップA*探索によってupdate_guideを行っているが，kステップ(< window size)ごとにlocal guidance作ることで処理時間を削減する．

・１つのアルゴリズムの改善ごとに評価を行い，評価値が改善したもののみ適用してください．
・計算時間を短縮するために事務的な処理(パスの保存等)を削減することはやめてください．目的は計算効率のよい良いアルゴリズムを見つけることです．
・アルゴリズムの修正はできる限りオプション化し，もとの実装が簡単に復旧・再現できるように改変してください．
・行った実装の意図とコードとの対応をanydocgen/docsにmdファイルでまとめて下さい．
・--lg_readonly_parallel_updateで実行できます．


### ビルド
make -C build -j4
### 評価
cd ~/aist_ws/lg_lacam/ && build/main -i assets/random-32-32-20.map-scen-random/scen-random/random-32-32-20-random-1.scen -m assets/random-32-32-20.map -N 400 -v 3 --lg --lg_window 8