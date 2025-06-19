評価によって得られる
comp_time_ms: 102   makespan: 61 (lb=53, ub=1.16)   sum_of_costs: 15536 (lb=8500, ub=1.83)  sum_of_loss: 13532
の値を改善するようにアルゴリズムの修正を行ってください．
計算時間(comp_time_ms)の減少，目的関数(sum_of_costs)の減少が両立する良いアルゴリズムの発見が目的です．

以下の修正が有効な可能性があります．
・現在lacam/src/local_guide.cppのENABLE_READONLY_PARALLEL_UPDATEを実装中です．
clearPathを全エージェント分行ったあとでupdate_guide_pathを行っても空のCollision Tableに対して読み取るだけなので
1. CTに対し読み取り専用でupdate_guide_pathを並列実行
2. updateする前の各エージェントにおけるguide_pathを使ってclearPath(直列実行)
3. updateした各エージェントにおけるguide_pathを使ってenrollPath(直列実装)
(自身の前ステップのguide_pathが含まれた状態でupdate_guide_pathしてしまう問題(自身のpathとのcollisionを考えてしまう)問題は別途対処する)
の順番で実装してください

・１つのアルゴリズムの改善ごとに評価を行い，評価値が改善したもののみ適用してください．
・計算時間を短縮するために事務的な処理(パスの保存等)を削減することはやめてください．目的は計算効率のよい良いアルゴリズムを見つけることです．
・アルゴリズムの修正はできる限りオプション化し，もとの実装が簡単に復旧・再現できるように改変してください．

### ビルド
make -C build -j4
### 評価
cd ~/aist_ws/lg_lacam/ && build/main -i assets/random-32-32-20.map-scen-random/scen-random/random-32-32-20-random-1.scen -m assets/random-32-32-20.map -N 400 -v 3 --lg --lg_window 8