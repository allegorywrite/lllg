評価によって得られる
comp_time_ms: 102   makespan: 61 (lb=53, ub=1.16)   sum_of_costs: 15536 (lb=8500, ub=1.83)  sum_of_loss: 13532
の値を改善するようにアルゴリズムの修正を行ってください．
計算時間(comp_time_ms)の減少，目的関数(sum_of_costs)の減少が両立する良いアルゴリズムの発見が目的です．

以下の修正が有効な可能性があります．
・LocalGuide::constructにおけるA*のヒューリスティック関数
\Phi_i = \underset{\pi \in \Pi_i}{\arg \min} \sum_{t=1}^{w_i} \text{cost}(\pi[t-1], \pi[t]) + \text{cost}_T(\pi[w_i], g_i)
を変更する．
・現在LocalGuide::constructにおけるupdate_guide_pathをエージェントオーダーに従って行っているが，衝突コストでソートしてA*っぽく衝突コスト高いものから処理する．(衝突コストないものは処理しない)

・１つのアルゴリズムの改善ごとに評価を行い，評価値が改善したもののみ適用してください．
・計算時間を短縮するために事務的な処理(パスの保存等)を削減することはやめてください．目的は計算効率のよい良いアルゴリズムを見つけることです．
・アルゴリズムの修正はできる限りオプション化し，もとの実装が簡単に復旧・再現できるように改変してください．

### ビルド
make -C build -j4
### 評価
cd ~/aist_ws/lg_lacam/ && build/main -i assets/random-32-32-20-random-1.scen -m assets/random-32-32-20.map -N 400 -v 3 --lg --lg_window 8