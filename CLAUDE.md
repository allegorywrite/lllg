現在benchmark/plot_individual_maps.pyを実装していますが，以下の様なplotモードオプションを実装してください．
・縦軸flowtime/LB, 横軸シナリオ番号で，ヒストグラムの形式で可視化する．
・異なる手法は同じシナリオ位置に重ねて棒グラフ描画し，flowtime/LBが小さいほうが手前に来る様にすることで
隠れるのを防ぐ．

benchmark/plot_individual_maps.pyの実行方法はbenchmark/README.mdを見てください．

### ビルド
make -C build -j4
### 評価
cd ~/aist_ws/lg_lacam/ && build/main -i assets/random-32-32-20.map-scen-random/scen-random/random-32-32-20-random-1.scen -m assets/random-32-32-20.map -N 400 -v 3 --lg --lg_window 8