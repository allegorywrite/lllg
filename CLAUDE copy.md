

### ビルド
make -C build -j4
### 評価
cd ~/aist_ws/lg_lacam/ && build/main -i assets/random-32-32-20.map-scen-random/scen-random/random-32-32-20-random-1.scen -m assets/random-32-32-20.map -N 400 -v 3 --lg --lg_window 8