1. 首先建立Linux系統 (不要用wsl,docker或VM,會繞很多彎路= = 主要是找不到顯卡的問題 還有carla渲染用的不是opengl而是vulkan, 而這東西裝在wsl上會有一堆dependency問題)

2. 按照這個網站（https://blog.csdn.net/shitiezhu/article/details/133237205）去下載carla
我是安裝0.9.15 可以下這個指令下載： wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
然後tar -xzvf CARLA_0.9.15.tar.gz -C /path/to/your/dist

3. 安裝好後就可以開始產生資料了 (預設當前路徑都在CarlaUE4.sh這層)

conda activate carla

3.1 啟用carla
./CarlaUE4.sh -RenderOffScreen (如果有接到nvidia的話也可以不加->檢查是否有接到顯卡可輸入nvidia-smi)
./CarlaUE4.sh -quality-level=Low --no-rendering
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
3.2 Load map
# town12是10*10的大城市 相當於新竹市大小 適合模擬整個城市的運作 但需要額外找地圖
python ~/project/PythonAPI/util/config.py --map Town05
python ~/project/PythonAPI/util/config.py --map Town12
3.3 Generate experiment scenes
進入我們自己的project folder (以我自己為例就是/BV)
python main.py 
**注意: 程式偶爾會崩潰，別在意，重開就是了。可以改變loop index來繼續崩潰之前建立到的folder編號**
3.4. 將四個不同的相機看到的點雲景象merge在一起
python mergeply_ego.py
This step is to merge 4 point cloud from 4 cameras into one point cloud. Also, add self point cloud model into point cloud because the camera can know scan self car body.

4. Test the ROI point cloud registration algorithm
4.1 Algorithm Exp
python algorithm_exp.py
The file is the main exp code of my ROI point cloud registration algorithm.
Also can try python algorithm_exp_network.py for multiple car scenario.

---------------------------------------------------------------------------
開啟copytranslator: .~/下載/copytranslator-9.1.0.AppImage --no-sandbox
每秒檢查一次nvidia-smi: watch -n 1 -d nvidia-smi
---------------------------------------------------------------------------
# 踩的坑
1. carla 0.9.15 只接受python 3.7, 建議去檢查python版本, 然後強烈建議使用conda虛擬環境以防出錯= =
2. 如果你不小心在base就安裝到更高的python (雖然我不覺得有別人會跟我一樣蠢到直接去刪python資料包),可以參考https://blog.csdn.net/qq_57162593/article/details/118533962
3. Carla已經不能apt update了，如果出現:
---------------------------------------------------------------------------
Err:5 http://dist.carla.org/carla-0.9.15 all InRelease
  Could not connect to dist.carla.org:80 (34.227.255.250), connection timed out
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
All packages are up to date.
W: Failed to fetch http://dist.carla.org/carla-0.9.15/dists/all/InRelease  Could not connect to dist.carla.org:80 (34.227.255.250), connection timed out
W: Some index files failed to download. They have been ignored, or old ones used instead.
---------------------------------------------------------------------------
目前查的方式都沒辦法解決，有說可能是伺服器本身就關掉了，去source.list也沒看到需要改的，總之就是無解
4. 如果出現"Disabling core dumps."別在意，沒跳畫面出來前都是小事。沒跳畫面或畫面很卡大概就是wsl在搞了。
5. Load Map 如果出現 "time-out of 10000ms while waiting for the simulator, make sure the simulator is ready and connected to localhost:2000"
應該是沒對到port口，看看是不是docker在背後運作


後續的常用指令
conda activate carla
cd ~/project/bv
python algorithm_exp.py
python algorithm_bv.py