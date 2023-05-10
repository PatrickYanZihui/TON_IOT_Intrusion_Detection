-------------------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/05/11 01:13
Description ： 
    1. 增加了描述資料集的位置
    2. 增加了該程式目的
-------------------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/10/14 15:49
Description ： 
    1.首次撰寫文檔 項目1和2
    2.補齊了特征表的txt檔
-------------------------------------
1 這個程序的目的
    這是一個基於機器學習的網絡入侵偵測練習。
    資料集為IOT_23的Train_Test_Network.csv
    目前僅完成了二分類的簡單訓練，還沒確定準備性。
    IOT23 官網 https://www.stratosphereips.org/datasets-iot23
2 運行前需要注意的東西
    2.1 Python請安裝64位版本的，32和86的都不行
    2.2 安裝pydot 
        pip install pydot
    2.3 安裝graphwiz
        因為pip install 没用
        從官網下載Windows 64位版本的graphwiz並安裝  https://graphviz.gitlab.io/download/
        複製安裝路徑(如D:\Graphviz\bin）並配置環境變數，新建變數名為Graphwiz,變數值為路徑
        重啟電腦
        詳細步驟 https://blog.csdn.net/QAQIknow/article/details/119188790
    2.4 另外運行程式時報缺的，pip install 對應的東西就行
    2.5 (1.2 & 1.3) 亦可解决報錯(‘You must install pydot (`pip install pydot`) and install graphviz (see...) ‘, ‘for plot_model..
4 運行順序
    00跑到04就好了
5 資料集位置
    5.1  rawDataSet資料夹里的 Train_Test_Network.csv 就是資料集   
    (https://github.com/PatrickYanZihui/TON_IOT_Intrusion_Detection/blob/2d_detection/rawDataSet/Train_Test_Network.csv)
    5.2 00_Prepossesing.py第42行
    df = pd.read_csv('./rawDataSet/Train_Test_Network.csv',header=0)
    就是讀取資料集的地方
