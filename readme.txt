-------------------------------------
Modified by : Patrick Yan
Modified DateTime : 2022/10/14 15:49
Description ： 
    1.首次撰寫文檔 項目1和2
    2.補齊了特征表的txt檔
-------------------------------------

1 運行前需要注意的東西
    1.1 Python請安裝64位版本的，32和86的都不行
    1.2 安裝pydot 
        pip install pydot
    1.3 安裝graphwiz
        因為pip install 没用
        從官網下載Windows 64位版本的graphwiz並安裝  https://graphviz.gitlab.io/download/
        複製安裝路徑(如D:\Graphviz\bin）並配置環境變數，新建變數名為Graphwiz,變數值為路徑
        重啟電腦
        詳細步驟 https://blog.csdn.net/QAQIknow/article/details/119188790
    1.4 另外運行程式時報缺的，pip install 對應的東西就行
    1.5 (1.2 & 1.3) 亦可解决報錯(‘You must install pydot (`pip install pydot`) and install graphviz (see...) ‘, ‘for plot_model..
2 運行順序
    00跑到04就好了