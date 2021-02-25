# anomaly-detection-AI-for-Industry
針對圖像進行異常點檢測，檢測出非正常圖片
應用場景為Industry AI(IAI)

檢測步驟
step1: 進行圖像預處理，計算Grayscale。
step2: 進行直方圖均衡化/二值化/邊緣檢測等OpenCV圖像處理
step3: 再藉由邊緣區域擷取進行異常點訓練(LOF, iForest, OneClass SVM等)
step4: 預測圖像為正常與否



