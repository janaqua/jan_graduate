============= OptimizationMethods ==============
simlation中會用到的各algorithm的程式

===============================================
==========按執行順序介紹各檔案 ================
===============================================


==============GenerateSimulationData.py =====================
生成模擬用的資料集存在simulation_data_fix_param 資料夾中

==============SGD.py ========================================
用SGD配適各資料集，並紀錄表現 會存成一個.csv 在ToConcatenate_fix_param資料夾下

==============ASGDandSGD_with_GammaDecay.py =================
用SGD with gamma_n decayed 以及 ASGD with gamma_n decayed配適各資料集，並紀錄表現 會存成一個.csv 在ToConcatenate_fix_param資料夾下

==============Batch_SGD.py ==================================
用Batch_SGD配適各資料集，並紀錄表現 會存成一個.csv 在ToConcatenate_fix_param資料夾下

==============Batch_ASGDandSGD_with_GammaDecay.py ===========
用Batch SGD with gamma_n decayed 以及Batch ASGD with gamma_n decayed配適各資料集，並紀錄表現 會存成一個.csv 在ToConcatenate_fix_param資料夾下

