# Diffusion_point_cloud_registration

将测试数据放在my_dataset/group1文件夹中（需自己创建），cad转点云结果放在my_dataset/group1/model，扫描点云结果放在my_dataset/group1/src。  
test命令：python my_test.py  
会产生mytest_results_T5.pth用于可视化。  
可视化命令：python visualize_newdata.py --res ./results/mytest_results_T5.pth --idx 0

2025.10.16
现在直接看open3d文件夹，final.py配准程序，会把转换矩阵保存在txt里，visualize.py是可视化代码，scanpoints.py是对cad点云取ROI的代码

2026.2.22
还是看open3d文件夹，ransac是配准程序，fitting是焊缝拟合程序
