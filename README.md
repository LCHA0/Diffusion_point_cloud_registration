# Diffusion_point_cloud_registration

将测试数据放在my_dataset/group1文件夹中（需自己创建），cad转点云结果放在my_dataset/group1/model，扫描点云结果放在my_dataset/group1/src。  
test命令：python my_test.py  
会产生mytest_results_T5.pth用于可视化。  
可视化命令：python visualize_newdata.py --res ./results/mytest_results_T5.pth --idx 0
