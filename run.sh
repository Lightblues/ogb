# 先构造数据集! 
cd tab2graph
python test/convert_dglData.python # 保存图结构到 datasets/tiny/dgl_data_processed

# 运行/调试模型 
cd ogb/examples/nodeproppred/mag
python sample_dgl.py