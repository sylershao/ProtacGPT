# import pickle
# import pandas as pd
# f = open('store.pkl','rb')
# data = pickle.load(f)
# pd.set_option('display.width',None)
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_colwidth',None)
# print(data)
# inf=str(data)
# ft = open('store1.csv', 'w')
# ft.write(inf)

import pandas as pd                         #导入pandas包
data = pd.read_csv("/home/xt/T-KG/AMDE/Data/Drop_BBBP-1/KGemb-BBBP-train(1600)-1.csv")           	#读取csv文件)
# df=data.drop(['0'],axis=1)
print(data.shape)                                 #打印所有文件


# import pandas as pd
# df = pd.read_csv('/home/xt/下载/AMDE/Data/BBBPsmiles_label-epoch50/BBBPsmiles_label_val(203)_epoch50.csv',header=None,names=['smiles','label'])
# df.to_csv('/home/xt/下载/AMDE/Data/BBBPsmiles_label-epoch50/BBBPsmiles_label_val(203)_epoch50.csv',index=False)
# #

