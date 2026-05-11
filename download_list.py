import baostock as bs
import pandas as pd
    
# 登陆系统
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
    
# 获取沪深300成分股
rs = bs.query_hs300_stocks()
print('query_hs300 error_code:'+rs.error_code)
print('query_hs300  error_msg:'+rs.error_msg)
    
# 打印结果集
hs300_stocks = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    hs300_stocks.append(rs.get_row_data())
result = pd.DataFrame(hs300_stocks, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv("hs300_list.csv",index=False)
print(result)

rs2 = bs.query_zz500_stocks()
print('query_zz500 error_code:'+rs2.error_code)
print('query_zz500  error_msg:'+rs2.error_msg)
   
# 打印结果集
zz500_stocks = []
while (rs2.error_code == '0') & rs2.next():
    # 获取一条记录，将记录合并在一起
    zz500_stocks.append(rs2.get_row_data())
result2 = pd.DataFrame(zz500_stocks, columns=rs2.fields)
# 结果集输出到csv文件
result2.to_csv("zz500_list.csv", index=False)
print(result2)
    
# 登出系统
bs.logout()