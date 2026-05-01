#买入率
buy=`grep ^2022  test.txt | awk '$3>0{print $0}' | wc -l`
total=`grep ^2022  test.txt | awk '{print $0}' | wc -l`
echo "买入率"
rate=$(echo "$buy $total" | awk '{print $1/$2}')
echo $rate

#买入价差
echo "买入价差"
grep ^2022  test.txt | awk '$3>0{print $0}' | awk '{a+=$3/($(NF-1)*(1+$8))-1;b+=1}END{print a,b,a/b}'
#止盈卖出价差
echo "止盈卖出价差"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep -v 145500000 | awk '{a+=$4/($(NF-1)*(1+$8))-1;b+=1}END{print a,b,a/b}'
#尾盘滑点
echo "尾盘滑点"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep  145500000 | awk '{a+=($4/$12-1);b+=1}END{print a,b,a/b}'
#尾盘自然收益
echo "尾盘自然卖出收益"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep  145500000 | awk '{a+=($4/$3-1);b+=1}END{print a,b,a/b}'
#止盈滑点
echo "止盈滑点"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep -v 145500000 | awk '{a+=($4/$12-1);b+=1}END{print a,b,a/b}'
#止盈平均收益
echo "止盈收益"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep -v 145500000 | awk '{a+=($4/$3-1);b+=1}END{print a,b,a/b}'
#止损滑点
echo "止损滑点"
grep ^2022  test.txt | awk '$3>0&&($5/$3-1)>-1&&($6/$3-1)<-1{print $0}' | awk '{a+=($4/$12-1);b+=1}END{print a,b,a/b}'
grep ^2022  test.txt | awk '$3>0&&($5/$3-1)<=-1&&($6/$5-1)<-0.0001{print $0}' | awk '{a+=($4/$12-1);b+=1}END{print a,b,a/b}'
#部成期望
echo "部成期望"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep 部成 | awk '{a+=($7/($(NF-1)*(1+$8))-1);b+=1}END{print a,b,a/b}'
#未成期望
echo "未成期望"
grep ^2022  test.txt | awk '$3==0{print $0}' | awk '{a+=($7/($(NF-1)*(1+$8))-1);b+=1}END{print a,b,a/b}'
#全成期望
echo "全成期望"
grep ^2022  test.txt | awk '$3>0{print $0}' | grep -v 部成 | awk '{a+=($7/($(NF-1)*(1+$8))-1);b+=1}END{print a,b,a/b}'
