##./a.out 48000 10
for((i=1048576;i<=10,73741824;i=i*2));  
do   
./vectorAdd $i 10 |tee >> bigbigres.txt
#echo $(expr $i \* 3 + 1);  
done  


