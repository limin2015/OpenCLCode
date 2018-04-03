##./a.out 48000 10
for((i=65536;i<=524288;i=i*2));  
do   
./vectorAdd $i 10 |tee >> bigres-1.txt
#echo $(expr $i \* 3 + 1);  
done  


