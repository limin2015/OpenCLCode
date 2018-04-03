##./a.out 48000 10
for((i=512;i<=65536;i=i*2));  
do   
./a.out $i 10 |tee >> res-1.txt
#echo $(expr $i \* 3 + 1);  
done  


