
参考nvidia blog中的例子https://developer.nvidia.com/blog/cuda-graphs/ ，
测试cuda-graph的性能。

Compile and Run: 

```
make;
./simpleCudaGraphs
```

Performance:


官方blog给出的结果：Tesla V100-SXM2-16G，cuda 10.1

![image](https://user-images.githubusercontent.com/11663212/131632720-bf0e045d-4d04-400e-842a-1ccb97f702e1.png)

自测结果：

Tesla V100-SXM2-16G，cuda 10.1

![image](https://user-images.githubusercontent.com/11663212/131632835-8e5109c7-7fe0-4026-83f1-5337fa8e72af.png)


A100-SXM4-40GB，cuda 11.2

![image](https://user-images.githubusercontent.com/11663212/131632888-299e2aa1-a20c-4df8-a0fe-c511e56907d9.png)

