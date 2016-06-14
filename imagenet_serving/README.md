## [ImageNet](http://www.image-net.org/)线上分类服务器
这里通过一个已经训练好的model来分类新的图片。Caicloud提供了一个已经编译好的镜像```index.caicloud.io/caicloud/inception_serving```。如果想要了解这个镜像是如何编译的，[这里](https://tensorflow.github.io/serving/serving_inception)有详细的介绍。

镜像准备好之后，可以通过serving.json来在kubernetes里启动service：
```
kubectl create -f serving.json
```

当服务在Kubernetes里面建好之后，使用以下命令得到服务端口：
```
kubectl describe svc inception-service
```

我们可以看到类似以下的信息：
```
Name:			inception-service
Namespace:		default
Labels:			<none>
Selector:		worker=inception-pod
Type:			NodePort
IP:				10.254.121.195
Port:			<unset>	9000/TCP
NodePort:		<unset>	32668/TCP
```
其中```NodePort```就是我们需要的端口号，有了端口号，我们还需要知道IP。通过下面的命令可以查到IP。先查看所有节点列表
```
kubectl get nodes
```
可以得到类似下面的信息：
```
NAME         STATUS                     AGE
i-dh4t40ez   Ready                      19d
i-jnr9dxhz   Ready,SchedulingDisabled   19d
i-tiga0i1q   Ready                      19d
```
随便选取一个节点，获取节点IP信息：
```
kubectl describe node i-dh4t40ez
```
可以得到类似如下的结果：
```
Name:			i-dh4t40ez
Labels:			failure-domain.beta.kubernetes.io/region=ac1,failure-domain.beta.kubernetes.io/zone=ac1,kubernetes.io/hostname=i-dh4t40ez
CreationTimestamp:	Thu, 26 May 2016 07:10:59 +0800
Phase:
Conditions:
  Type		Status	LastHeartbeatTime			LastTransitionTime			Reason				Message
  ----		------	-----------------			------------------			------				-------
  OutOfDisk 	False 	Tue, 14 Jun 2016 10:53:07 +0800 	Thu, 26 May 2016 07:10:59 +0800 	KubeletHasSufficientDisk 	kubelet has sufficient disk space available
  Ready 	True 	Tue, 14 Jun 2016 10:53:07 +0800 	Thu, 26 May 2016 07:10:59 +0800 	KubeletReady 			kubelet is posting ready status
Addresses:	180.101.191.78,10.244.1.0
```
其中```Addresses```中给出了外网IP```180.101.191.78```，这样这个图片分类器的服务地址就是```180.101.191.78:32668```

## client端
#### 直接使用Docker镜像
```
docker run -it -v "$PWD"/pic:/pic index.caicloud.io/caicloud/inception_serving 
cd serving
```

#### 在本地编译
1. 根据[文档](https://tensorflow.github.io/serving/setup#prerequisites)安装必要的工具

2. 下载代码并编译client端（第一次编译的时间比较长，需要2-4个小时）：
```
git clone --recurse-submodules https://github.com/tensorflow/serving
cd serving/tensorflow
./configure
cd ..
bazel build -c opt tensorflow_serving/...
```

## 使用client端分类图片
```
bazel-bin/tensorflow_serving/example/inception_client --server=180.101.191.78:32668 --image=/pic/02ea79e4aad9d6275da78a9170fa4e82.jpg
```
参数```server```需要替换成启动的服务器的地址。

运行时有可能会出现超时问题，如果出现此问题，可以修改时间限制的参数：
```
vi tensorflow_serving/example/inception_client.py
```
修改下面超时时限：
```
result = stub.Classify(request, 10.0)  # 10 secs timeout
```

