# TensorFlow Examples
This repository includes TensorFlow example codes for both distributed and non-distributed version. **Contributions are very welcome.**

## Local examples
To run the local examples on Jupyter Notebooks, you can either use caicloud.io directly or run it in docker with caicloud TensorFlow image.

### Use caicloud.io machine learning SaaS
- Step 1. Login into [caicloud.io](https://console.caicloud.io/login). Registry [here](https://console.caicloud.io/reg) if you don't have a caicloud account. After login, you may see something like this
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/homepage.png)

- Step 2. Click on "机器学习" and then click on "单机实验”. You may see something like the picture below if you haven't created one. If you have already created one, you can skip Step 3.
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/create_local.png)

- Step 3. Creat an experiment environment by click “创建单机实验” and fill the required fields.
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/list_view.png)
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/expanded_view.png)

- Step 4. Open Jupyter Notebook
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/jupyter.png)

### Use caicloud TensorFlow docker image
- Step 1. [Install Docker](https://docs.docker.com/engine/installation/)

- Step 2. Pull image

  ```
  docker pull index.caicloud.io/tensorflow:0.8.0
  ```

  Note you need to have a [caicloud account](https://console.caicloud.io/reg) to pull the image.

- Step 3. Start the image

  ```
  docker run --net=host index.caicloud.io/tensorflow:0.8.0
  ```

- Step 4. Access the Jupyter Notebook at ```localhost:8888```


## Distributed examples
Distributed TensorFlow examples could only be run on [caicloud.io](caicloud.io).

- Step 1. Create distributed TensorFlow cluster. This may take a few minutes. Note you'll need to create a kubernetes cluster before deploying a TensorFlow cluster. This [doc](http://www.clipular.com/c/4898024607711232.png?k=8TxxmTwy57gXs7SZ9iVVopscjKg) describes how to create a kubernetes cluster on caicloud.io.
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/dist_creation.png)

- Step 2. Open Jupyter Notebook.

- Step 3. Create a terminal.
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/create_terminal.png)
  ![alt text](https://github.com/caicloud/tensorflow-demo/blob/master/picture/terminal_view.png)

- Step 4. Go into the distrubted examples directory:
 
  ```
  cd /distributed
  ls
  ```

- Step 5. Run examples follow instructions [here](https://github.com/caicloud/tensorflow-demo/blob/master/distributed/README.md)

