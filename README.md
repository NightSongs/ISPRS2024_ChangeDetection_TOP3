# 基于基础模型的高分辨率遥感影像变化检测方法

这个仓库是earth-insights团队在[ISPRS2024第一技术委员会多模态遥感应用算法智能解译大赛](https://www.gaofen-challenge.com/challenge)中的解决方案，我们的决赛结果在所有团队的全部提交中精度排名第一，总成绩排名第三。

## &#x1F3AC; 开始

### :one: 环境安装
我们在比赛中使用Python 3.8和Pytorch1.8.1， 您可以使用pip install -rrequirements.txt安装环境。

### :two: 预训练权重下载

我们使用CLIP方法预训练的ConvNeXt模型，以convnext-large为例，预训练权重下载地址为https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/tree/main. 您可以将它们下载(.bin文件)并手动存放于pretrain目录下面。

您也可以使用百度网盘下载我们使用到的预训练权重：
链接：https://pan.baidu.com/s/13DVT5JIFPPd7yCwWeUZyuQ?pwd=wj92 
提取码：wj92 

### :three: 数据集路径

您可以依照自己的需求任意的指定数据集的存放路径，并在train_all_data.py或其他train代码中修改数据集路径。例如以下路径：

```
dataset_tmp
├── T1
│   ├── xxx第二期影像.tif
│   
├── T2
|   ├── xxx第三期影像.tif
|
├── gt
|   ├── xxx.tif

```

## &#x2699; 训练

#### 模型训练

在比赛中，我们实验了五折交叉验证和全部数据训练两种方式：

train.py 👉 仅训练五折中的第一折

train_kfold.py 👉  五折训练

train_all_data.py 👉  不划分训练和测试集，直接训练全部的数据（保存train loss最低的权重）

⚠️在本次比赛中，我们强烈建议您使用train_all_data.py，实验表明训练全部数据会有明显的性能提升。您可以使用以下代码调用train.sh实现模型训练：

```bash
# Run the training script
bash code/train.sh
```

#### 多模型集成

我们同样探索了多模型集成策略，实验表明，使用hrnet w48和convnext large融合后可以取得较高的精度。您可以通过修改训练代码中的load_model方法来更改需要训练的模型

您可以尝试更多的backbone，只要它们被0.6.12版本中的timm库支持。您可以使用tools/get_timm_list.py来模糊查找支持的backbone。

#### 权重下载

我们将本次比赛使用的权重上传至百度网盘，您可以直接下载它们进行推理：

链接：https://pan.baidu.com/s/1q2D81M6cdNGysOd6zWTx3A?pwd=ycv4 
提取码：ycv4 

## &#x1F9EA; 本地测试&提交

```bash
# Run the testing script (docker)
python run.py /input_path /output_path
# Run the testing script (local)
python infer.py
```

## &#x1F9CA; 后处理

后处理在本次比赛中能够带来微小的涨点，我们使用了去除离群图斑来优化结果。但是该策略在决赛中会影响时间分数，因此它的使用需要取舍。你可以在infer.py或run.py中最下方修改注释掉的部分来决定是否使用后处理。

## &#x1F4DA; Citation

如果你觉得本代码有用，可以考虑引用我们的工作：

```bibtex
TODO
```