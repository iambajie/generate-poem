## 描述

（1）用户输入需要生成的诗句类型，如选择自由生成诗句，需要输入诗句数量；如选择藏头诗句，需要输入藏头文字。

（2）前台将用户输入传回，后台根据输入的要求调用模型进行采样生成诗句。

（3）后台将生成的诗句传回需要渲染的模板中进行显示。

例如：输入2，选择生成藏头诗，藏头的文字为：八戒我爱你，则生成的诗句显示在页面如下：（文件夹下MP4文件展示了验证过程）

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200714201829.png)

## 文件构成

![](https://cdn.jsdelivr.net/gh/iamxpf/pageImage/images/20200721125525.png)

（1）data文件夹下存放原始古诗词poems_edge_split.txt和通过skip-gram模型生成的词向量模型vectors_poem.txt

（2）output_poem文件夹下的result.json最佳模型位置及网络参数，best_model存放每次迭代的最佳模型，save_model存放每次得到的模型参数，tensorboard_log存放日志文件

（3）save文件夹下存放根据模型生成的诗句，templates文件夹下存放前台页面

（4）python文件

train.py：训练模型

poem_charnn.py：字符级模型结构及模型采样方法

poem_config.py：模型训练及采样参数设置

poem_loaddata.py：训练数据生成

poem_web.py：控制前后台数据传输，用于验证模型

poem_word2vec.py：导入训练的词向量模型，并加入< unkonwn >字符

poem_writer.py：按照前台要求进行采样，即生成诗句

Predata.py：预处理诗句数据集，在前后加上开始和结束符号