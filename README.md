# 时间序列数据预处理和LSTM模型

## data_preparation.py
这个Python脚本用于对给定的数据集进行数据预处理和准备,以满足后续的数据分析、建模或机器学习任务的需求。

功能和作用:

1. **处理缺失值**: 使用前向填充的方式(ffill)填充缺失值,确保数据的连续性。
2. **数据转换**: 将电力负荷列中的逗号去除,并将其转换为浮点数类型,以便后续处理。
3. **时间列处理**: 将时间列转换为日期时间格式,使其具有时间序列的性质,方便后续时间相关的分析和建模。
4. **数据排序**: 按照时间升序对数据进行排序,以确保数据按照时间顺序排列。
5. **保留所需列**: 只保留电力负荷列,去除其他不需要的列,以简化数据集并专注于感兴趣的特征。
6. **数据归一化**: 使用最小-最大缩放(MinMaxScaler)将电力负荷值归一化到0到1的范围内,以消除不同数值范围的影响,并使得数据更适合某些模型或算法。
7. **保存处理后的数据集**: 将经过处理和准备的数据保存为新的CSV文件,以便后续使用。

## dataset_split.py
该脚本用于将数据集`YearlyEliaGridLoadByQuarterHour_2022`进行处理,划分为train、val、test三个数据集。

## dataset_model.py
这个Python脚本定义了两个类:

### TimeSeriesDataset
一个自定义的PyTorch数据集,用于处理时间序列数据。它接受输入数据和序列长度作为参数,并在`__getitem__`方法中返回按序列长度切片的输入和目标数据。这个类可以用于创建训练集和验证集的数据集对象。

### LSTMModel
定义了一个LSTM模型的结构,继承自`nn.Module`。在`__init__`方法中定义了LSTM层和线性层,在`forward`方法中实现了模型的前向传播过程。

这两个类的目的是为了在时间序列数据上构建和训练LSTM模型。`TimeSeriesDataset`类用于处理数据集的加载和处理,而`LSTMModel`类定义了LSTM模型的结构和前向传播方法。

## train.py
该脚本用于训练LSTM模型,每一轮epoch打印训练loss和验证loss,最终保存训练完成后最好的模型`best.pt`。

## test.py
该脚本用于在测试集上对已训练好的模型进行评估,主要功能包括:

1. 加载已训练好的模型参数,使用`torch.load`函数加载已训练好的模型参数,将其应用到模型实例中。
2. 在测试集上进行预测,使用训练好的LSTM模型在测试集上进行预测,将预测结果存储在列表`predictions`中。
3. 将预测结果转换为一维数组,使用`np.concatenate`将列表中的预测结果连接为一个一维数组。
4. 绘制预测值和真实值的曲线图,使用`matplotlib.pyplot`绘制测试集中的真实值和预测值的曲线图,以便进行可视化对比。
