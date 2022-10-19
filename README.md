<!--
 * @Author: Peng Bo
 * @Date: 2022-08-11 13:47:21
 * @LastEditTime: 2022-08-13 22:17:59
 * @Description: 
 * 
-->


### 步骤
1. 利用人体姿态模型得到视频每一帧的关键点，写入文件（检测不到时，该帧的关键点设置为 0）;
2. 利用 `generate_labels.py` 生成样本。几个需要注意的参数：
    a. duration_window，单个样本的时间窗口
    b. pool_window，平滑窗口，提取特征时用于消除抖动的影响
    c. step_window，生成样本时的步长
3. 人工标定的两类类别：动作、状态，时间是连续的，比如在 0～15s 时为动作，则在这15s内生成多个动作类样本
4. 生成了样本之后，训练二分类神经网络模型来区分动作还是状态


#### 需要明确的几个问题：
1. 人工标定的类别可以更为精细，比如状态可以分为：坐在升降桌前、站在升降桌前、坐着但不在升降桌前、站着但不在升降桌前；
2. 特征工程（将关键点坐标转化为样本特征）在 `generate_labels.py` 文件中的 `embedded_lms` 函数，目前只是非常简单地进行拼接，可以考虑将 pool_window 内的关键点做一个平滑操作；
3. 其他改进想法。。。