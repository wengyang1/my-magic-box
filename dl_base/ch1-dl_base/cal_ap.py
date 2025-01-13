import numpy as np
import matplotlib.pyplot as plt
# 增加样本数到20的假设数据
confidences = np.random.rand(20)  # 随机生成20个置信度值作为示例
# 注意：在实际应用中，ious应该是通过计算预测框与真实框的交并比得到的。
# 这里我们随机生成一些IoU值作为示例，但通常你需要一个匹配过程来确定每个预测框与哪个真实框对应。
ious = np.random.rand(20)  # 随机生成20个IoU值作为示例
# true_labels在实际中应该是基于IoU阈值判断得到的。这里我们假设一些标签作为示例。
# 假设前10个样本中有5个是正例（true_labels为1），后10个样本中也有5个是正例。
true_labels = np.array([1] * 5 + [0] * 5 + [1] * 5 + [0] * 5)

# IoU阈值
iou_threshold = 0.5

# 根据IoU阈值判断TP/FP
tps = (ious >= iou_threshold) & (true_labels == 1)
fps = (ious < iou_threshold) | (true_labels == 0)

# 对置信度进行排序，并获取排序后的索引
sorted_indices = np.argsort(-confidences)
confidences = confidences[sorted_indices]
tps = tps[sorted_indices]
fps = fps[sorted_indices]

# 累计TP和FP以计算Precision和Recall
cumulative_tps = np.cumsum(tps)
cumulative_fps = np.cumsum(fps)

# 由于我们不知道正例的总数（在实际应用中，这通常是通过数据集标注得知的），
# 在这里我们假设true_labels中1的数量代表正例的总数（这在这个简化的例子中是成立的）。
num_positive_samples = np.sum(true_labels == 1)

# 计算Precision和Recall
precision = cumulative_tps / (cumulative_tps + cumulative_fps + 1e-6)  # 加1e-6防止除以0
recall = cumulative_tps / num_positive_samples

# 绘制PR曲线
plt.figure()
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)
plt.show()

# 使用VOC 2007方法计算AP（11点插值）
recall_points = np.linspace(0, 1, 11)
ap = 0.0
for r in recall_points:
    # 找到大于或等于当前recall值的最大precision（使用插值前的最后一个点）
    # 注意：这里的逻辑是为了模拟VOC 2007的11点插值方法，但在实际代码中，
    # 我们通常会使用更高效的方法来计算PR曲线下的面积，而不是逐个recall点去查找。
    precision_at_r = max(precision[recall >= r]) if np.any(recall >= r) else 0.0
    ap += precision_at_r
ap /= len(recall_points)

print(f"AP: {ap:.4f}")

# 为了VOC 2010及以后的插值方法，我们需要所有独特的召回率点
unique_recall_points = np.unique(recall)
interpolated_precisions = []

for r in unique_recall_points:
    # 找到所有recall <= r的点
    mask = recall <= r
    # 如果存在这样的点，则取这些点中precision的最大值作为插值后的precision
    if np.any(mask):
        interpolated_precisions.append(np.max(precision[mask]))
    else:
        # 理论上不应该发生，因为unique_recall_points是从recall中得到的
        # 但为了代码的健壮性，还是加上这个else分支
        interpolated_precisions.append(0.0)

# 计算AP：插值后的precision的平均值
ap = np.mean(interpolated_precisions)

print(f"AP: {ap:.4f}")

# 为了COCO的AP计算，我们需要对所有独特的召回率点进行插值
# 通常，我们会使用一个更精细的召回率网格来进行插值
recall_grid = np.linspace(0, 1, 101)  # COCO使用101个点进行插值
interpolated_precisions = []

for r in recall_grid:
    # 找到所有recall >= r的点（注意这里是>=，与VOC的<=不同）
    mask = recall >= r
    # 如果存在这样的点，则取这些点中最后一个点的precision作为插值后的precision
    # 这是因为COCO使用“最大precision”插值方法的变体，但在这里我们简化为最后一个点的precision
    # 在实际应用中，可能需要更复杂的插值逻辑，特别是当召回率网格非常精细时
    if np.any(mask):
        interpolated_precision = precision[mask][-1] if np.any(mask) else 0.0
        interpolated_precisions.append(interpolated_precision)
    else:
        interpolated_precisions.append(0.0)

# 由于我们简化了插值逻辑，这里直接取最后一个点的precision可能会导致问题
# 在实际应用中，应该实现更准确的插值方法，比如线性插值或使用COCO官方工具中的方法
# 但为了示例目的，我们继续使用这个简化的方法

# 计算AP：插值后的precision的平均值（这里实际上可能不准确，因为插值方法被简化了）
ap = np.mean(interpolated_precisions)

print(f"AP: {ap:.4f}")