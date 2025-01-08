import numpy as np

# 假设有20个样本，每个样本可能包含一个或多个真实框（这里为了简化，每个样本只包含一个）
# 真实框数据：类别ID，x_min, y_min, x_max, y_max（坐标是归一化的）
ground_truths = [
    {'image_id': i, 'category_id': np.random.choice([1, 2]),
     'bbox': [np.random.rand() * 100, np.random.rand() * 100, np.random.rand() * 50 + 50, np.random.rand() * 50 + 50]}
    for i in range(20)
]

# 模拟检测数据：图像ID，类别ID，置信度，x_min, y_min, x_max, y_max
# 注意：这里为了简化，我们假设检测数据已经按置信度排序，并且每个检测框都与某个真实框匹配（IoU >= 0.5）
detections = [
    {'image_id': np.random.choice(range(20)), 'category_id': np.random.choice([1, 2]), 'confidence': np.random.rand(),
     'bbox': [np.random.rand() * 100, np.random.rand() * 100, np.random.rand() * 50 + 50, np.random.rand() * 50 + 50]}
    for _ in range(100)  # 假设有100个检测框，实际中可能更多
]

# 设置IoU阈值
iou_threshold = 0.5

# 初始化变量来存储每个类别的TP和FP
true_positives = {1: [], 2: []}
false_positives = {1: [], 2: []}


# 辅助函数：计算IoU
def compute_iou(box1, box2):
    x1_max, y1_max = box1[2], box1[3]
    x2_max, y2_max = box2[2], box2[3]

    # 计算交集区域
    inter_area = max(0, min(x1_max, x2_max) - max(box1[0], box2[0])) * max(0,
                                                                           min(y1_max, y2_max) - max(box1[1], box2[1]))

    # 计算并集区域
    box1_area = (x1_max - box1[0]) * (y1_max - box1[1])
    box2_area = (x2_max - box2[0]) * (y2_max - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou


# 匹配检测框和真实框
for detection in detections:
    best_iou = 0
    best_gt_idx = None
    matched = False
    for gt_idx, gt in enumerate([gt for gt in ground_truths if gt['category_id'] == detection['category_id']]):
        iou = compute_iou(detection['bbox'], gt['bbox'])
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_gt_idx = gt_idx
            # 标记这个真实框为已匹配（在实际应用中，你需要一个更复杂的逻辑来确保每个真实框只被匹配一次）
            # 这里为了简化，我们省略了这个逻辑

    if best_gt_idx is not None:
        true_positives[detection['category_id']].append(detection['confidence'])
    else:
        false_positives[detection['category_id']].append(detection['confidence'])


# 计算每个类别的AP
def compute_ap(true_positives, false_positives, total_positives):
    cumulative_true_positives = np.cumsum(np.ones_like(true_positives))
    cumulative_false_positives = np.cumsum(np.ones_like(false_positives))

    recall = cumulative_true_positives / total_positives
    recall_points = np.linspace(0, 1, 11)
    interpolated_precisions = []

    for r in recall_points:
        mask = recall <= r
        if np.any(mask):
            precision_at_r = cumulative_true_positives[mask][-1] / (
                        cumulative_true_positives[mask][-1] + cumulative_false_positives[mask][-1] + 1e-6)
            interpolated_precisions.append(precision_at_r)
        else:
            interpolated_precisions.append(0.0)

    ap = np.mean(interpolated_precisions)
    return ap


# 计算每个类别的总正样本数（真实框数）
total_positives = {1: sum(1 for gt in ground_truths if gt['category_id'] == 1),
                   2: sum(1 for gt in ground_truths if gt['category_id'] == 2)}

# 计算每个类别的AP
aps = {category_id: compute_ap(true_positives[category_id], false_positives[category_id], total_positives[category_id])
       for category_id in [1, 2]}

# 计算mAP
map_score = np.mean(list(aps.values()))

print(f"AP for category 1: {aps[1]:.4f}")
print(f"AP for category 2: {aps[2]:.4f}")
print(f"mAP: {map_score:.4f}")