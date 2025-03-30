import numpy as np
import matplotlib.pyplot as plt

# 特征：x1, x2；标签：y（0 或 1）
X = np.array([
    [2.7, 2.5],
    [1.3, 3.3],
    [3.1, 1.8],
    [2.0, 2.2],
    [2.0, 3.0],
    [1.0, 1.0]
])
y = np.array([0, 0, 1, 0, 1, 1])


def gini_index(groups, classes):
    # groups: 左右子集 [left, right]
    # classes: 所有可能的类别
    n_instances = sum(len(group) for group in groups)

    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini += (1 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] <= value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = None, None, 1, None

    for index in range(len(dataset[0]) - 1):  # 遍历所有特征
        for row in dataset:
            value = row[index]
            groups = test_split(index, value, dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, value, gini, groups
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


def to_terminal(group):
    labels = [row[-1] for row in group]
    return max(set(labels), key=labels.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del node['groups']

    # 如果某个分支为空
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # 到达最大深度
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # 左分支继续分
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    # 右分支继续分
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = best_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['index']] <= node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# 构造数据集（最后一列是标签）
dataset = np.hstack((X, y.reshape(-1, 1))).tolist()

# 训练一棵树
tree = build_tree(dataset, max_depth=3, min_size=1)

# 预测
for row in dataset:
    print(f"Sample: {row[:-1]}, True: {int(row[-1])}, Predicted: {predict(tree, row)}")


def plot_decision_boundary(predict_fn, X, y, steps=100):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, steps),
                         np.linspace(y_min, y_max, steps))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_dataset = [row.tolist() + [0] for row in grid]  # 添加假标签

    Z = np.array([predict_fn(tree, row) for row in grid_dataset])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("CART Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


# 调用
plot_decision_boundary(predict, X, y)
