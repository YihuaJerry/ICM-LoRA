import os
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt

# 加载指定关键词的向量
def load_vectors_from_dir(directory, label, keyword):
    vectors = []
    labels = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".pth") and keyword in file:
            vector = torch.load(os.path.join(directory, file))
            vectors.append(vector.cpu().numpy())
            labels.append(label)
    if not vectors:
        raise ValueError(f"No vectors with keyword '{keyword}' found in {directory}")
    return np.array(vectors), np.array(labels)

# 设置路径
dog_vector_dir = "dog_vector"
sofa_vector_dir = "sofa_vector"
cat_vector_dir = "cat_vector"
bicycle_vector_dir = "bicycle_vector"
motorbike_vector_dir = "motorbike_vector"
# 只加载 encoder 向量
keyword = "decoder"  # 可以修改为 "decoder" 来加载 decoder 向量

# 加载 Dog 和 Sofa 的向量
dog_vectors, dog_labels = load_vectors_from_dir(dog_vector_dir, "dog", keyword)
sofa_vectors, sofa_labels = load_vectors_from_dir(sofa_vector_dir, "sofa", keyword)
cat_vectors, cat_labels = load_vectors_from_dir(cat_vector_dir, "cat", keyword)
bicycle_vectors, bicycle_labels = load_vectors_from_dir(bicycle_vector_dir, "bicycle", keyword)
motorbike_vectors, motorbike_labels = load_vectors_from_dir(motorbike_vector_dir, "motorbike", keyword)
if keyword == "encoder":
    dog_vectors = dog_vectors.squeeze(1)
    sofa_vectors = sofa_vectors.squeeze(1)
    cat_vectors = cat_vectors.squeeze(1)
elif keyword == "decoder":
    dog_vectors = dog_vectors.reshape(-1, dog_vectors.shape[2])
    sofa_vectors = sofa_vectors.reshape(-1, sofa_vectors.shape[2])
    cat_vectors = cat_vectors.reshape(-1, cat_vectors.shape[2])
    bicycle_vectors = bicycle_vectors.reshape(-1, bicycle_vectors.shape[2])
    motorbike_vectors = motorbike_vectors.reshape(-1, motorbike_vectors.shape[2])

# 检查向量维度
if dog_vectors.shape[1] != sofa_vectors.shape[1]:
    raise ValueError("Dog and Sofa vectors have inconsistent dimensions!")

if keyword == "decoder":
    dog_labels = np.repeat(dog_labels, 3)  # 每个dog的标签重复3次
    sofa_labels = np.repeat(sofa_labels, 3)  # 每个sofa的标签重复3次
    cat_labels = np.repeat(cat_labels, 3)  # 每个cat的标签重复3次
    bicycle_labels = np.repeat(bicycle_labels, 3) 
    motorbike_labels = np.repeat(motorbike_labels, 3) 

# 合并向量和标签
all_vectors = np.vstack([dog_vectors, sofa_vectors, cat_vectors, bicycle_vectors, motorbike_vectors])
all_labels = np.concatenate([dog_labels, sofa_labels, cat_labels, bicycle_labels, motorbike_labels])

print("Loaded vectors shape:", all_vectors.shape)
print("Labels:", np.unique(all_labels))

# 检查样本数量是否足够用于 t-SNE
if all_vectors.shape[0] < 2:
    raise ValueError("Not enough samples for t-SNE!")

# 执行 t-SNE 降维
perplexity = min(30, all_vectors.shape[0] - 1) 
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, learning_rate=200)
tsne_results = tsne.fit_transform(all_vectors)

# 绘制 t-SNE 图
plt.figure(figsize=(12, 10))

morandi_colors = [
    "#33A9AC",  
    "#FFA646",  
    "#F86041",  
    "#982062",  
    "#343779"   
]

for label, color in zip(["dog", "sofa", "cat", "bicycle", "motorbike"], morandi_colors):
    indices = all_labels == label
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                label=label, alpha=0.8, edgecolors='k', s=100, color=color)

plt.title(f"t-SNE Plot Vectors", fontsize=21)
plt.legend(fontsize=16)
plt.grid(True)
plt.savefig(f"tsne_all_{keyword}.png", dpi=300)  # 提高保存图像的分辨率
plt.show()
