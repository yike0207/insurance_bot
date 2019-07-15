import json
import pickle
from pathlib import Path


from configuration.config import data_dir

print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['SimHei']
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)


# digits = datasets.load_digits(n_class=6)
# X = digits.data
# y = digits.target
sel_col = ['保障范围_保险责任','核赔','核保','核保_不可投保疾病','保障范围_保险条款',
           '查询_保单','核保_不可投保职业','保全','保障范围_保费费率_属性','保障范围_保额_属性']
dic = {'保障范围_保险责任':'duty',
       '保障范围_保险条款': 'clause',
       '核赔': 'claim',
       '核保': 'underwriting',
       '核保_不可投保疾病': 'underwriting_non-insurable_disease'
       }
y = []
X = []

embed_dic = pickle.load((Path(data_dir)/'embed_bert_cnn.pkl').open('rb'))
for label, embeds in embed_dic.items():
    for embed in embeds:
        X.append(embed)
        y.append(label)
X = np.array(X)
n_samples, n_features = X.shape
n_neighbors = 30
# ----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plt.text(X[i, 0], X[i, 1], str(sel_col.index(y[i])),
        plt.text(X[i, 0], X[i, 1], str('+'),
                 color=plt.cm.Set1(sel_col.index(y[i]) / 10.),
                 fontdict={'weight': 'bold', 'size': 9},height=10, aspect=1)

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(X.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# # ----------------------------------------------------------------------
# # Plot images of the digits
# n_img_per_row = 20
# img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
# for i in range(n_img_per_row):
#     ix = 10 * i + 1
#     for j in range(n_img_per_row):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
#
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.title('A selection from the 64-dimensional digits dataset')


# ----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)

plot_embedding(X_tsne,
               "t-SNE embedding of the insurance classification")


plt.show()