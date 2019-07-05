import pickle
from pathlib import Path

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from configuration.config import data_dir

"Creates and TSNE model and plots it"
labels = []
tokens = []

embed_dic = pickle.load((Path(data_dir)/'embed.pkl').open('rb'))
for label, embeds in embed_dic.items():
    for embed in embeds:
        tokens.append(embed)
        labels.append(label)

tsne_model = TSNE(n_components=2, init='pca', random_state=23)
new_values = tsne_model.fit_transform(tokens)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])

plt.figure(figsize=(16, 16))
for i in range(len(x)):
    plt.scatter(x[i], y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.show()