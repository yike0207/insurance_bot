# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
# df = pd.DataFrame({'x': range(1, 11), 'y1': np.random.randn(10), 'y2': np.random.randn(10) + range(1, 11),
#                    'y3': np.random.randn(10) + range(11, 21)})

df = pd.read_excel('effect.xlsx')


# multiple line plot
plt.plot('training_samples', 'TextCNN', data=df, marker='o', markerfacecolor='red', markersize=6, color='red', linewidth=2)
plt.plot('training_samples', 'HAN', data=df, marker='o', markerfacecolor='olive', markersize=6, color='olive', linewidth=2)
plt.plot('training_samples', 'ELMo', data=df, marker='o', markerfacecolor='green',markersize=6, color='green', linewidth=2)
plt.plot('training_samples', 'BERT', data=df, marker='o', markerfacecolor='blue', markersize=6,color='blue', linewidth=2)
plt.xlabel('Number of training samples')
plt.ylabel('Test error rate(%)')
plt.legend()

plt.savefig('filename.png', dpi=600)
# plt.show()
