import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
# tips = sns.load_dataset('tips')
# sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
# plt.show()
df = pd.read_csv('effect.csv')
sns.relplot(x="Training samples", y="Error rate (%)",hue='Model',style='Model', kind='line',
            data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()


print('done')
