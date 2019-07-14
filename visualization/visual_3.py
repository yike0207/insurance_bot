import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
# tips = sns.load_dataset('tips')
# sns.relplot(x="total_bill", y="tip", hue="smoker", data=tips);
# plt.show()
df = pd.read_excel('effect.xlsx')
b = sns.relplot(x="Training samples", y="Error rate (%)",hue='Model',style='Model', kind='line',height=10, aspect=1,
            data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=20)
plt.figure(figsize=(20,15))

b.set_xlabel("X Label",fontsize=30)
b.set_ylabel("Y Label",fontsize=20)
b.tick_params(labelsize=5)

plt.show()


print('done')
