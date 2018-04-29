
import matplotlib.pyplot as plt
import pandas as pd

tips2 = pd.read_csv('/media/ghost/Games And Images/INF552/project/TSAwithSSL-master/ratio.csv')

x=tips2.values[:,:1]
y=tips2.values[:,1:]
print("end")


plt.plot(x,y,'bx-')
plt.title('Ratio of data v/s Improvement using Semi-supervised Approach')
plt.xlabel("Ratio of data used for Supervised-Training")
plt.ylabel("Percentage of Improvement Average F-1 score")
plt.show()




import numpy as np
N = 8
error = (0.5204, 0.5324,0.5256,0.5121,0.2182,0.5258,0.495,0.4857)
f_score=(0.5095,0.5936,0.5503,0.4866,0.1702, 0.5115,0.1108,0.4104)
ind = np.arange(N)  # the x locations for the groups
width = 0.30       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, error, width, color='b')
rects2 = ax.bar(ind+ width, f_score, width, color='g')
ax.set_ylabel('Accuracy and F-1 Score')
ax.set_title('Analysis of Classifiers')
ax.set_xticks(ind + width / 2)
ax.legend((rects1[0], rects2[0]), ('Accuracy', 'Avg F-1 Score'))
ax.set_xticklabels(('SVM-L2-rbf','SVM-L1-rbf','SVM-L2-Linear','SVM-L1-Linear','Logistic-Regression','SGDClassifier','Naive-Bayes','DecisionTree'))
plt.show()

