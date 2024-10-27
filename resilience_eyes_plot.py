# 数据预处理
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simps
from scipy.stats import pearsonr

#########################################################################################################
##### 数据预处理得到最后的结果
es = pd.read_csv('result/es_all.csv', encoding='gb18030')  # es 结果
xzbj = gpd.read_file('../../Data/全国一普到七普人口数据/DJ_XZQH.shp')  # 行政区划底图
xzbj['total_gdp'] = xzbj['gdp'] * xzbj['SUM_七普']
dj_citylist = list(xzbj[xzbj['地级'] == 1]['市'])  # 取其中是地级市的
names = pd.read_csv(r"F:\Research\全国韧性可视化\names.csv", encoding='gb18030')  # 中英文名对照表格

# 计算es 标准化
es = es / es.max()  # 与T0相比的e s
e = es.filter(like='_e')
s = es.filter(like='_s')

# 对每个城市e s计算积分面积
es_simps = {f'{city[:-2]}': [] for city in list(e.columns)}
for city in e.columns:
    x = [0.05 * i for i in range(20)]  # x的取值范围
    ye = e[f'{city[:-2]}_e']
    ys = s[f'{city[:-2]}_s']
    # 使用simps函数进行数值积分
    area1 = simps(ye, x)
    area2 = simps(ys, x)
    es_simps[f'{city[:-2]}'].append(area1)
    es_simps[f'{city[:-2]}'].append(area2)
newes = pd.DataFrame(es_simps).T
newes.columns = ['e', 's']
# newes['e'] = (newes['e'] - newes['e'].min()) / (newes['e'].max() - newes['e'].min())
# newes['s'] = (newes['s'] - newes['s'].min()) / (newes['s'].max() - newes['s'].min())
newes['all'] = (newes['e'] / newes['e'].max()) * (newes['s'] / newes['s'].max())

newes = newes[newes.index.isin(dj_citylist)]
newes = newes.sort_values(by='all', ascending=False)  # 得到按es汇总计算后的排序

temp = xzbj[['市','SUM_七普','gdp','total_gdp']].sort_values(by='gdp', ascending=False)
temp = temp[temp['市'].isin(dj_citylist)].reset_index()  # 得到按gdp汇总计算后的排序
# 按城市名merge得到最终结果 'result/result.csv'


#########################################################################################################
##### 数据可视化 - 圆环图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import matplotlib.patches as patches
import geopandas as gpd

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 30
alltemp = pd.read_csv(r"F:\Research\全国韧性可视化\result\result.csv", encoding='gb18030')
alltemp = alltemp[alltemp['CH_name'] != '三沙市'].reset_index()

alltemp['EN_name_1'] = [i if 'Autonomous' not in i else i.replace('\nAutonomous Prefecture','\nAP') for i in alltemp['EN_name_1']]
##### 城市圆环图绘图
def ringplot(ax, e, s, cityname):
    center, radius1 = (0, 0), 0  # 固定参数
    # 内环
    width1 = e * 12  # e乘10 方便可视化
    radius2 = radius1 + width1
    # 外环
    width2 = s * 12  # s乘10 方便可视化
    radius3 = radius2 + width2
    circle = patches.Wedge(center, radius1, 0, 360, fc='None')
    ring1 = patches.Wedge(center, radius2, 0, 360, width=width1, fc=colors1[i], ec='white')
    ring2 = patches.Wedge(center, radius3, 0, 360, width=width2, fc=colors11[i], ec='white', alpha=.7)
    # 添加圆环到图形
    ax.add_patch(circle)
    ax.add_patch(ring1)
    ax.add_patch(ring2)
    # 设置坐标轴范围
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    # 添加尺度标尺
    # ax.hlines(-4, radius1, radius2, color='k', lw=2)
    # ax.hlines(-4, radius2, radius3, color='k', lw=2)
    # ax.vlines(radius1, -4.2, -3.8, color='k', lw=2)
    # ax.vlines(radius2, -4.2, -3.8, color='k', lw=2)
    # ax.vlines(radius3, -4.2, -3.8, color='k', lw=2)
    # # 添加半径标签
    # ax.text(radius1 + width1 / 2, -3.6, f"{round(width1/10, 2)}",
    #            ha='center', va='center', fontsize=20)
    # ax.text(radius2 + width2 / 2, -4.8, f"{round(width2/10, 2)}",
    #            ha='center', va='center', fontsize=20)
    # 显示图形
    ax.set_aspect('equal', adjustable='box')
    # ax.set_title(cityname, fontsize=40)
    ax.axis('off')  # 关闭坐标轴

colors1 = plt.cm.Reds_r(np.linspace(0, 1, len(alltemp)))
colors11 = plt.cm.Oranges_r(np.linspace(0, 1, len(alltemp)))

x, y = 13, 27
fig, ax = plt.subplots(x, y, figsize=(60, 27), dpi=200)
ax = ax.flatten()
for i, city in enumerate(alltemp['EN_name_1']):
    e = alltemp.loc[alltemp['EN_name_1']==city,'E'].iloc[0]
    s = alltemp.loc[alltemp['EN_name_1']==city,'S'].iloc[0]
    ringplot(ax=ax[i], e=e, s=s, cityname=city)
    print(f'city done {round(100*i/len(alltemp),2)}%', end='\r')
for i in range(len(alltemp), x * y):
    ax[i].axis('off')
plt.tight_layout()
fig.savefig('../figure/es.jpg', dpi=300)

##### 行政边界绘图
xzbj = gpd.read_file('../../../Data/全国七普人口数据+上海分区人口/DJ_XZQH.shp')  # 行政区划底图
rank = alltemp.sort_values(by='Per capita GDP', ascending=False).reset_index()

colors2 = plt.cm.YlGnBu_r(np.linspace(0, 1, len(alltemp)))
x, y = 13, 27
fig, ax = plt.subplots(x, y, figsize=(60, 27), dpi=200)
ax = ax.flatten()
xzbj = xzbj.to_crs("EPSG:2433")
for i, city in enumerate(alltemp['CH_name']):
    colorindex = rank[rank['CH_name'] == city].index[0]  # 城市排序索引
    xzbj[xzbj['市'] == city].plot(ax=ax[i], color=colors2[colorindex])
    ax[i].axis('off')  # 关闭坐标轴
    print(f'city done {round(100*i/len(alltemp),2)}%', end='\r')
for i in range(len(alltemp), x * y):
    ax[i].axis('off')
plt.tight_layout()
fig.savefig('../figure/人均gdp1.jpg', dpi=300)





##### 城市名图例
plt.figure(figsize=(20, 40))
# 生成随机数据
x = np.arange(5)  # 生成x坐标
y = np.arange(68)  # 生成y坐标
X, Y = np.meshgrid(x, y)  # 生成网格坐标
plt.scatter(X, Y, alpha=0)
# 添加文本
count = 0
names = list(alltemp['EN_name'])
# names = [s.replace('\n', ' ') if '\n' in s else s for s in names]
for i in range(68):
    for j in range(5):
        if i == 67 and j > 1:
            plt.text(X[67-i, j], Y[67-i, j], ' ')
        else:
            if 'Autono' in names[count]:
                names[count] = names[count].replace('\n','\n'+(2*len(str(count+1))+1)*' ')
                plt.text(X[67-i, j], Y[67-i, j], f'{count+1} {names[count]}', fontsize=20, ha='left', va='center')
            else:
                plt.text(X[67-i, j], Y[67-i, j], f'{count+1} {names[count]}', fontsize=20, ha='left', va='center')
        count += 1
# 显示图形
plt.axis('off')
plt.tight_layout()
plt.show()




##### 数据可视化 - 散点图
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 30

result = pd.read_csv('result/result.csv', encoding='gb18030')
# alltemp = result
alltemp = result[~result['CH_name'].isin(['三沙市','嘉峪关市'])].reset_index()

# shilist = [i for i in result['CH_name'] if '市' in i]
# alltemp = result[result['CH_name'].isin(shilist)].reset_index()
# alltemp = alltemp[~alltemp['CH_name'].isin(['三沙市'])].reset_index()

plt.figure(figsize=(21, 15))
sns.regplot(x=alltemp['E&S'], y=alltemp['Per capita GDP'], scatter=False, color='k', line_kws={'linestyle':'--'}, ci=95)
scatter_plot = sns.scatterplot(data=alltemp, x='E&S', y='Per capita GDP', hue='Per capita GDP', size='E&S', sizes=(100, 1000), palette='Spectral_r', legend='auto')


# 计算回归线
slope, intercept, r_value, p_value, std_err = stats.linregress(alltemp['E&S'], alltemp['Per capita GDP'])
plt.text(0.5, 115000, f'r = {r_value:.2f}, p = {p_value:.3f}', fontsize=20, color='k')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size':20})  # 调整prop参数来设置图例文本的大小

# 添加每个点的文本标注
for line in range(0, alltemp.shape[0]):
    if alltemp['E&S'][line]>0.4 or alltemp['Per capita GDP'][line]>160000:
        if alltemp['EN_name'][line]=='Shenzhen':
            scatter_plot.text(alltemp['E&S'][line]-0.06, alltemp['Per capita GDP'][line]-1000, alltemp['EN_name'][line], horizontalalignment='left', size=15)
        elif alltemp['EN_name'][line]=='Changzhou':
            scatter_plot.text(alltemp['E&S'][line]+0.01, alltemp['Per capita GDP'][line]+1000, alltemp['EN_name'][line], horizontalalignment='left', size=15)
        elif alltemp['EN_name'][line]=='Shenyang':
            scatter_plot.text(alltemp['E&S'][line]+0.01, alltemp['Per capita GDP'][line]-3000, alltemp['EN_name'][line], horizontalalignment='left', size=15)
        elif alltemp['EN_name'][line]=='Xuchang':
            scatter_plot.text(alltemp['E&S'][line]+0.01, alltemp['Per capita GDP'][line]-1000, alltemp['EN_name'][line], horizontalalignment='left', size=15)
        elif alltemp['EN_name'][line]=='Zibo':
            scatter_plot.text(alltemp['E&S'][line]+0.01, alltemp['Per capita GDP'][line], alltemp['EN_name'][line], horizontalalignment='left', size=15)
        else:
            scatter_plot.text(alltemp['E&S'][line]-0.02, alltemp['Per capita GDP'][line]+4000, alltemp['EN_name'][line], horizontalalignment='left', size=15)
plt.grid()
plt.xlabel('E&S')
plt.ylabel('Per capita GDP / RMB')
plt.yticks(list(range(0,260000,50000)), [f'{round(i/1000)}k' for i in range(0,260000,50000)])
plt.tight_layout()
# plt.show()
plt.savefig('figure/scatter_plot.jpg', dpi=300)
