import seaborn as sns

from matplotlib import pyplot as plt


def box_plot_grid_x(data, x_list, y, showfliers):
    fig, axes = plt.subplots(2, 3, figsize=(20, 7))
    axes = axes.flatten()
    
    for i in range(len(x_list)):
        order = sorted(data[x_list[i]].unique().tolist())
        ax = sns.boxplot(x=x_list[i], y=y, data=data, orient='v', 
            ax=axes[i], showfliers=showfliers, order=order)
        
        
def box_plot_grid_y(data, x, y_list, showfliers):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes = axes.flatten()
    
    for i in range(len(y_list)):
        order = sorted(data[x].unique().tolist())
        ax = sns.boxplot(x=x, y=y_list[i], data=data, orient='v', 
            ax=axes[i], showfliers=showfliers, order=order)
        
        
def barplot_grid(data, x_list, y, hue):
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    for i in range(len(x_list)):
        data_grouped = data.groupby([x_list[i],hue])[y].count().reset_index()
        data_totals = data.groupby(hue)[y].count().reset_index()
        data_grouped = data_grouped.merge(data_totals, on=hue, how="left")
        data_grouped["prop_total_users"] = data_grouped[f"{y}_x"]/data_grouped[f"{y}_y"]
        ax = sns.barplot(x=x_list[i], y = "prop_total_users", hue = hue, data=data_grouped, orient='v', 
            ax=axes[i])
        ax.legend(loc="upper right")