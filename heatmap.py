import seaborn as sn
import matplotlib.pyplot as plt
def draw_heatmap(corrMatrix, n, m):
    fig, ax = plt.subplots(figsize=(500,50))   

    res = sn.heatmap(corrMatrix.iloc[0:4, n:m], annot=True, annot_kws={"fontsize":150})
    res.set_xticklabels(res.get_xmajorticklabels(),rotation = 30, fontsize =155)
    res.set_yticklabels(res.get_ymajorticklabels(),rotation = 0, fontsize = 155)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
