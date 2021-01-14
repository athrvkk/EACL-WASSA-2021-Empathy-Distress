import seaborn as sn
import matplotlib.pyplot as plt
def draw_heatmap(corrMatrix):
    fig, ax = plt.subplots(figsize=(50,15))   

    res = sn.heatmap(corrMatrix.iloc[0:4, :], annot=True, annot_kws={"fontsize":30})
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 25)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 25)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
