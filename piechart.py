#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import matplotlib.font_manager as mfm


def piechart(df, name):
    fig, ax = plt.subplots(figsize=(6.0,6.0))
    # 中文字體
    font_path = "data/SourceHanSansTW-Regular.otf"
    prop = mfm.FontProperties(fname=font_path)

    # Plot pie chart
    patch, texts, autotexts = ax.pie(df['次數'], labels = df["Unnamed: 0"], autopct='%1.2f%%')

    # Plot label
    plt.title(label = f"Pie chart of {name}" ,fontproperties=prop)
    plt.setp(autotexts, fontproperties=prop)
    plt.setp(texts, fontproperties=prop)
    #plt.show()
    plt.savefig(f"png/{name}_pie.png")
    plt.close()



if __name__ == "__main__":
    path_list = glob("theme/*")
    for pth in path_list:
        df = pd.read_csv(pth, encoding='utf-8')
        name = pth[6:]
        piechart(df, name)
