import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualizeByClass(df):
    agg = df.groupby('y').mean()
    print(agg)
    df_plot = agg.T
    df_plot.plot(xticks=range(0, 173, 10)).legend(title='Class (y)', bbox_to_anchor=(1, 1))
    plt.show()

if __name__=='__main__':
    path = "./data.csv"
    dataset = pd.read_csv(path)
    visualizeByClass(dataset)