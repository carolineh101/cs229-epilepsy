import pandas as pd
import matplotlib.pyplot as plt
import argparse 

def visualize(df, index):
    x = list(range(len(df.columns) - 2))
    plt.title("EEG Visualization, Rows {}".format(index))
    plt.xlabel("time (in 1/178 s increments; total = 1s)")
    for i in index:
        y = df.iloc[i][1:-1].astype('int8') # this removes the patient ID and the label
        plt.plot(x, y)

    plt.legend(["Row {}, y={}".format(i, df.iloc[i][-1]) for i in index])
    plt.show()

if __name__=='__main__':
    path = "./data.csv"
    dataset = pd.read_csv(path)
    parser = argparse.ArgumentParser()
    parser.add_argument('index', nargs='+', type=int)
    args = parser.parse_args()
    visualize(dataset, args.index)

