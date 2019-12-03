import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.fftpack

def visualize_aggregate_fft(df, t=0.01):
    agg = df.groupby('y').mean().T
    n = len(agg)
    print(agg.shape)
    for colname in agg.columns:
        xf = np.linspace(t, 1.0/(2.0*t), n/2)[1:]
        yf = scipy.fftpack.fft(agg.loc[:, colname])
        plt.plot(xf, 2.0/n * np.abs(yf[1:n//2]))
    plt.legend(agg.columns)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Microvolts (μV)")
    plt.xlim(1, 20)
    plt.show()
    

def sample_eegs_from_class(df, class_label, n=1, **kwargs):
    x = list(range(len(df.columns) - 2))
    indices = np.random.choice(list(df[df['y'] == class_label].index), n, replace=False)
    visualize(df, indices, **kwargs)


def visualize(df, index, fft=False, t=0.01):
    n = len(df.columns) - 2
    x = list(range(n))
    plt.title("EEG Visualization, Rows {}".format(index))
    for i in index:

        y = df.iloc[i][1:-1].astype('int8') # this removes the patient ID and the label
        color = ['r', 'b', 'c', 'g', 'y']
        if fft:
            x = np.linspace(0.0, n*t, n)
            yf = scipy.fftpack.fft(y)

            xf = np.linspace(0.0, 1.0/(2.0*t), n/2)
            plt.plot(xf, 2.0/n * np.abs(yf[:n//2]), color[df.iloc[i][-1] - 1])
        else:
            plt.xlabel("time (in 1/178 s increments; total = 1s)")
            plt.plot(x, y, color[df.iloc[i][-1] - 1])
            plt.legend(["Row {}, y={}".format(i, df.iloc[i][-1]) for i in index])
    plt.show()

if __name__=='__main__':
    path = "./data.csv"
    dataset = pd.read_csv(path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fft', action='store_true')
    parser.add_argument('index', nargs='+', type=int)
    args = parser.parse_args()
    visualize(dataset, args.index, args.fft)

