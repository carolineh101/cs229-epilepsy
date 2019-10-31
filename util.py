import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product


DATA_PATH = 'data.csv'


def parse_ids():
    """
    This function parses the ID column of the data, where each entry is formatted "X<CHUNK>.<ID>".
    CHUNK, which has range [1, 23], is the location of the activity instance in the given person's
    entire activity sequence.
    ID is the identifier for the given person associated with the activity instance.

    Returns:
      - id_ordering: the indices that would sort the data by ID.
      - times: the unsorted (n, 1) array containing the CHUNK for each activity instance.
    """
    raw_ids = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=True, usecols=[0], dtype=str)
    num_instances = raw_ids.shape[0]
    chunks = np.zeros((num_instances, 1), dtype=np.uint8)
    ids = np.zeros((num_instances), dtype='S7')

    # Extract the CHUNK and ID for each activity instance.
    for i in range(num_instances):
        split_id = raw_ids[i].strip('\"').strip('X').split('.')
        chunks[i, 0] = split_id[0]
        id = split_id[1]
        if len(split_id) == 3:
            id += '.' + split_id[2]
        ids[i] = id

    id_ordering = np.argsort(ids)
    return id_ordering, chunks


def parse_data(multiclass=True):
    """
    This function parses the dataset, where each row is an instance of activity for a person,
    into X and y where each row corresponds to a person (full activity sequence and label).

    Parameter:
      - multiclass: whether classification problem is multiclass (else binary for epilepsy)
    Returns:
      - X: a (500, 4094) array where each row is a person's activity sequence.
      - y: a (500,) array where each entry is the label of the corresponding person's activity.
    """
    # Parse ID information from the first column of the data.
    id_ordering, chunks = parse_ids()
    c = np.unique(chunks).shape[0]  # The number of chunks an individual's activity is split into.

    data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=True, dtype=np.int16)[:, 1:]
    # Index the data by CHUNK and sort by ID.
    data = np.concatenate((chunks, data), axis=1)[id_ordering]
    n = int(data.shape[0] / c)  # The number of unique individuals (500).

    # Sort each individual's data by CHUNK.
    for i in range(n):
        start = i * c
        end = start + c
        chunk_ordering = np.argsort(data[start:end, 0])
        data[start:end, :] = data[start:end, :][chunk_ordering]

    X = data[:, 1:data.shape[1] - 1]
    # Reshape the data so that each row contains the activity/label for a given person.
    X = np.reshape(X, (n, c * X.shape[1]))
    y = np.array([data[i, data.shape[1] - 1] for i in range(0, data.shape[0], c)])
    if not multiclass: y = y == 1
    return X, y


def visualize_confusion_matrix(cm, classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
