import matplotlib.pyplot as plt
import numpy as np
import os
import re
from itertools import product
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle


DATA_PATH = 'data.csv'
K = 10

# Hyperparameter search grids.
CNN_PARAM_GRID = {
    'filter': [2 ** i for i in range(4, 8)],
    'window': range(3, 34, 6),
    'dropout': [i / 10. for i in range(6)],
    'epochs': [25],
    'batch_size': [5, 10, 50]
}
KNN_PARAM_GRID = {'n_neighbors': range(1, 10)}
SREG_PARAM_GRID = {'C': [0.1, 0.5, 1, 5]}
PARAM_GRID_DICT = {'cnn': CNN_PARAM_GRID, 'knn': KNN_PARAM_GRID, 'sreg': SREG_PARAM_GRID}


def parse_data(multiclass=True, normalize=True):
    """
    This function parses the dataset.

    Parameter:
      - multiclass: whether classification problem is multiclass (else binary for epilepsy)
    Returns:
      - X: a (11500, 178) array where each row is an instance of activity for a person.
      - y: a (11500,) array where each entry is the label of the activity instance.
      - ids: a (11500,) array where each entry is the ID of the person.
      - chunks: a (11500,) array where each entry is the time chunk.
    """
    # Parse ID/chunk information from the first column of the data using a regex.
    raw_ids = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=True, usecols=[0], dtype=str)
    pattern = '\"X(?P<chunk>\d+)\.(?P<id>.+)\"'
    id_match = np.vectorize(lambda x : re.match(pattern, x).group('id'))
    chunk_match = np.vectorize(lambda x: int(re.match(pattern, x).group('chunk')))
    ids = id_match(raw_ids)
    chunks = chunk_match(raw_ids)
    # Parse X, y data.
    data = np.genfromtxt(DATA_PATH, delimiter=',', skip_header=True, dtype=np.int16)[:, 1:]
    X = data[:, :data.shape[1] - 1]
    if normalize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = data[:, data.shape[1] - 1]
    if not multiclass: y = y == 1
    return X, y, ids, chunks


def get_scoring_metrics(classes, multiclass=True):
    # Generate scoring function dict (accuracy, macro F1, and F1 per class).
    scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
    f1_class_fn = 'def f1_class_%d(y_true, y_pred, classes=None): ' + \
        'return f1_score(y_true, y_pred, average=None, labels=classes)[%d]'
    for c in classes:
        scoring_fn = 'f1_class_%d' % c
        def make_f1_class_fn(c):
            exec(f1_class_fn % (c, c - 1 if multiclass else c))
            return locals()[scoring_fn]
        scoring[scoring_fn] = make_scorer(make_f1_class_fn(c), classes=classes)
    return scoring


def create_cnn_model(input_shape, num_classes, filter, window, dropout):
    # Adapted from "Sequence classification with 1D convolutions" in keras.io/getting-started/sequential-model-guide.
    model = Sequential()
    model.add(Conv1D(filter, window, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filter, window, activation='relu'))
    model.add(MaxPooling1D(window))
    model.add(Conv1D(filter * 2, window, activation='relu'))
    model.add(Conv1D(filter * 2, window, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_estimator(model_type, seed, input_shape, num_classes):
    if model_type == 'sreg':
        return LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial', max_iter=5000)
    elif model_type == 'knn':
        return KNeighborsClassifier()
    elif model_type == 'cnn':
        return KerasClassifier(build_fn=create_cnn_model, input_shape=input_shape, num_classes=num_classes, filter=64, window=3, dropout=0.5)


def evaluate_model(X, y, groups, classes, model_type, scoring, k=K, seed=0, multiclass=True):
    """
    This function evaluates a given model using nested cross-validation.
    Adapted from scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html.
    """
    assert model_type in PARAM_GRID_DICT, 'model_type currently only supports sreg, knn, and cnn.'
    param_grid = PARAM_GRID_DICT[model_type]
    clf = get_estimator(model_type, seed, (X.shape[1], 1), len(classes))

    # Run nested cross-validation.
    shuffled_X, shuffled_y, shuffled_groups = shuffle(X, y, groups, random_state=seed)
    inner_cv = GroupKFold(n_splits=k)
    outer_cv = GroupKFold(n_splits=k)
    clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=inner_cv, iid=False)
    clf.fit(shuffled_X, y=shuffled_y, groups=shuffled_groups)

    # Calculate scores.
    cv_scores = cross_validate(
        clf, shuffled_X, y=shuffled_y, groups=shuffled_groups, cv=outer_cv, scoring=scoring,
        fit_params={'groups': shuffled_groups}, return_train_score=True)
    train_scores = {metric : np.mean(cv_scores['train_%s' % metric]) for metric in scoring}
    test_scores = {metric : np.mean(cv_scores['test_%s' % metric]) for metric in scoring}
    fold_accuracy = cv_scores['test_accuracy']

    # Calculate confusion matrix.
    y_pred = cross_val_predict(
        clf, shuffled_X, y=shuffled_y, groups=shuffled_groups, cv=outer_cv,
        fit_params={'groups': shuffled_groups})
    cm = confusion_matrix(shuffled_y, y_pred)
    return train_scores, test_scores, fold_accuracy, cm


def plot_fold_accuracy(fold_accuracy, k=K):
    plt.bar(range(1, k + 1), fold_accuracy)
    plt.title('Per-Fold Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Fold')
    plt.show()


def visualize_confusion_matrix(cm, classes):
    # Adapted from scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html.
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


########## BELOW FUNCTIONS FOR DATA VISUALIZATION ##########


def parse_ids_one_row_per_person():
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


def parse_data_one_row_per_person(multiclass=True):
    """
    This function parses the dataset, where each row is an instance of activity for a person,
    into X and y where each row corresponds to a person (full activity sequence and label)
    primarily for the purposes of visualization.
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
