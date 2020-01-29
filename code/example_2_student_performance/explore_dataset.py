import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from os.path import abspath, exists

DATA_DIR = abspath('../../datasets/student')
NON_NUMERIC_COLUMNS = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', ]
YES_NO_COLUMNS = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', ]


def get_raw_data(*, subject='por'):
    return pd.read_csv(f'{DATA_DIR}/student-{subject}.csv', delimiter=';')


def _to_erasmus(grade):
    if grade >= 16:
        return 1
    elif grade >= 14:
        return 2
    elif grade >= 12:
        return 3
    elif grade >= 10:
        return 4
    return 5


def create_preprocessed_csv(*, subject='por'):
    df = get_raw_data(subject=subject)
    # convert all yes and no to 1 and 0
    df = df.replace({'yes': 1., 'no': 0.})
    # Convert from 20 to 5 categories using erasmus system
    df['G3'] = df.aggregate({'G3': _to_erasmus})
    # Convert categorical data to 1-of-N encoding
    le = LabelEncoder()
    for col in NON_NUMERIC_COLUMNS:
        df[col] = le.fit_transform(df[col])
    # scale outputs to between 1 and 0
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df['G3'] = df['G3'] * 4 + 1  # convert back to labels [0, 1, 2, 3, 4] from [0, 0.25, 0.5, 0.75, 1]
    df.to_csv(f'{DATA_DIR}/student-{subject}-processed.csv', sep=';', index=False)


def load_preproccessed_dataset(test_split=0.1, *, subject='por', include_grades=False):
    if not exists(f'{DATA_DIR}/student-{subject}-processed.csv'):
        create_preprocessed_csv(file=f'student-{subject}')
    df = pd.read_csv(f'{DATA_DIR}/student-{subject}-processed.csv', delimiter=';')
    feature_cut_off = 32 if include_grades else 30  # where to separate the features and the labels
    training_cut_off = int(len(df.index) * (1.0 - test_split))
    # convert from pandas DataFrame to numpy matrix
    data = df.to_numpy()
    features = data[:, :feature_cut_off]
    labels = data[:, -1]
    train = (features[:training_cut_off, :], labels[:training_cut_off])
    test = (features[training_cut_off:], labels[training_cut_off:])
    return train, test


def visualise():
    raw_df = get_raw_data()
    (X, y), _ = load_preproccessed_dataset(test_split=0.0)
    raw_grades = Counter(raw_df['G3'].values)
    erasmus_grades = Counter(y)

    plt.figure(1)  # grade distribution
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Grade distributions')

    ax1.bar(*zip(*raw_grades.items()))
    ax1.set_xlabel('Grade')
    ax1.set_ylabel('Count')

    plt.bar(*zip(*erasmus_grades.items()))
    ax2.set_xlabel('Grade')
    ax2.set_ylabel('Count')

    plt.show()


if __name__ == '__main__':
    visualise()
