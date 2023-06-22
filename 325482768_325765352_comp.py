import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

np.random.seed(0)


class Recommender:

    def __init__(self, train, test):
        self.R = None
        self.songs = None
        self.users = None
        self.train = train
        self.test = test
        self.train_test = None
        self.fit()

    def fit(self):
        # add user_id and song_id from test to train, and weight = 0 only for the new rows, train_test is a dataframe
        self.train_test = pd.concat([self.train, self.test], ignore_index=True)
        self.train_test = self.train_test.drop_duplicates(subset=['user_id', 'song_id'], keep='first')
        self.train_test = self.train_test.reset_index(drop=True)
        self.train_test['weight'] = self.train_test['weight'].fillna(0)

        self.users = self.train_test['user_id'].unique()
        self.songs = self.train_test['song_id'].unique()
        indices = []
        values = []
        for i in range(len(self.train_test)):
            indices.append((np.where(self.train_test['user_id'][i] == self.users)[0][0],
                            np.where(self.train_test['song_id'][i] == self.songs)[0][0]))
            values.append(self.train_test['weight'][i])

        self.R = scipy.sparse.coo_matrix((values, zip(*indices)), shape=(len(self.users), len(self.songs)))

    def q3(self):
        k = 20
        # calculate PCA with k of R
        R = self.R.astype(np.float64)
        u, s, vt = scipy.sparse.linalg.svds(R, k=k)
        # calculate R_hat
        R_hat = np.dot(np.dot(u, np.diag(s)), vt)
        # calculate f3
        f3 = np.sum([(row['weight'] - R_hat[
            np.where(row['user_id'] == self.users)[0][0], np.where(row['song_id'] == self.songs)[0][0]]) ** 2
                     for _, row in self.train.iterrows()])

        return f3

    def q4(self):
        R = np.asarray(self.R.todense())

        # Creating and training the MLP model
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)
        model.fit(R, R)
        R_hat = model.predict(R)
        f4 = np.sum([(row['weight'] - R_hat[
            np.where(row['user_id'] == self.users)[0][0], np.where(row['song_id'] == self.songs)[0][0]]) ** 2
                        for _, row in self.train.iterrows()])

        test_pred = np.zeros(len(self.test))
        for i in range(len(self.test)):
            test_pred[i] = R_hat[
                np.where(self.test['user_id'][i] == self.users)[0][0],
                np.where(self.test['song_id'][i] == self.songs)[0][0]]

        # add test_pred to test as a column
        test_pred_df = self.test.copy(deep=True)
        test_pred_df['weight'] = test_pred
        test_pred_df.to_csv('325482768_325765352_task4.csv', index=False)

        return f4


    def cross_validation(self):
        indices = []
        values = []
        dense_R = self.R.todense()
        R_splitted = []
        for row, col in zip(*self.R.nonzero()):
            indices.append((row, col))
            values.append(dense_R[row, col])
        values = np.array(values)
        indices = np.array(indices)
        # pick 5 independent random samples of 20% of the indexes of values
        folds = []
        for i in range(4):
            folds.append(np.random.choice(len(values), int(len(values) * 0.25), replace=False))

        losses = []
        for i, fold in enumerate(folds):
            without_fold = [i for i in range(len(values)) if i not in fold]
            train_mat = scipy.sparse.coo_matrix((values[without_fold], zip(*indices[without_fold])), shape=(len(self.users), len(self.songs)))
            test_mat = scipy.sparse.coo_matrix((values[fold], zip(*indices[fold])), shape=(len(self.users), len(self.songs)))
            # train the model
            train_mat_dense = np.asarray(train_mat.todense())
            R_hat = MLPR_model(train_mat_dense)
            test_mat_dense = np.asarray(test_mat.todense())
            loss = np.sum([(test_mat_dense[row, col] - R_hat[row, col]) ** 2
                             for row, col in zip(*test_mat.nonzero())])
            losses.append(loss)
            print(f'fold {i}: loss = {loss}')
        print(f'average loss = {np.mean(losses)}')

def NMF_model(R, n_components):
    model = NMF(n_components=n_components, init='random', max_iter=1000, alpha_W=1.5, alpha_H=1.5)
    W = model.fit_transform(R)
    H = model.components_
    R_hat = np.dot(W, H)
    return R_hat

def MLPR_model(R):
    # Creating and training the MLP model
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)
    model.fit(R, R)
    R_hat = model.predict(R)
    return R_hat


def main():
    train = pd.read_csv('user_song.csv')
    test = pd.read_csv('test.csv')
    rec = Recommender(train, test)
    rec.cross_validation()


main()
