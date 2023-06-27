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


    def q4(self):
        f4 = lambda R_hat, indexes, values: np.sum([(R_hat[index] - value) ** 2
                                                    for index, value in zip(indexes, values)])
        indices = []
        values = []
        dense_R = self.R.todense()
        for row, col in zip(*self.R.nonzero()):
            indices.append((row, col))
            values.append(dense_R[row, col])

        # split to train and test
        indicies_train, indicies_test, values_train, values_test = train_test_split(indices, values, test_size=0.33)

        # create R_train
        R_train = scipy.sparse.coo_matrix((values_train, zip(*indicies_train)),
                                          shape=(len(self.users), len(self.songs)))

        R_train = np.asarray(R_train.todense())
        # Creating and training the MLP model
        R_train_hat = NMF_model(R_train, n_components=20, max_iter=1000, alpha_W=1.5, alpha_H=1.5, solver='cd')

        # calculate f4
        loss = f4(R_train_hat, indicies_test, values_test)
        print(loss)

        # test_pred = np.zeros(len(self.test))
        # for i in range(len(self.test)):
        #     test_pred[i] = R_hat[
        #         np.where(self.test['user_id'][i] == self.users)[0][0],
        #         np.where(self.test['song_id'][i] == self.songs)[0][0]]
        #
        # # add test_pred to test as a column
        # test_pred_df = self.test.copy(deep=True)
        # test_pred_df['weight'] = test_pred
        # test_pred_df.to_csv('325482768_325765352_task4.csv', index=False)

        return loss


def NMF_model(R, n_components, max_iter=1000, alpha_W=1.5, alpha_H=1.5, solver='cd'):
    model = NMF(n_components=n_components, init='random', max_iter=max_iter, alpha_W=alpha_W, alpha_H=alpha_H, solver=solver)
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
    rec.q4()


main()
