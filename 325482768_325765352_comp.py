import numpy as np
import pandas as pd
import scipy
import sklearn


from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF

np.random.seed(0)



import matplotlib.pyplot as plt

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
        for k in [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]:
            u, s, vt = scipy.sparse.linalg.svds(R_train, k=k)
            # calculate R_hat
            R_train_hat = np.dot(np.dot(u, np.diag(s)), vt)
            loss = f4(R_train_hat, indicies_test, values_test)
            print(f'for k={k}, the loss is {loss}')

        # R_train = np.asarray(R_train.todense())
        # # Creating and training the MLP model
        # h_values = [1.5, 2, 2.5, 3, 5]
        # for h in h_values:
        #     R_train_hat = NMF_model(R_train, n_components=40, max_iter=2500, alpha_W=1.5, alpha_H=h, solver='cd')
        #     # calculate f4
        #     loss = f4(R_train_hat, indicies_test, values_test)
        #     print(f'for h = {h}, the loss is {loss}')

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

    def q3(self):
        k=150
        # calculate PCA with k of R
        R = self.R.astype(np.float64)
        u, s, vt = scipy.sparse.linalg.svds(R, k=k)

        plt.plot(range(k), np.flip(s))
        plt.show()
        # calculate R_hat
        R_hat = np.dot(np.dot(u, np.diag(s)), vt)


    def q2(self):
        k = 10
        N = 300000
        P = np.random.normal(size=(len(self.users), k))
        Q = np.zeros((len(self.songs), k))
        dense_R = self.R.todense()

        f2 = lambda P, Q: np.sum([(row['weight'] - np.dot(P[np.where(row['user_id'] == self.users)[0][0]],
                                                          Q[np.where(row['song_id'] == self.songs)[0][0]])) ** 2
                                  for _, row in self.train.iterrows()])

        song_indexes = [[i for i in range(len(self.users)) if dense_R[i, j] != 0] for j in range(len(self.songs))]
        user_indexes = [[j for j in range(len(self.songs)) if dense_R[i, j] != 0] for i in range(len(self.users))]

        b_j = [np.array([dense_R[i, j] for i in song_indexes[j]].append(0)) for j in range(len(self.songs))]
        b_i = [np.array([dense_R[i, j] for j in user_indexes[i]].append(0)) for i in range(len(self.users))]

        count = 0
        lamda = 2
        prev_loss = f2(P, Q)
        while True:
            count += 1
            print(f'iteration {count} loss is {prev_loss}')
            for j in range(len(self.songs)):
                A_j = [P[i, :] for i in song_indexes[j]]
                A_j.append([lamda] * k)

                Q[j, :] = np.linalg.lstsq(A_j, b_j[j], rcond=None)[0]

            for i in range(len(self.users)):
                A_i = np.array([Q[j, :] for j in user_indexes[i]].append([lamda] * k))
                P[i, :] = np.linalg.lstsq(A_i, b_i[i], rcond=None)[0]

            new_loss = f2(P, Q)
            if prev_loss - new_loss < N:
                break
            prev_loss = new_loss
        print(f'num of iterations: {count}')





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
    rec.q2()


main()
