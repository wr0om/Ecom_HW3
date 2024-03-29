import numpy as np
import pandas as pd
import scipy
import sklearn

np.random.seed(0)


class Recommender:

    def __init__(self, train, test):
        self.R = None
        self.songs = None
        self.users = None
        self.train = train
        self.test = test
        self.fit()

    def fit(self):
        self.users = self.train['user_id'].unique()
        self.songs = self.train['song_id'].unique()
        indices = []
        values = []
        for i in range(len(self.train)):
            indices.append((np.where(self.train['user_id'][i] == self.users)[0][0],
                            np.where(self.train['song_id'][i] == self.songs)[0][0]))
            values.append(self.train['weight'][i])

        self.R = scipy.sparse.coo_matrix((values, zip(*indices)), shape=(len(self.users), len(self.songs)))

    def q1(self):
        user_num, song_num = self.R.shape
        b_size = user_num + song_num
        count_non_zero = self.R.nnz
        r_avg = self.R.sum() / count_non_zero

        c = np.zeros(count_non_zero)
        indices = []
        values = []
        k = 0
        dense_R = self.R.todense()
        for row, col in zip(*self.R.nonzero()):
            indices.append((k, row))
            indices.append((k, user_num + col))
            values.append(1)
            values.append(1)
            c[k] = dense_R[row, col] - r_avg
            k += 1
        A = scipy.sparse.coo_matrix((values, zip(*indices)), shape=(count_non_zero, b_size))

        b = scipy.sparse.linalg.lsqr(A, c)[0]
        f1 = np.sum([(row['weight'] -
                      (r_avg + b[np.where(row['user_id'] == self.users)[0][0]] + b[
                          user_num + np.where(row['song_id'] == self.songs)[0][0]])) ** 2
                     for _, row in self.train.iterrows()])

        test_pred = np.zeros(len(self.test))
        for i in range(len(self.test)):
            user_index = np.where(self.test['user_id'][i] == self.users)[0][0]
            song_index = np.where(self.test['song_id'][i] == self.songs)[0][0]
            test_pred[i] = r_avg + b[user_index] + b[user_num + song_index]

        # deep copy of test
        test_pred_df = self.test.copy(deep=True)
        test_pred_df['weight'] = test_pred
        test_pred_df.to_csv('325482768_325765352_task1.csv', index=False)
        return f1

    def q2(self):
        k = 20
        N = 300000
        P = np.random.normal(size=(len(self.users), k))
        Q = np.zeros((len(self.songs), k))
        dense_R = self.R.todense()

        f2 = lambda P, Q: np.sum([(row['weight'] - np.dot(P[np.where(row['user_id'] == self.users)[0][0]],
                                                          Q[np.where(row['song_id'] == self.songs)[0][0]])) ** 2
                                  for _, row in self.train.iterrows()])

        song_indexes = [[i for i in range(len(self.users)) if dense_R[i, j] != 0] for j in range(len(self.songs))]
        user_indexes = [[j for j in range(len(self.songs)) if dense_R[i, j] != 0] for i in range(len(self.users))]

        b_j = [np.array([dense_R[i, j] for i in song_indexes[j]]) for j in range(len(self.songs))]
        b_i = [np.array([dense_R[i, j] for j in user_indexes[i]]) for i in range(len(self.users))]

        count = 0
        prev_loss = f2(P, Q)
        while True:
            count += 1
            print(f'iteration {count} loss is {prev_loss}')
            for j in range(len(self.songs)):
                A_j = np.array([P[i, :] for i in song_indexes[j]])
                Q[j, :] = np.linalg.lstsq(A_j, b_j[j], rcond=None)[0]

            for i in range(len(self.users)):
                A_i = np.array([Q[j, :] for j in user_indexes[i]])
                P[i, :] = np.linalg.lstsq(A_i, b_i[i], rcond=None)[0]

            new_loss = f2(P, Q)
            if prev_loss - new_loss < N:
                break
            prev_loss = new_loss
        print(f'num of iterations: {count}')

        test_pred = np.zeros(len(self.test))
        for i, row in self.test.iterrows():
            user_index = np.where(row['user_id'] == self.users)[0][0]
            song_index = np.where(row['song_id'] == self.songs)[0][0]
            test_pred[i] = np.dot(P[user_index], Q[song_index].T)

        # add test_pred to test as a column
        test_pred_df = self.test.copy(deep=True)
        test_pred_df['weight'] = test_pred
        test_pred_df.to_csv('325482768_325765352_task2.csv', index=False)
        return new_loss

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

        test_pred = np.zeros(len(self.test))
        for i in range(len(self.test)):
            user_index = np.where(self.test['user_id'][i] == self.users)[0][0]
            song_index = np.where(self.test['song_id'][i] == self.songs)[0][0]
            test_pred[i] = R_hat[user_index, song_index]

        # add test_pred to test as a column
        test_pred_df = self.test.copy(deep=True)
        test_pred_df['weight'] = test_pred
        test_pred_df.to_csv('325482768_325765352_task3.csv', index=False)
        return f3


def main():
    train = pd.read_csv('user_song.csv')
    test = pd.read_csv('test.csv')
    rec = Recommender(train, test)

    # f1 = rec.q1()
    # print(f"f1 = {f1}")
    f2 = rec.q2()
    print(f"f2 = {f2}")
    # f3 = rec.q3()
    # print(f"f3 = {f3}")


main()