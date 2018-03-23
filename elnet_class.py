import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet



class ElasticNetPredictor:
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False,
                 max_iter=1000, copy_X=True, tol=1e-4, warm_start=False, positive=False, random_state=None,
                 selection='cyclic'):
        self.feature_to_predict = None
        self.words_data = None
        self.documents_data = None
        self.model = ElasticNet(alpha, l1_ratio, fit_intercept, normalize, precompute, max_iter, copy_X, tol,
                                warm_start, positive, random_state, selection)
        self.label = None
        self.features = None

    @staticmethod
    def _get_collinear_columns(df, threshold):
        good_list = list(df[df > threshold].stack().index)
        ans = set()
        setleft = set()
        good_list = set(good_list)
        while len(good_list) > 0:
            t = good_list.pop()

            good_list = set([x for x in set(good_list) if x[1] != t[1]])
            good_list = set([x for x in set(good_list) if x[0] != t[1]])

            ans.add(t[1])
        return list(set(df.columns).difference(ans))

    def fit(self, train_feature, train_labels):
        print('Start fitting')
        self.model.fit(train_feature, train_labels)

    def process_data(self, documents_data=None, words_data=None, previous_periods=2, previous_words=1,
                     word_sample_size=4000, collinear_threshhold=0.56):
        assert documents_data is not None or self.documents_data is not None
        assert words_data is not None or self.words_data is not None
        if documents_data is not None:
            self.documents_data = documents_data
        if words_data is not None:
            self.words_data = words_data

        titles = self.documents_data.values
        n_hours, n_titles = titles.shape

        words = self.words_data.values
        n_word_hours, n_words = words.shape
        assert n_hours == n_word_hours
        self.number_of_titles = n_titles
        self.number_of_periods = n_hours
        self.num_of_words = n_words
        mean = titles.mean(axis=0)
        std = titles.std(axis=0)
        sumed_words = words.cumsum(axis=0)[previous_words + 1:, :] - words.cumsum(axis=0)[:-previous_words - 1, :]
        sumed_words /= previous_words
        sum_std = sumed_words.std(axis=0)
        sum_mean = sumed_words.mean(axis=0)
        sum_std += 1

        sample = np.random.randint(low=sumed_words.shape[1], size=word_sample_size)
        sumed_words = sumed_words[:, sample]
        sum_mean = sum_mean[sample]
        sum_std = sum_std[sample]
        data_sum = pd.DataFrame(sumed_words)
        corr = data_sum.corr(method='pearson')
        corr.fillna(0, inplace=True)
        corr -= np.eye(corr.shape[0])
        corr = np.abs(corr)

        rem = self._get_collinear_columns(corr, collinear_threshhold)
        sumed_words = sumed_words[:, rem]
        sum_mean = sum_mean[rem]
        sum_std = sum_std[rem]
        target = titles[previous_periods:, :]

        sumed_words = sumed_words[-target.shape[0] - 1:, :]
        target = np.hstack((target, sumed_words[1:, :]))
        feature = np.zeros((target.shape[0], previous_periods * n_titles * 2 + sumed_words.shape[1]))
        for i in np.arange(previous_periods, n_hours):
            feature[i - previous_periods, :] = np.hstack((np.hstack((titles[i - previous_periods:i, :] / std,
                                                                     (titles[i - previous_periods:i,
                                                                      :] - mean) / std)).ravel(),
                                                          (sumed_words[i - previous_periods, :] - sum_mean) / sum_std))
        self.label = target
        self.features = feature
        sum_size = sumed_words.shape[1]
        self.number_of_used_words = sum_size
        self.used_words_mean = sum_mean
        self.used_words_std = sum_std
        self.feature_mean = mean
        self.feature_std = std

        cur_feature = np.hstack((self.features[-1, 2 * n_titles:-sum_size], self.label[-1, :-sum_size] / std,
                                 (self.label[-1, :-sum_size] - mean) / std,
                                 (self.label[-1, -sum_size:] - sum_mean) / sum_std))
        self.feature_to_predict = cur_feature
        self.fit(self.features, self.label)

    def predict(self, number_of_periods=7):
        cur_feature = self.feature_to_predict.copy()
        prediction = np.zeros((number_of_periods, self.label.shape[1]))
        for i in range(number_of_periods):
            predicted = self.model.predict(cur_feature.reshape((1, -1)))
            prediction[i, :] = predicted
            cur_feature = np.hstack((cur_feature[2 * self.number_of_titles:-self.number_of_used_words],
                                     predicted.ravel()[:-self.number_of_used_words] / self.feature_std,
                                     (predicted.ravel()[:-self.number_of_used_words] - self.feature_mean) / self.feature_std,
                                     (predicted.ravel()[-self.number_of_used_words:] - self.used_words_mean) / self.used_words_std))
        return prediction

    def friendly_prediction(self, number_of_periods=7):
        prediction = self.predict(number_of_periods)
        df = {self.documents_data.columns[i] : prediction[:, i] for i in range(self.number_of_titles)}
        return pd.DataFrame(df)

    def compute_proba(self, number_of_periods=7):
        data = self.friendly_prediction(number_of_periods)
        data = data.apply(lambda row: row / np.sum(row), axis=1)
        return data

    
class Formater:
    def __init__(self, model=ElasticNetPredictor(), row_per_day=12, previous_days=4, previous_words=1,
                 words_to_perform=4000, collinear_threshold=0.56):
        self.model = model
        self.row_per_day = row_per_day
        self.previous_days = previous_days
        self.previous_words = previous_words
        self.words_to_perform = words_to_perform
        self.collinear_threshold = collinear_threshold

    def fit(self, counted_data, word_data):
        self.model.process_data(counted_data, word_data, previous_periods=int(self.row_per_day*self.previous_days),
                                previous_words=int(self.row_per_day*self.previous_words),
                                word_sample_size=self.words_to_perform, collinear_threshhold=self.collinear_threshold)
        self.last_day_count = counted_data.values[-self.row_per_day:, :].sum(axis=0)

    def predict(self, days_to_predict=7):
        num_titles = self.model.number_of_titles
        predicted_values = self.model.friendly_prediction(days_to_predict * self.row_per_day)
        predicted_probas = self.model.compute_proba(days_to_predict * self.row_per_day)
        values = predicted_values.values
        values = values.reshape((self.row_per_day, num_titles, days_to_predict))
        summed_val = values.sum(axis=0)

        probas = predicted_probas.values
        probas = probas.reshape((self.row_per_day, num_titles, days_to_predict))
        daily_probas = probas.mean(axis=0)
        
        summed_val = np.hstack((self.last_day_count.reshape((-1, 1)), summed_val))
        daily_probas = np.hstack((self.last_day_count.reshape((-1, 1)) / self.last_day_count.sum(), daily_probas))
        
        daily_forecast = []
        for i in np.arange(days_to_predict):
            daily_forecast.append([{'topic':predicted_values.columns[j], 'documents': summed_val[j, i + 1], 'diff_documents': summed_val[j, i + 1] - summed_val[j, i], 'percentage': daily_probas[j, i + 1], 'diff_percentage' : daily_probas[j, i + 1] - daily_probas[j, i]} for j in range(len(predicted_values.columns))])
        return daily_forecast