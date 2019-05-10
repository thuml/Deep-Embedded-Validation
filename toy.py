import numpy as np

from sklearn.linear_model import LinearRegression

from scipy.stats import norm


def f(x):
    return np.sin(x * np.pi) / (x * np.pi)


def w(x):
    return norm.pdf(x, loc=2, scale=0.25) / norm.pdf(x, loc=1, scale=0.5)


test_means = []
test_stds = []
train_means = []
train_stds = []
iwcv_means = []
iwcv_stds = []
dev_means = []
dev_stds = []

for i in range(10):
    lambda_value = 0.1 * i

    test_errors = []
    train_errors = []
    iwcv_errors = []
    dev_errors = []
    for i in range(1000):
        training_X = np.random.randn(150, 1) * 0.5 + 1
        training_Y = f(training_X) + np.random.randn(150, 1) * 0.25

        density_ratio = w(training_X)

        weight = density_ratio ** lambda_value

        model = LinearRegression()

        model.fit(training_X, training_Y, weight.flatten())

        test_X = np.random.randn(150, 1) * 0.25 + 2
        test_Y = f(test_X) + np.random.randn(150, 1) * 0.25

        predict_Y = model.predict(test_X)

        predict_train_Y = model.predict(training_X)

        test_error = np.mean((test_Y - predict_Y) ** 2)

        train_error = np.mean((training_Y - predict_train_Y) ** 2)

        wl = density_ratio * ((training_Y - predict_train_Y) ** 2)
        weighted_val_error = np.mean(wl)

        cov = np.cov(np.concatenate((wl, density_ratio), axis=1), rowvar=False)[0][1]
        var_w = np.var(density_ratio, ddof=1)
        c = - cov / var_w

        dev_error = weighted_val_error + c * np.mean(density_ratio) - c

        test_errors.append(test_error)
        train_errors.append(train_error)
        iwcv_errors.append(weighted_val_error)
        dev_errors.append(dev_error)

    mean, std = np.mean(test_errors), np.std(test_errors, ddof=1)
    test_means.append(mean)
    test_stds.append(std)

    mean, std = np.mean(train_errors), np.std(train_errors, ddof=1)
    train_means.append(mean)
    train_stds.append(std)

    mean, std = np.mean(iwcv_errors), np.std(iwcv_errors, ddof=1)
    iwcv_means.append(mean)
    iwcv_stds.append(std)

    mean, std = np.mean(dev_errors), np.std(dev_errors, ddof=1)
    dev_means.append(mean)
    dev_stds.append(std)
