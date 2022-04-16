import matplotlib.pyplot as plt

from struct import calcsize, Struct
from numpy import atleast_2d, transpose, clip, histogram
from math import fabs

from sklearn.svm import SVR


def read_source_data(filename, struct_unpacker, struct_length):
    results = []
    with open(filename, "rb") as f:
        while True:
            data = f.read(struct_length)
            if not data:
                break
            results.append(struct_unpacker(data))

    return results


# Assumes 9 by N matrix where indices 0 - 7 are features and 8 is a time-series of timestamps
def plot_features(data):
    for i in range(len(data) - 2):
        plt.title(f"Graph: {i}")
        plt.plot(data[len(data) - 1], data[i])
        plt.show()


def calculate_timestamp_differences(data):
    timestamp_differences = []
    for i, j in zip(range(len(data) - 1), range(1, len(data))):
        timestamp_differences.append(fabs(data[i] - data[j]))
    return timestamp_differences


def plot_histogram(data):
    for i in range(len(data)):
        timestamp_differences = calculate_timestamp_differences(data[i])
        timestamp_differences = timestamp_differences[:500]

        data_histogram, data_bins = histogram(timestamp_differences, bins='auto')
        print(data_histogram)
        print(data_bins)
        plt.hist(data_histogram, data_bins)
        plt.title(f"Histogram of data[{i}]")
        plt.ylabel("Frequency")
        plt.xlabel("Value")
        plt.show()


# Create linear SVM regressor
# Create linear GP regressor
def main():
    # Data format is in big endian (requires >) and is a sequence of 9 numbers: 8 doubles (8d) and 1 8-byte long int (q)
    struct_format = ">8dq"
    struct_length = calcsize(struct_format)
    struct_unpack = Struct(struct_format).unpack_from

    filename = "shortdata.out"
    data = atleast_2d(
        read_source_data(
            filename,
            struct_unpack,
            struct_length
        )
    )
    data = transpose(data)
    for i in range(len(data) - 2):
        data[i] = clip(data[i], None, 4)

    # plot_histogram(data)

    # plot_features(data)
    ratio = 0.8
    cutoff = int(len(data[4]) * ratio)
    train_x = data[8][:cutoff].reshape(-1, 1)
    train_y = data[4][:cutoff]
    test_x = data[8][cutoff + 1:].reshape(-1, 1)
    test_y = data[4][cutoff + 1:]

    print(f"train_x shape: {train_x.shape}")
    print(f"train_y shape: {train_y.shape}")
    print(f"test_x shape: {test_x.shape}")
    print(f"test_y shape: {test_y.shape}")

    C = [0.001, 0.01, 0.1]
    for i in C:
        linear_svm_model = SVR(kernel='linear', C=i)
        rbf_svm_model = SVR(kernel='rbf', C=i)

        print(f"Training linear SVR")
        linear_svm_model.fit(train_x, train_y)
        print(f"Training RBF SVR")
        rbf_svm_model.fit(train_x, train_y)

        print(f"Testing with Linear SVR")
        linear_pred = linear_svm_model.predict(test_x)
        print(f"Testing with RBF SVR")
        rbf_pred = rbf_svm_model.predict(test_x)

        # print(f"linear prediction shape: {linear_pred.shape}")
        # print(f"rbf prediction shape: {rbf_pred.shape}")
        plt.plot(test_x[:, 0], test_y, "k", label="Actual")
        # plt.plot(test_x[:, 0], linear_pred, "m", label="Linear prediction")
        plt.plot(test_x[:, 0], rbf_pred, "c", label="RBF prediction")
        plt.title(f"Plot of Linear SVR and RBF SVR for C = {i}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
