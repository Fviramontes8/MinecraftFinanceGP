import matplotlib.pyplot as plt

from struct import calcsize, Struct
from numpy import atleast_2d, transpose, clip, isnan


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
        plt.plot(data[len(data) - 1], data[i])
        plt.show()


def calculate_timestamp_differences(data):
    timestamp_differences = []
    for i, j in zip(range(len(data) - 1), range(1, len(data))):
        print(f"i: {i}, j: {j}")
        timestamp_differences.append(data[i] - data[j])
    return timestamp_differences


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
        print(isnan(data[i]))
        data[i] = clip(data[i], None, 4)
    # plot_features(data)

    timestamp_diffs = calculate_timestamp_differences(data[8])
    print(f"Mean of differences: {sum(timestamp_diffs) / len(timestamp_diffs)}")


if __name__ == "__main__":
    main()
