import csv
import nltk
import sys
from TSum import TSum


def read_file(file_path):
    file_type = file_path.split(".")[-1]

    if file_type == "csv":
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            data = [row for row in reader][0]
    elif file_type == "txt":
        with open(file_path, "r") as f:
            data = f.read()
            data = nltk.sent_tokenize(data)
    else:
        raise ValueError("Invalid file type")

    return data


if __name__ == "__main__":
    file_path = sys.argv[1]

    text = read_file(file_path)

    tSum = TSum(device="cpu", context_window=9)

    summaries = tSum.execute(text)

    print(summaries)
