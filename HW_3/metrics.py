import numpy as np


def levenstein(str_a, str_b) -> int:
    len_a, len_b = len(str_a), len(str_b)

    a = np.zeros((len_b + 1, len_a + 1), dtype=int)
    a[0] = [i for i in range(len_a + 1)]
    a[:, 0] = [i for i in range(len_b + 1)]

    for i in range(1, len_b + 1):
        for j in range(1, len_a + 1):
            if str_a[j - 1] == str_b[i - 1]:
                a[i, j] = a[i - 1, j - 1]
            else:
                a[i, j] = 1 + min(a[i, j - 1], a[i - 1, j], a[i - 1, j - 1])
    return a[-1, -1]


def cer(str_a, str_b) -> float:
    len_a, len_b = len(str_a), len(str_b)

    a = np.zeros((len_b + 1, len_a + 1), dtype=int)
    a[0] = [i for i in range(len_a + 1)]
    a[:, 0] = [i for i in range(len_b + 1)]

    for i in range(1, len_b + 1):
        for j in range(1, len_a + 1):
            if str_a[j - 1] == str_b[i - 1]:
                a[i, j] = a[i - 1, j - 1]
            else:
                a[i, j] = 1 + min(a[i, j - 1], a[i - 1, j], a[i - 1, j - 1])
    return a[-1, -1] / len(str_a)


def wer(str_a, str_b) -> float:
    str_a = str_a.split(' ')
    str_b = str_b.split(' ')

    len_a, len_b = len(str_a), len(str_b)

    a = np.zeros((len_b + 1, len_a + 1), dtype=int)
    a[0] = [i for i in range(len_a + 1)]
    a[:, 0] = [i for i in range(len_b + 1)]

    for i in range(1, len_b + 1):
        for j in range(1, len_a + 1):
            if str_a[j - 1] == str_b[i - 1]:
                a[i, j] = a[i - 1, j - 1]
            else:
                a[i, j] = 1 + min(a[i, j - 1], a[i - 1, j], a[i - 1, j - 1])
    return a[-1, -1] / len(str_a)
