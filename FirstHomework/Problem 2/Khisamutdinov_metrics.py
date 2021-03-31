# - Если percent=None то метрика должна быть рассчитана по порогу вероятности >=0.5
# - Если параметр percent принимает значения от 1 до 100 то метрика должна быть рассчитана на соответствующем ТОПе
# - 1 - верхний 1% выборки
# - 100 - вся выборка
# - y_predict - имеет размерность (N_rows; N_classes)
import numpy as np


def choose_class(y_p, y_true, percent):
    y_p_1 = y_p[:, -1]
    quantile = np.percentile(a=y_p_1, q=100-percent)
    top_elements = [(y_p_1[i], i) for i in range(len(y_p_1)) if y_p_1[i] >= quantile]
    index = [i[1] for i in top_elements]
    top_data = []
    threshold = (100 - percent) / 100
    for element in top_elements:
        if element[0] > threshold:
            top_data.append(1)
        else:
            top_data.append(0)
    top_y = [y_true[j] for j in index]
    tp, fp, tn, fn = divide(top_y, top_data)
    return tp, fp, tn, fn


def divide(y_t, y_p):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(y_t)):
        if y_t[i] == 1 and y_p[i] == 1:
            tp += 1
        elif y_t[i] == 1 and y_p[i] == 0:
            fn += 1
        elif y_t[i] == 0 and y_p[i] == 0:
            tn += 1
        elif y_t[i] == 0 and y_p[i] == 1:
            fp += 1
    return tp, fp, tn, fn


def accuracy_score(y_true, y_pred, percent=50):
    tp, fp, tn, fn = choose_class(y_pred, y_true,  percent)
    return (tp + tn)/(tp + tn + fp + fn)


def precision_score(y_true, y_pred, percent=50):
    tp, fp, tn, fn = choose_class(y_pred, y_true,  percent)
    return tp / (tp + fp)


def recall_score(y_true, y_pred, percent=50):
    tp, fp, tn, fn = choose_class(y_pred, y_true,  percent)
    return tp / (tp + fn)


def lift_score(y_true, y_pred, percent=50):
    tp, fp, tn, fn = choose_class(y_pred, y_true,  percent)
    return (tp / (tp + fp)) / ((tp + fn) / (tp + tn + fp + fn))


def f1_score(y_true, y_pred, percent=50):
    tp, fp, tn, fn = choose_class(y_pred, y_true,  percent)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return 2 * (precision * recall) / (precision + recall)
