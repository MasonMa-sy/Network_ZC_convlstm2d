import numpy as np
import math


def pearson_distance(vector1, vector2):
    """
    Calculate distance between two vectors using pearson method
    See more : http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
    """
    sum1 = sum(vector1)
    sum2 = sum(vector2)

    sum1Sq = sum([pow(v, 2) for v in vector1])
    sum2Sq = sum([pow(v, 2) for v in vector2])

    pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])

    num = pSum - (sum1 * sum2 / len(vector1))
    den = math.sqrt((sum1Sq - pow(sum1, 2) / len(vector1)) * (sum2Sq - pow(sum2, 2) / len(vector1)))

    if den == 0: return 0.0
    return num / den


def calculate_rmse(y_true, y_pred):
    return math.sqrt(np.sum((y_true-y_pred)**2)/y_true.size)
