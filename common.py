import logging
import time
import numpy as onp

logger = logging.getLogger(__file__)


EPSILON = 1e-7
INPUTS = 1836
LABELS = 159


class Config(dict):
    def __getattr__(self, key):
        val = self[key]
        if isinstance(val, dict):
            val = Config(val)
        return val


def load_bibtex(train_file: str, test_file: str):
    def load_(dataset):
        with open(dataset) as f:
            while not next(f).startswith('@data'): pass
            size = sum(1 for _ in f)
            logger.info(f'dataset: {dataset}, size: {size}')
            f.seek(0)
            while not next(f).startswith('@data'): pass
            xs, ys = onp.zeros((size, INPUTS)), onp.zeros((size, LABELS))
            for i, line in enumerate(f):
                line = line.strip()[1:-1]
                entries = [int(entry[:-2]) for entry in line.split(',')]
                xs[i, [e for e in entries if e < INPUTS]] = 1
                ys[i, [e - INPUTS for e in entries if e >= INPUTS]] = 1
            return xs, ys
    train = load_(train_file)
    test = load_(test_file)
    dev = test
    return train, test, dev


def data_stream(xs, ys, batch_size=64, random_seed=None, infty=False):
    assert len(xs) == len(ys)
    if random_seed is not None:
        rng = onp.random.RandomState(random_seed)
        perm = rng.permutation(len(xs))
    else:
        perm = onp.arange(len(xs))
    num_batches = int(onp.ceil(len(xs) / batch_size))
    while True:
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield xs[batch_idx], ys[batch_idx]
        if not infty:
            return


def evaluate(params, inference_fun, xs, ys, batch_size, threshold=None):
    predicts = onp.zeros((len(xs), LABELS))
    total_time = 0
    for i, (x, _) in enumerate(data_stream(xs, ys, batch_size=batch_size)):
        start = time.time()
        predict_batch = inference_fun(params, x)
        total_time += time.time() - start
        predicts[i * batch_size:(i + 1) * batch_size] = predict_batch

    thresholds = threshold or [
        0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10,
        0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.60, 0.65, 0.70, 0.75
    ]
    if isinstance(thresholds, float):
        thresholds = [thresholds]

    best_f1 = best_accuracy = threshold_f1 = threshold_accuracy = -1.
    for threshold in thresholds:
        f1_score = compute_f1(predicts, ys, threshold)
        if f1_score > best_f1:
            best_f1 = f1_score
            threshold_f1 = threshold 

    for threshold in thresholds:
        accuracy = compute_accuracy(predicts, ys, threshold)
        if accuracy > best_accuracy:
            threshold_accuracy = threshold
            best_accuracy = accuracy

    logger.info(f'Speed (example / sec.) = {float(len(xs)) / total_time:.2f}')
    logger.info(f'Threshold: {threshold_f1:.4f}, F1 score: {best_f1:.4f}')
    logger.info(f'Threshold: {threshold_accuracy:.4f}, Accuracy: {best_accuracy:.4f}')
    return best_f1


def compute_f1(predicts, answers, threshold):
    outputs = onp.greater(predicts, threshold)
    tp = onp.count_nonzero(
        onp.isclose(2 * outputs - answers, 1), axis=1)
    fp = onp.count_nonzero(
        onp.isclose(outputs - answers, 1), axis=1)
    fn = onp.count_nonzero(
        onp.isclose(outputs - answers, -1), axis=1)
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1_total = (2 * precision * recall) / (precision + recall + EPSILON)
    f1 = onp.sum(f1_total) / len(outputs)
    return f1


def compute_accuracy(predicts, answers, threshold):
    # outputs = onp.greater(predicts, threshold)
    # difference = onp.sum(onp.abs(outputs - answers), axis=1)
    # correct = onp.count_nonzero(onp.isclose(difference, 0))

    outputs = onp.greater(predicts, threshold)
    return onp.mean((onp.abs(outputs - answers) == 0) * (answers != 0))
