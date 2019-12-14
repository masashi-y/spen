#%%
import numpy as onp
import jax.numpy as np
import jax.random as npr
from jax import jit, value_and_grad, grad, lax, vmap
from jax.experimental.optimizers import l2_norm, adam
from jax.nn.initializers import glorot_normal, normal
from jax.nn import relu, softplus, log_sigmoid
import jax
import time
from collections import namedtuple
import logging

EPSILON = 1e-7
INPUTS = 1836
LABELS = 159

#%%
logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.DEBUG)


#%%
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

# %%
(train_xs, train_ys), (test_xs, test_ys), (dev_xs, dev_ys) = \
    load_bibtex('bibtex-train.arff', 'bibtex-test.arff')

# %%
Param = namedtuple('Param', ['A_1', 'A_2', 'B', 'C_1', 'c_2'])

# %%
def init_dense(rng, shape, W_init=glorot_normal(), b_init=normal()):
    assert len(shape) == 2
    k1, k2 = npr.split(rng)
    return W_init(k1, shape), b_init(k2, (shape[-1],))


def dense(params, inputs):
    W, b = params
    return np.matmul(inputs, W) + b

# %%
def init_param(rng, input_units, feature_size, label_size, label_units, hidden_units):
    init = glorot_normal()
    k1, k2, k3, k4, k5 = npr.split(rng, num=5)
    A_1 = init_dense(k1, (input_units, hidden_units))
    A_2 = init_dense(k2, (hidden_units, feature_size))
    B = init(k3, (feature_size, label_size))
    C_1 = init_dense(k4, (feature_size, label_units))
    c_2 = init(k5, (label_units, 1))
    return Param(A_1, A_2, B, C_1, c_2)


def compute_feature_energy(params, x):
    A_1, A_2, *_ = params
    return relu(dense(A_2, relu(dense(A_1, x))))


def compute_global_energy(params, x_hat, y):
    _, _, B, C_1, c_2 = params
    E_loc = np.mean(np.matmul(x_hat, B) * y, axis=1, keepdims=True)  # (batch size, 1)
    E_lab = np.matmul(softplus(dense(C_1, x_hat)), c_2)
    return E_loc + E_lab


def compute_spen(params, x, y):
    x_hat = compute_feature_energy(params, x)
    y_hat = compute_global_energy(params, x_hat, y)
    return y_hat

# %%
rng = npr.PRNGKey(0)
init_params = init_param(rng,
                         input_units=INPUTS,
                         feature_size=200,
                         label_size=LABELS,
                         label_units=15,
                         hidden_units=150)


# %%
def project(x):
    return np.clip(x, EPSILON, 1 - EPSILON)


@jit
def inference(params, x, steps=10, step_size=0.1):
    x_hat = compute_feature_energy(params, x)
    y_hat = np.zeros(x.shape[:-1] + (LABELS,))
    energy_grad_fun = grad(lambda y: compute_global_energy(params, x_hat, y).sum())
    def loop_fun(i, y_hat):
        g = energy_grad_fun(y_hat)
        return project(y_hat - step_size * g)
    return lax.fori_loop(0, steps, loop_fun, y_hat)


def ssvm_loss(params, x, y, steps=10, step_size=0.1):
    x_hat = compute_feature_energy(params, x)
    def cost_augment_inference_step(y_hat):
        delta = np.square(y_hat - y).sum(axis=1)
        energy = compute_global_energy(params, x_hat, y_hat)
        pred_energy = -1 * delta + energy
        return pred_energy.mean(), pred_energy
    y_hat = np.zeros_like(y)
    step = grad(cost_augment_inference_step, has_aux=True)
    for _ in range(steps):
        g, pred_energy = step(y_hat)
        y_hat = project(y_hat - step_size * g)
    true_energy = compute_global_energy(params, x_hat, y)
    return np.mean(np.maximum(true_energy - pred_energy, 0))


ssvm_loss_grad_fun = jit(value_and_grad(ssvm_loss))


# %%
def pretrain_loss(params, x, y):
    x_hat = compute_feature_energy(params, x)
    return - np.mean(log_sigmoid(np.matmul(x_hat, params.B)) * y)

pretrain_loss_grad_fun = jit(value_and_grad(pretrain_loss))


# %%
def data_stream(xs, ys, batch_size=64, random_seed=None):
    assert len(xs) == len(ys)
    if random_seed is not None:
        rng = onp.random.RandomState(random_seed)
        perm = rng.permutation(len(xs))
    else:
        perm = onp.arange(len(xs))
    num_batches = int(onp.ceil(len(xs) / batch_size))
    for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield xs[batch_idx], ys[batch_idx]
# %%
stream = data_stream(train_xs, train_ys)


# %%
def evaluate(params, xs, ys, batch_size):
    predicts = onp.zeros((len(xs), LABELS))
    total_time = 0
    for i, (x, _) in enumerate(data_stream(xs, ys, batch_size=batch_size)):
        start = time.time()
        predict_batch = inference(params, x)
        total_time += time.time() - start
        predicts[i * batch_size:(i + 1) * batch_size] = predict_batch

    thresholds = [
        0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10,
        0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.60, 0.65, 0.70, 0.75
    ]
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


def compute_f1(predicts, answers, threshold):
    outputs = np.greater(predicts, threshold)
    tp = onp.count_nonzero(
        onp.isclose(2 * outputs - answers, 1), axis=1)
    fp = onp.count_nonzero(
        onp.isclose(outputs - answers, 1), axis=1)
    fn = onp.count_nonzero(
        onp.isclose(outputs - answers, -1), axis=1)
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1_total = (2 * precision * recall) / (precision + recall + EPSILON)
    f1 = np.sum(f1_total) / len(outputs)
    return f1

def compute_accuracy(predicts, answers, threshold):
    outputs = onp.greater(predicts, threshold)
    difference = onp.sum(onp.abs(outputs - answers), axis=1)
    correct = onp.count_nonzero(onp.isclose(difference, 0))
    return correct / len(outputs)



# %%
opt_init, opt_update, get_params = adam(0.1)
opt_state = opt_init(init_params)

@jit
def update_pretrain(i, opt_state, batch):
    params = get_params(opt_state)
    loss, g = pretrain_loss_grad_fun(params, *batch)
    return opt_update(i, g, opt_state), loss

@jit
def update_ssvm(i, opt_state, batch):
    params = get_params(opt_state)
    loss, g = ssvm_loss_grad_fun(params, *batch)
    return opt_update(i, g, opt_state), loss


for seed in range(10):
    logger.info('pretraining')
    for i, (xs, ys) in enumerate(data_stream(train_xs, train_ys, batch_size=64, random_seed=seed)):
        opt_state, loss = update_pretrain(i, opt_state, (xs, ys))
        print(loss)

for seed in range(20):
    logger.info('ssvm training')
    for i, (xs, ys) in enumerate(data_stream(train_xs, train_ys, batch_size=64, random_seed=seed)):
        opt_state, loss = update_ssvm(i, opt_state, (xs, ys))
        print(loss)
    evaluate(get_params(opt_state), test_xs, test_ys, batch_size=64)

# %%
