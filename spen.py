import numpy as onp
import jax.numpy as np
import jax.random as npr
from jax import jit, value_and_grad, grad, lax, vmap
from jax.experimental.optimizers import l2_norm, momentum, adam
from jax.nn.initializers import glorot_normal, normal
from jax.nn import relu, softplus, log_sigmoid, sigmoid
import jax
import time
from collections import namedtuple
from functools import partial
from tqdm import tqdm
from enum import Enum
import logging
from absl import app, flags


EPSILON = 1e-7
INPUTS = 1836
LABELS = 159

logger = logging.getLogger(__file__)

FLAGS = flags.FLAGS
flags.DEFINE_integer('feature_size', 200, '')
flags.DEFINE_integer('label_units', 15, '')
flags.DEFINE_integer('hidden_units', 150, '')
flags.DEFINE_integer('random_seed', 0, 'random seed')
flags.DEFINE_integer('pretrain_epoch', 30, 'number of epochs for feature network pretraining (1st stage)')
flags.DEFINE_integer('energy_pretrain_epoch', 30, 'number of epochs for energy network pretraining (2nd stage)')
flags.DEFINE_integer('e2e_train_epoch', 30, 'number of epochs for end-to-end training (final stage)')
flags.DEFINE_integer('pretrain_batch_size', 64, 'batch size used in feature network pretraining')
flags.DEFINE_integer('ssvm_batch_size', 32, 'batch size used in energy pretraining and end-to-end training')
flags.DEFINE_boolean('debug',  False,'debug mode')
flags.DEFINE_string('train', None, 'train dataset')
flags.DEFINE_string('test',  None,'test dataset')


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


def evaluate(params, xs, ys, batch_size, pretrained=False):
    predicts = onp.zeros((len(xs), LABELS))
    total_time = 0
    inference_fun = inference_pretrained if pretrained else inference
    for i, (x, _) in enumerate(data_stream(xs, ys, batch_size=batch_size)):
        start = time.time()
        predict_batch = inference_fun(params, x)
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


Param = namedtuple('Param', ['A_1', 'A_2', 'B', 'C_1', 'c_2'])


def init_dense(rng, shape, W_init=glorot_normal(), b_init=normal()):
    assert len(shape) == 2
    k1, k2 = npr.split(rng)
    return W_init(k1, shape), b_init(k2, (shape[-1],))


def dense(params, inputs):
    W, b = params
    return np.matmul(inputs, W) + b


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



def project(x):
    return np.clip(x, EPSILON, 1 - EPSILON)


@jit
def inference(params, x, max_steps=20, step_size=0.1):
    x_hat = compute_feature_energy(params, x)
    opt_init, opt_update, get_params = momentum(0.1, 0.95)
    opt_state = opt_init(np.full(x.shape[:-1] + (LABELS,), 1. / LABELS))
    prev_pred_energy = None
    for i in range(max_steps):
        y_hat = project(get_params(opt_state))
        pred_energy, g = value_and_grad(lambda y: compute_global_energy(params, x_hat, y).sum())(y_hat)
        opt_state = opt_update(i, g, opt_state)
        # if np.all(np.isclose(get_params(opt_state), y_hat)):
        #     print('achieve tolerance with y_hat', i)
        #     break
        # if prev_pred_energy is not None and np.isclose(pred_energy, prev_pred_energy):
        #     print('achieve tolerance with energy', i)
        #     break
        # prev_pred_energy = pred_energy
    return get_params(opt_state)


def ssvm_loss(params, x, y, lamb=0.001, max_steps=20, step_size=0.1, pretrain_global_energy=False):
    x_hat = compute_feature_energy(params, x)
    if pretrain_global_energy:
        x_hat = lax.stop_gradient(x_hat)
    x_hat = compute_feature_energy(params, x)
    def cost_augment_inference_step(y_hat):
        delta = np.square(y_hat - y).sum(axis=1)
        energy = compute_global_energy(params, x_hat, y_hat)
        pred_energy = -1 * delta + energy
        return pred_energy.mean(), pred_energy

    opt_init, opt_update, get_params = momentum(0.1, 0.95)
    opt_state = opt_init(np.full_like(y, 1. / LABELS))
    prev_pred_energy = None
    for i in range(max_steps):
        y_hat = project(get_params(opt_state))
        g, pred_energy = grad(cost_augment_inference_step, has_aux=True)(y_hat)
        opt_state = opt_update(i, g, opt_state)
        # if np.all(np.isclose(get_params(opt_state), y_hat)):
        #     print('achieve tolerance with y_hat', i)
        #     break
        # if prev_pred_energy is not None and np.all(np.isclose(pred_energy, prev_pred_energy)):
        #     print('achieve tolerance with energy', i)
        #     break
        # prev_pred_energy = pred_energy

    true_energy = compute_global_energy(params, x_hat, y)
    loss = np.mean(np.maximum(true_energy - pred_energy, 0))
    return loss + lamb * l2_norm(params)


def pretrain_loss(params, x, y, lamb=0.001):
    neglogprob = - np.mean(np.log(inference_pretrained(params, x)) * y)
    return neglogprob + lamb * l2_norm(params)


@jit
def inference_pretrained(params, x):
    x_hat = compute_feature_energy(params, x)
    return sigmoid(np.matmul(x_hat, params.B))


class Stage(Enum):
    PretrainFeatureNetwork = 0
    PretrainEnergyNetwork = 1
    End_to_end = 2


def main(unused_argv):
    logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO)

    (train_xs, train_ys), (test_xs, test_ys), (dev_xs, dev_ys) \
        = load_bibtex('bibtex-train.arff', 'bibtex-test.arff')

    init_params = init_param(npr.PRNGKey(FLAGS.random_seed),
                             input_units=INPUTS,
                             label_size=LABELS,
                             feature_size=FLAGS.feature_size,
                             label_units=FLAGS.label_units,
                             hidden_units=FLAGS.hidden_units)


    opt_init, opt_update, get_params = adam(0.1)  # momentum(0.1, 0.95)
    opt_state = opt_init(init_params)


    @jit
    def update_pretrain(i, opt_state, batch):
        params = get_params(opt_state)
        loss, g = value_and_grad(pretrain_loss)(params, *batch)
        return opt_update(i, g, opt_state), loss

    # @jit
    def update_ssvm(i, opt_state, batch, pretrain_global_energy=False):
        params = get_params(opt_state)
        loss, g = value_and_grad(ssvm_loss)(params, *batch, pretrain_global_energy=pretrain_global_energy)
        return opt_update(i, g, opt_state), loss

    stages = {
        Stage.PretrainFeatureNetwork: {
            'batch_size': FLAGS.pretrain_batch_size,
            'epochs': FLAGS.pretrain_epoch,
            'update_fun': update_pretrain,
            'msg': 'pretraining feature network'
        },
        Stage.PretrainEnergyNetwork: {
            'batch_size': FLAGS.ssvm_batch_size,
            'epochs': FLAGS.energy_pretrain_epoch,
            'update_fun': partial(update_ssvm, pretrain_global_energy=True),
            'msg': 'pretraining energy network'
        },
        Stage.End_to_end: {
            'batch_size': FLAGS.ssvm_batch_size,
            'epochs': FLAGS.e2e_train_epoch,
            'update_fun': update_ssvm,
            'msg': 'finetune the entire network end-to-end'
        }
    }

    for name, stage in stages.items():
        logger.info(stage['msg'])
        num_batches = int(onp.ceil(len(train_xs) / stage['batch_size']))
        for epoch in range(stage['epochs']):
            train_stream = data_stream(train_xs, train_ys, batch_size=stage['batch_size'], random_seed=epoch)
            step_loss = 0.
            for i, (xs, ys) in tqdm(enumerate(train_stream), total=num_batches):
                opt_state, loss = stage['update_fun'](i, opt_state, (xs, ys))
                step_loss += loss
            logger.info(f'epoch: {epoch} loss = {step_loss / num_batches}')
            evaluate(get_params(opt_state), test_xs, test_ys, batch_size=stage['batch_size'], pretrained=name == Stage.PretrainFeatureNetwork)



if __name__ == '__main__':
    app.run(main)

