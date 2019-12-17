import numpy as onp
import jax.numpy as np
import jax.random as npr
from jax import jit, value_and_grad, grad, lax, vmap
from jax.experimental.optimizers import l2_norm, momentum, adam
from jax.nn.initializers import glorot_normal, normal
from jax.nn import relu, softplus, log_sigmoid, sigmoid
import jax
from collections import namedtuple
from functools import partial
import itertools
from tqdm import tqdm
import logging
from absl import app, flags

from common import EPSILON, INPUTS, LABELS
from common import Config, load_bibtex, data_stream, evaluate, compute_f1, compute_accuracy


logger = logging.getLogger(__file__)


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
    C_1 = init_dense(k4, (label_size, label_units))
    c_2 = init(k5, (label_units, 1))
    return Param(A_1, A_2, B, C_1, c_2)


@jit
def compute_feature_energy(params, x):
    A_1, A_2, *_ = params
    return relu(dense(A_2, relu(dense(A_1, x))))


@jit
def compute_global_energy(params, x_hat, y):
    _, _, B, C_1, c_2 = params
    E_loc = np.sum(np.matmul(x_hat, B) * y, axis=1, keepdims=True)  # (batch size, 1)
    E_lab = np.matmul(softplus(dense(C_1, y)), c_2)
    return E_loc + E_lab


def project(x):
    return np.clip(x, EPSILON, 1 - EPSILON)


def sigmoid_cross_entropy(x, y):
    pos_logprob = log_sigmoid(x) * y
    neg_logprob = -softplus(x) * (1 - y)
    return np.sum(pos_logprob + neg_logprob, axis=1)


def check_saddle_point(step, y, prev_y, energy, prev_energy):
    if np.all(np.max(np.abs(y - prev_y), axis=1) < 0.01):
        logger.debug(f'achieve tolerance with y_hat at {step}th step')
        return True
    if np.all(np.isclose(energy, prev_energy)):
        logger.debug(f'achieve tolerance with energy at {step}th step')
        return True
    return False


# @jit
# def inference(params, x, max_steps=80, step_size=0.1):
#     x_hat = compute_feature_energy(params, x)
#     opt_init, opt_update, get_params = momentum(0.1, 0.95)
#     opt_state = opt_init(np.full(x.shape[:-1] + (LABELS,), 1. / LABELS))
#     prev_pred_energy = None
#     for i in range(max_steps):
#         y_hat = project(get_params(opt_state))
#         pred_energy, g = value_and_grad(lambda y: compute_global_energy(params, x_hat, y).sum())(y_hat)
#         opt_state = opt_update(i, g, opt_state)
#         # if np.all(np.isclose(get_params(opt_state), y_hat)):
#         #     print('achieve tolerance with y_hat', i)
#         #     break
#         # if prev_pred_energy is not None and np.isclose(pred_energy, prev_pred_energy):
#         #     print('achieve tolerance with energy', i)
#         #     break
#         # prev_pred_energy = pred_energy
#     return get_params(opt_state)


def inference(y_hat, y, x_hat, params):
    energy = compute_global_energy(params, x_hat, y_hat)
    return energy.mean(), energy
inference_step = jit(grad(inference, has_aux=True))


def cost_augment_inference(y_hat, y, x_hat, params):
    delta = np.square(y_hat - y).sum(axis=1)
    energy = -1 * delta + compute_global_energy(params, x_hat, y_hat)
    return energy.mean(), energy
cost_augmented_inference_step = jit(grad(cost_augment_inference, has_aux=True))


def ssvm_loss(params, x, y, lamb=0.001, max_steps=40, step_size=0.1, pretrain_global_energy=False):
    prediction = y is None
    x_hat = compute_feature_energy(params, x)
    if pretrain_global_energy:
        x_hat = lax.stop_gradient(x_hat)
    grad_fun = inference_step if prediction else cost_augmented_inference_step

    opt_init, opt_update, get_params = momentum(0.1, 0.95)
    # opt_state = opt_init(np.full(x.shape[:-1] + (LABELS,), 1. / LABELS))
    opt_state = opt_init(np.zeros(x.shape[:-1] + (LABELS,)))
    prev_energy = None
    for step in range(max_steps):
        y_hat = project(get_params(opt_state))
        g, energy = grad_fun(y_hat, y, x_hat, params)
        opt_state = opt_update(step, g, opt_state)
        if step > 0 and check_saddle_point(
                step, get_params(opt_state), y_hat, energy, prev_energy):
            break
        prev_energy = energy

    y_hat = lax.stop_gradient(project(get_params(opt_state)))
    if prediction:
        return y_hat

    pred_energy = compute_global_energy(params, x_hat, y_hat)
    true_energy = compute_global_energy(params, x_hat, y)
    delta = np.square(y_hat - y).sum(axis=1)
    loss = np.mean(np.maximum(delta + true_energy - pred_energy, 0))
    return loss + lamb * l2_norm(params)


def inference(params, x, **kwargs):
    return ssvm_loss(params, x, None, max_steps=80, pretrain_global_energy=False, **kwargs)


@jit
def inference_pretrained(params, x):
    x_hat = compute_feature_energy(params, x)
    return - np.matmul(x_hat, params.B)


def pretrain_loss(params, x, y, lamb=0.001):
    neglogprob = - np.mean(sigmoid_cross_entropy(inference_pretrained(params, x), y))
    return neglogprob + lamb * l2_norm(params)


FLAGS = flags.FLAGS
flags.DEFINE_integer('feature_size', 200, '')
flags.DEFINE_integer('label_units', 15, '')
flags.DEFINE_integer('hidden_units', 150, '')
flags.DEFINE_integer('random_seed', 0, 'random seed')
flags.DEFINE_integer('pretrain_epoch', 100, 'number of epochs for feature network pretraining (1st stage)')
flags.DEFINE_integer('energy_pretrain_epoch', 30, 'number of epochs for energy network pretraining (2nd stage)')
flags.DEFINE_integer('e2e_train_epoch', 10, 'number of epochs for end-to-end training (final stage)')
flags.DEFINE_integer('pretrain_batch_size', 64, 'batch size used in feature network pretraining')
flags.DEFINE_integer('ssvm_batch_size', 32, 'batch size used in energy pretraining and end-to-end training')
flags.DEFINE_boolean('debug',  False,'debug mode')
flags.DEFINE_string('train', None, 'train dataset')
flags.DEFINE_string('test',  None,'test dataset')


def main(unused_argv):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.DEBUG if FLAGS.debug else logging.INFO)

    (train_xs, train_ys), (test_xs, test_ys), (dev_xs, dev_ys) \
        = load_bibtex('bibtex-train.arff', 'bibtex-test.arff')

    init_params = init_param(npr.PRNGKey(FLAGS.random_seed),
                             input_units=INPUTS,
                             label_size=LABELS,
                             feature_size=FLAGS.feature_size,
                             label_units=FLAGS.label_units,
                             hidden_units=FLAGS.hidden_units)


    opt_init, opt_update, get_params = adam(0.01)  # momentum(0.1, 0.95)
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

    stages = [
        Config(
            batch_size=FLAGS.pretrain_batch_size,
            epochs=FLAGS.pretrain_epoch,
            update_fun=update_pretrain,
            inference_fun=inference_pretrained,
            msg='pretraining feature network'
        ),
        Config(
            batch_size=FLAGS.ssvm_batch_size,
            epochs=FLAGS.energy_pretrain_epoch,
            update_fun=partial(update_ssvm, pretrain_global_energy=True),
            inference_fun=inference,
            msg='pretraining energy network'
        ),
        Config(
            batch_size=FLAGS.ssvm_batch_size,
            epochs=FLAGS.e2e_train_epoch,
            update_fun=update_ssvm,
            inference_fun=inference,
            msg='finetune the entire network end-to-end'
        )
    ]
    best_f1 = 0.
    for stage in stages:
        logger.info(stage.msg)
        num_batches = int(onp.ceil(len(train_xs) / stage.batch_size))
        train_stream = data_stream(train_xs, train_ys, batch_size=stage.batch_size, random_seed=FLAGS.random_seed, infty=True)
        itercount = itertools.count()
        for epoch in range(stage.epochs):
            step_loss = 0.
            for _ in tqdm(range(num_batches)):
                opt_state, loss = stage.update_fun(next(itercount), opt_state, next(train_stream))
                step_loss += loss
            logger.info(f'epoch: {epoch} loss = {step_loss / num_batches}')
            f1 = evaluate(get_params(opt_state), stage.inference_fun, test_xs, test_ys, batch_size=stage.batch_size)
            if f1 > best_f1:
                best_f1 = f1
    logger.info(f'best F1 score = {best_f1}')



if __name__ == '__main__':
    app.run(main)

