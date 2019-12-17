
import numpy as onp
import jax.numpy as np
import jax.random as npr
from jax import jit, value_and_grad, grad, lax, vmap
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu
from jax.experimental.optimizers import l2_norm, momentum, adam, sgd
from jax.nn.initializers import glorot_normal, normal
from jax.nn import relu, softplus, log_sigmoid, sigmoid
from collections import namedtuple
import itertools
from tqdm import tqdm
import logging
from absl import app, flags

from common import INPUTS, LABELS
from common import load_bibtex, data_stream, evaluate, compute_f1, compute_accuracy


logger = logging.getLogger(__file__)


init_mlp, apply_mlp = stax.serial(
        Dense(150), Relu,
        Dense(200), Relu,
        Dense(LABELS)
)

def sigmoid_cross_entropy(x, y):
    pos_logprob = log_sigmoid(x) * y
    neg_logprob = -softplus(x) * (1 - y)
    return np.sum(pos_logprob + neg_logprob, axis=1)


def cross_entropy_loss(params, x, y, lamb=0.001):
    neglogprob = - np.mean(sigmoid_cross_entropy(- apply_mlp(params, x), y))
    return neglogprob + lamb * l2_norm(params)


@jit
def inference(params, x):
    return sigmoid(- apply_mlp(params, x))


FLAGS = flags.FLAGS
flags.DEFINE_integer('random_seed', 0, 'random seed')
flags.DEFINE_integer('epochs', 30, 'number of epochs for training MLP')
flags.DEFINE_integer('batch_size', 64, 'batch size used')
flags.DEFINE_boolean('debug',  False,'debug mode')
flags.DEFINE_string('train', None, 'train dataset')
flags.DEFINE_string('test',  None,'test dataset')


def main(unused_argv):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.DEBUG if FLAGS.debug else logging.INFO)

    (train_xs, train_ys), (test_xs, test_ys), (dev_xs, dev_ys) \
        = load_bibtex('bibtex-train.arff', 'bibtex-test.arff')

    _, init_params = init_mlp(npr.PRNGKey(FLAGS.random_seed), (FLAGS.batch_size, INPUTS))

    opt_init, opt_update, get_params = adam(0.001)
    # opt_init, opt_update, get_params = momentum(0.001, 0.9)
    # opt_init, opt_update, get_params = sgd(0.001)
    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        loss, g = value_and_grad(cross_entropy_loss)(params, *batch)
        return opt_update(i, g, opt_state), loss

    num_batches = int(onp.ceil(len(train_xs) / FLAGS.batch_size))
    train_stream = data_stream(train_xs, train_ys, batch_size=FLAGS.batch_size, random_seed=FLAGS.random_seed, infty=True)
    itercount = itertools.count()
    best_f1 = 0.
    for epoch in range(FLAGS.epochs):
        step_loss = 0.
        for _ in tqdm(range(num_batches)):
            opt_state, loss = update(next(itercount), opt_state, next(train_stream))
            step_loss += loss
        logger.info(f'epoch: {epoch} loss = {step_loss / num_batches}')
        f1 = evaluate(get_params(opt_state), inference, test_xs, test_ys, batch_size=FLAGS.batch_size, threshold=0.5)
        if f1 > best_f1:
            best_f1 = f1
    logger.info(f'best F1 score = {best_f1}')



if __name__ == '__main__':
    app.run(main)

