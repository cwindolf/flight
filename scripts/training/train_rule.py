import numpy as np
import tensorflow as tf
from src.models.rule2vec import Rule2Vec
from src.systems import cell
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import seaborn
import warnings
from multiprocessing import Pool
warnings.filterwarnings('ignore')


# *************************************************************************** #
# Model params

EMBEDDING_DIMS = 64
HIDDEN_LAYER_SIZE = 18
LEARNING_RATE = 1e-4
TRAIN_STEPS = Rule2Vec.NUM_RULES
BATCH_SIZE = 256
REPORT_IVAL = 100

# *************************************************************************** #
# Helper for multiprocessing

def tup_no_pad_step(tup):
    return cell.no_pad_step(*tup)

# *************************************************************************** #
# Init model and placeholders

prev_states = tf.placeholder(tf.int32, shape=(BATCH_SIZE, 3, 3), name='PrevState')
next_centers = tf.placeholder(tf.int32, shape=BATCH_SIZE, name='NextCenter')
rules = tf.placeholder(tf.int32, shape=BATCH_SIZE, name='RuleCode')
r2v = Rule2Vec(prev_states, rules, next_centers,
               EMBEDDING_DIMS, HIDDEN_LAYER_SIZE, LEARNING_RATE, BATCH_SIZE)

# *************************************************************************** #
# Do training

with tf.Session() as sess:
    try:
        p = Pool(4)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        inds, losses, percents = [], [], []
        plt.ion()
        ax = plt.gca()
        plt.title('Score')
        ax.set_autoscale_on(True)
        pline, = ax.plot(inds, percents, label='Percent')
        lline, = ax.plot(inds, losses, label='Loss')
        plt.legend()
        report_pct, report_loss = 0.0, 0.0
        for i in trange(1, TRAIN_STEPS, smoothing=0.7, ncols=100):
            # Input data and true value
            rule_codes = np.random.randint(r2v.NUM_RULES, size=BATCH_SIZE)
            in_states = np.random.randint(2, size=(BATCH_SIZE, 3, 3))
            out_states = np.asarray(p.map(tup_no_pad_step, zip(in_states, rule_codes)))

            # Train
            pct, loss, _ = sess.run([r2v.percent_correct, r2v.loss, r2v.train_op], feed_dict={
                    prev_states: in_states,
                    next_centers: out_states[:, 0, 0],
                    rules: rule_codes
                })

            # Show progress
            report_pct += pct
            report_loss += loss
            if not i % REPORT_IVAL:
                percents.append(report_pct / REPORT_IVAL)
                losses.append(report_loss * (100 / (BATCH_SIZE * REPORT_IVAL)))
                inds.append(i)
                pline.set_xdata(inds)
                pline.set_ydata(percents)
                lline.set_xdata(inds)
                lline.set_ydata(losses)
                ax.relim()
                ax.autoscale_view(True, True, True)
                plt.draw()
                plt.pause(0.01)
                report_pct = 0.0
                report_loss = 0.0
    finally:
        # Save Embeddings
        np.savez('E.npz', r2v.E.eval())
        while True:
            plt.pause(100)



