import tensorflow as tf
import tensorflow_probability as tfp
import xarray as xr
import numpy as np


def param_bounds(dtype):
    if isinstance(dtype, tf.DType):
        min = dtype.min
        max = dtype.max
        dtype = dtype.as_numpy_dtype
    else:
        dtype = np.dtype(dtype)
        min = np.finfo(dtype).min
        max = np.finfo(dtype).max

    sf = dtype(2.5)
    bounds_min = {
        "a": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "b": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "log_mu": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "log_r": np.log(np.nextafter(0, np.inf, dtype=dtype)) / sf,
        "mu": np.nextafter(0, np.inf, dtype=dtype),
        "r": np.nextafter(0, np.inf, dtype=dtype),
        "probs": dtype(0),
        "log_probs": np.log(np.nextafter(0, np.inf, dtype=dtype)),
    }
    bounds_max = {
        "a": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "b": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "log_mu": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "log_r": np.nextafter(np.log(max), -np.inf, dtype=dtype) / sf,
        "mu": np.nextafter(max, -np.inf, dtype=dtype) / sf,
        "r": np.nextafter(max, -np.inf, dtype=dtype) / sf,
        "probs": dtype(1),
        "log_probs": dtype(0),
    }
    return bounds_min, bounds_max


def clip_param(param, name):
    bounds_min, bounds_max = param_bounds(param.dtype)
    return tf.clip_by_value(
        param,
        bounds_min[name],
        bounds_max[name]
    )


data = xr.open_dataset("example.h5", group="data")
params = xr.open_dataset("example.h5", group="params")

X = tf.convert_to_tensor(data.X.values, dtype="float32", name="X")
design_loc = tf.convert_to_tensor(data.design_loc.values, dtype="float32", name="design_loc")
design_scale = tf.convert_to_tensor(data.design_scale.values, dtype="float32", name="design_scale")
init_a = tf.convert_to_tensor(params.a.values, dtype="float32", name="a")
init_b = tf.convert_to_tensor(params.b.values, dtype="float32", name="b")

param_vec = tf.Variable(tf.concat([init_a, init_b], axis=0), name="param_vec")

p_shape_a = init_a.shape[0]
p_shape_b = init_b.shape[0]

a, b = tf.split(param_vec, tf.TensorShape([p_shape_a, p_shape_b]))


def loss_fn(X, design_loc, design_scale, a, b):
    with tf.name_scope("mu"):
        log_mu = tf.matmul(design_loc, a, name="log_mu_obs")
        log_mu = clip_param(log_mu, "log_mu")
        mu = tf.exp(log_mu)

    with tf.name_scope("r"):
        log_r = tf.matmul(design_scale, b, name="log_r_obs")
        log_r = clip_param(log_r, "log_r")
        r = tf.exp(log_r)

    p = mu / (r + mu)
    dist_obs = tfp.distributions.NegativeBinomial(probs=p, total_count=r)

    with tf.name_scope("log_probs"):
        log_probs = dist_obs.log_prob(X)
        log_probs = clip_param(log_probs, "log_probs")

    norm_neg_log_likelihood = - tf.reduce_mean(log_probs, axis=0, name="log_likelihood")

    with tf.name_scope("loss"):
        return tf.reduce_sum(norm_neg_log_likelihood)


# ### SCIPY ###
loss = loss_fn(X, design_loc, design_scale, a, b)

scipy_bfgs = tf.contrib.opt.ScipyOptimizerInterface(
    loss,
    method='L-BFGS-B',
)


# ### BFGS ###
def value_and_grad_fn(param_vec):
    a_split, b_split = tf.split(param_vec, tf.TensorShape([p_shape_a, p_shape_b]))

    loss = loss_fn(X, design_loc, design_scale, a_split, b_split)

    return loss, tf.gradients(loss, param_vec)[0]


hessian = tf.hessians(loss, param_vec)[0]
# tfp_bfgs = tfp.optimizer.bfgs_minimize(value_and_grad_fn, param_vec, initial_inverse_hessian_estimate=hessian)
tfp_bfgs = tfp.optimizer.bfgs_minimize(value_and_grad_fn, param_vec)

# ### Init session ###
init_op = tf.global_variables_initializer()

sess = tf.Session()

# ### RUN ADAM ###
sess.run(init_op)
print("Begin SciPy BFGS; current loss:\t", sess.run(loss))
scipy_bfgs.minimize(sess)
print(sess.run(loss))
print("Loss after SciPy BFGS:\t", sess.run(loss))

# ### RE-INIT AND RUN BFGS ###
sess.run(init_op)
print("Begin tfp BFGS; current loss:\t", sess.run(loss))
res = sess.run(tfp_bfgs)
print("tfp BFGS finished:")
print(res)
