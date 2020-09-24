# %%
import os
import time
import numpy as np
from functools import partial
from datasets import boston_data, regression_data
from models import run_cvxpy, unconstrained, sdr, lpr, sklearn_lr, maxmin, ste
from losses import matrix_linear, l1, l2, w_star_dist, np_loss, tf_huber
from misc import dump_evals, avg_evals, plot_dicts


# %%
def run_by_data_cb(cb_arr, metrics, methods, seed):
    eval_dict = {}
    inter_eval_dict = {}
    inter_vals = {}
    w_hats_arr = []
    i=0
    for cb in cb_arr:
        i += 1
        x_train, y_train, x_test, y_test, w_star = cb(seed=seed)
        w_hats = {name: run(y_train, x_train, seed=seed) for name, run in methods.items()}
        w_hats_arr.append(w_hats.copy())

        for k,v in w_hats.items():
            if type(v) is list:
                inter_vals[k] = v
                w_hats[k] = v[-1]

        for metric, metric_fun in metrics.items():
            for method_name, w_inter in inter_vals.items():
                test_metric = str(i) + '_test_' + metric
                train_metric = str(i) + '_train_' + metric
                if (inter_eval_dict.get(test_metric + '_w_gen') is None):
                    inter_eval_dict[train_metric + '_w_gen'] = {}
                    inter_eval_dict[test_metric + '_w_gen'] = {}

                inter_eval_dict[train_metric + '_w_gen'][method_name] = [metric_fun(y_train, x_train, w, w_star) for w in w_inter]
                inter_eval_dict[test_metric + '_w_gen'][method_name] = [metric_fun(y_test, x_test, w, w_star) for w in w_inter]

            for method_name, w_hat in w_hats.items():
                test_metric = 'test_' + metric
                train_metric = 'train_' + metric
                if (eval_dict.get(test_metric + '_w_gen') is None):
                    eval_dict[train_metric + '_w_gen'] = {}
                    eval_dict[test_metric + '_w_gen'] = {}

                if eval_dict[train_metric + '_w_gen'].get(method_name) is None:
                    eval_dict[train_metric + '_w_gen'][method_name] = []
                    eval_dict[test_metric + '_w_gen'][method_name] = []
                eval_dict[train_metric + '_w_gen'][method_name].append(metric_fun(y_train, x_train, w_hat, w_star))
                eval_dict[test_metric + '_w_gen'][method_name].append(metric_fun(y_test, x_test, w_hat, w_star))
    return eval_dict, inter_eval_dict, w_hats_arr 


# %%
if __name__ == "__main__":
    d = 30
    n_train = 60
    n_runs = 200
    n_iter = int(1e3)
    var_arr = np.linspace(0.2,1.5,6)
    # data_gen_cb = [partial(regression_data, var=var, noise_fn=np.random.laplace ,d=d, n_train=n_train) for var in var_arr]
    data_gen_cb = [partial(regression_data, var=var, noise_fn=np.random.normal ,d=d, n_train=n_train) for var in var_arr]

    params = {'d': d,
              'n_iter': n_iter,
              'n_train': n_train,
              }
    methods = {
        # 'LR l1': partial(unconstrained, obj_loss=l1),
        # 'STE l1': partial(ste, obj_loss=l1, n_iter=params['n_iter']),
        # 'MAXIMIN l1': partial(maxmin, obj_loss=l1, n_iter=params['n_iter']),
        # 'MAXIMIN l2': partial(maxmin, obj_loss=l2, n_iter=params['n_iter']),
        # 'SDR': partial(sdr, obj_loss=matrix_linear),
        # 'LPR l1': partial(lpr, obj_loss=l1),

        'LR l2': partial(unconstrained, obj_loss=l2),
        'STE l2': partial(ste, obj_loss=l2, n_iter=params['n_iter']),
        'MAXIMIN l1': partial(maxmin, obj_loss=l1, n_iter=params['n_iter']),
        'MAXIMIN l2': partial(maxmin, obj_loss=l2, n_iter=params['n_iter']),
        'SDR': partial(sdr, obj_loss=matrix_linear),
        'LPR l2': partial(lpr, obj_loss=l2),
    }

    metrics = {

        'w_star_dist': w_star_dist,
        'w_star_dist_round': lambda y, X, w, w_star: w_star_dist(y, X, np.sign(w), w_star),
        
        # 'l1_loss': partial(np_loss, loss=l1),

        # 'l2_loss': partial(np_loss, loss=l2),
        'l2_loss_round': lambda y, X, w, w_star: np_loss(y, X, np.sign(w), w_star,loss=l2),
    }

    params_str = ''.join([k + '-' + str(v) + '_' for k, v in params.items()])
    base_path = os.path.join('plots', time.strftime('%m-%d-%H%M%S') + params_str)

    eval_dicts = []

    os.mkdir(base_path)
    for seed in range(1, n_runs+1):

        eval_dict, inter_dict, w_hats_arr = run_by_data_cb(data_gen_cb, metrics, methods, seed)
        save_path = os.path.join(base_path, str(seed))
        os.mkdir(save_path)
        # plot_dicts(var_arr, eval_dict, save_path, save=True)
        # plot_dicts(np.linspace(0, params['n_iter'], 101), inter_dict, save_path, save=True)
        eval_dicts.append(eval_dict)
        dump_evals(save_path, eval_dict)

    avg_eval_dict = avg_evals(eval_dicts)
    save_path = os.path.join(base_path, 'avged')
    os.mkdir(save_path)
    plot_dicts(10.*np.log(1./var_arr), avg_eval_dict, save_path, save=False)


# %%
