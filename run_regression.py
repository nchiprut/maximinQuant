# %%
import argparse
import os
import time
import numpy as np
from functools import partial
from datasets import boston_data, regression_data
from models import run_cvxpy, unconstrained, sdr, lpr, sklearn_lr, maxmin, ste,scipy_lpr
from losses import matrix_linear, l1, l2, w_star_dist, np_sum_loss, np_mean_loss
from misc import dump_evals, avg_evals, plot_dicts, print_evals_stat


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
                if (inter_eval_dict.get(test_metric) is None):
                    inter_eval_dict[train_metric] = {}
                    inter_eval_dict[test_metric] = {}

                inter_eval_dict[train_metric][method_name] = [metric_fun(y_train, x_train, w, w_star) for w in w_inter]
                inter_eval_dict[test_metric][method_name] = [metric_fun(y_test, x_test, w, w_star) for w in w_inter]

            for method_name, w_hat in w_hats.items():
                test_metric = 'test_' + metric
                train_metric = 'train_' + metric
                if (eval_dict.get(test_metric) is None):
                    eval_dict[train_metric] = {}
                    eval_dict[test_metric] = {}

                if eval_dict[train_metric].get(method_name) is None:
                    eval_dict[train_metric][method_name] = []
                    eval_dict[test_metric][method_name] = []
                eval_dict[train_metric][method_name].append(metric_fun(y_train, x_train, w_hat, w_star))
                eval_dict[test_metric][method_name].append(metric_fun(y_test, x_test, w_hat, w_star))
    return eval_dict, inter_eval_dict, w_hats_arr 

def evals_by_data_cb(args, methods, metrics, data_gen_cb):
    eval_dicts = []
    for seed in range(1, args.n_runs+1):
        print('run #', seed)

        eval_dict, inter_dict, w_hats_arr = run_by_data_cb(data_gen_cb, metrics, methods, seed)
        # save_path = os.path.join(args.base_path, str(seed))
        # os.mkdir(save_path)
        eval_dicts.append(eval_dict)
        # dump_evals(save_path, eval_dict)

    avg_eval_dict = avg_evals(eval_dicts)
    # save_path = os.path.join(args.base_path, 'avged')
    # os.mkdir(save_path)
    # dump_evals(save_path, eval_dict)
    return avg_eval_dict, eval_dicts

def quad_params(args):
    methods = {
        # 'LR_cvx l2': partial(unconstrained, obj_loss=l2),
        'LR_sk l2': partial(sklearn_lr, obj_loss=l2),
        'STE l2': partial(ste, obj_loss=l2, n_iter=args.n_iter),
        'MAXIMIN l2': partial(maxmin, obj_loss=l2, n_iter=args.n_iter, d_epochs=10),
        'MAXIMIN l1': partial(maxmin, obj_loss=l1, n_iter=args.n_iter, d_epochs=10),
        'SDR': partial(sdr, obj_loss=matrix_linear),
        # 'LPR_cvx l2': partial(lpr, obj_loss=l2),
        'LPR_spy l2': partial(scipy_lpr, obj_loss=l2),
    }
    metrics = {
        'w_star_dist': w_star_dist,
        'w_star_dist_round': lambda y, X, w, w_star: w_star_dist(y, X, np.sign(w), w_star),
        'l2_loss_round': lambda y, X, w, w_star: np_mean_loss(y, X, np.sign(w), w_star,loss=l2),
        'NRMSE': lambda y, X, w, w_star: np.sqrt(np_sum_loss(y, X, np.sign(w), w_star,loss=l2)) / np.sqrt(y.T @ y).item(),
    }
    return methods, metrics

def huber_params(args):
    methods = {
        'LR l1': partial(unconstrained, obj_loss=l1),
        'STE l1': partial(ste, obj_loss=l1, n_iter=args.n_iter),
        'MAXIMIN l1': partial(maxmin, obj_loss=l1, n_iter=args.n_iter, d_epochs=10),
        'MAXIMIN l2': partial(maxmin, obj_loss=l2, n_iter=args.n_iter, d_epochs=10),
        'SDR': partial(sdr, obj_loss=matrix_linear),
        'LPR l1': partial(lpr, obj_loss=l1),
    }
    metrics = {
        'w_star_dist': w_star_dist,
        'w_star_dist_round': lambda y, X, w, w_star: w_star_dist(y, X, np.sign(w), w_star),
        'l2_loss_round': lambda y, X, w, w_star: np_mean_loss(y, X, np.sign(w), w_star,loss=l2),
        'l1_loss_round': lambda y, X, w, w_star: np_mean_loss(y, X, np.sign(w), w_star,loss=l1),
        'NRMSE': lambda y, X, w, w_star: np.sqrt(np_sum_loss(y, X, np.sign(w), w_star,loss=l2)) / np.sqrt(y.T @ y).item(),
    }
    return methods, metrics

def main(args):
    noise_fn = np.random.normal if args.loss == 'l2' else np.random.laplace
    if args.loss == 'l2':
        noise_fn = np.random.normal
        methods, metrics = quad_params(args)
    elif args.loss == 'l1':
        noise_fn = np.random.laplace
        methods, metrics = huber_params(args)
    if args.ds == 'synthetic':
        ds = regression_data
        data_gen_cb = [partial(regression_data, var=var, noise_fn=noise_fn ,d=args.dim, n_train=args.n_train) for var in args.var]
        x_axis = 10.*np.log(1./np.array(args.var))
    elif args.ds == 'boston':
        ds = boston_data
        data_gen_cb = [partial(boston_data, p_out=p_out) for p_out in args.p_out]
        x_axis = args.p_out

    avg_evals, evals = evals_by_data_cb(args, methods, metrics, data_gen_cb)
    params_str = ''
    base_path = os.path.join('plots', time.strftime('%m-%d-%H%M%S') + params_str)
    os.mkdir(base_path)
    dump_evals(base_path, 'evals.yaml', evals)
    dump_evals(base_path, 'evals.yaml', avg_evals)
    if args.plt is not None:
        plot_dicts(x_axis, avg_evals, base_path, save=False, logscale=args.plt == 'logscale')
    return avg_evals, evals


# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--ds', default='synthetic', choices=['synthetic', 'boston'], help='')
    parser.add_argument('--loss', default='l2', choices=['l1', 'l2'], help='')
    parser.add_argument('--var', default=np.linspace(0.15,.6,6), nargs='+', type=float, help='')
    parser.add_argument('--p_out', default=[0], nargs='+', type=float, help='')
    parser.add_argument('--lr', default=1e-2, type=float, help='')
    parser.add_argument('--n_iter', default=int(1e3), type=int, help='')
    parser.add_argument('--dim', default=30, type=int, help='')
    parser.add_argument('--n_train', default=60, type=int, help='')
    parser.add_argument('--n_test', default=None, type=int, help='')
    parser.add_argument('--n_runs', default=60, type=int, help='')
    parser.add_argument('--plt', default=None, choices=['logscale', 'linear', None],)

    # args = parser.parse_args(['--loss', 'l2','--n_run', '200','--plt', 'logscale'])                                   #synthetic l2
    # args = parser.parse_args(['--loss', 'l1','--n_run', '200','--plt', 'logscale'])                                   #synthetic l1
    # args = parser.parse_args(['--ds', 'boston','--loss', 'l2','--n_run', '20', '--n_iter', '10000'])                  #boston l2
    # args = parser.parse_args(['--ds', 'boston','--loss', 'l1','--n_run', '20','--p_out', '.25', '--n_iter', '10000']) #boston l1
    args = parser.parse_args([])

    avg_eval_dict, eval_dicts = main(args)
    # print_evals_stat(eval_dicts)                                                                                      # print boston stats
