# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pytz
import datetime
from pathlib import Path
import sys
import scipy
import scipy.sparse as sp
import numpy as np
import signal
import pickle
import random
import requests
import re
from time import time
from threading import Timer
import subprocess
import klepto
from collections import OrderedDict
from socket import gethostname
from os import makedirs, system, environ, remove
from os.path import dirname, abspath, exists, join, isfile, expanduser
import os
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import mstats
import matplotlib
from pathlib import Path
import math
from copy import deepcopy

matplotlib.use('pdf')


def get_root_path():
    return dirname(dirname(abspath(__file__)))


def get_log_path():
    return join(get_root_path(), 'logs')


def get_file_path():
    return join(get_root_path(), 'file')


def get_save_path():
    return join(get_root_path(), 'save')


def get_src_path():
    return join(get_root_path(), 'src')


def create_dir_if_not_exists(dir):
    if not exists(dir):
        makedirs(dir)


def _get_y(data):
    return getattr(data, FLAGS.target.replace('-', '_'))


def _get_y_with_target(data, target):
    return getattr(data, target.replace('-', '_'))


def _get_y_multi_obj(data):
    assert isinstance(FLAGS.target, list)
    y_list = [getattr(data, t.replace('-', '_')) for t in FLAGS.target]
    return y_list


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def save(obj, filepath, print_msg=True, use_klepto=True):
    if type(obj) is not dict and type(obj) is not OrderedDict:
        raise ValueError(
            'Can only save a dict or OrderedDict' ' NOT {}'.format(type(obj))
        )
    fp = proc_filepath(filepath, ext='.klepto' if use_klepto else '.pickle')
    if use_klepto:
        create_dir_if_not_exists(dirname(filepath))
        save_klepto(obj, fp, print_msg)
    else:
        save_pickle(obj, fp, print_msg)
    return fp


def load(filepath, print_msg=True):
    fp = proc_filepath(filepath)
    if os.path.exists(fp):
        return load_klepto(fp, print_msg)
    elif print_msg:
        print('Trying to load but no file {}'.format(fp))


def save_klepto(dic, filepath, print_msg):
    if print_msg:
        print('Saving to {}'.format(filepath))
    klepto.archives.file_archive(filepath, dict=dic).dump()


def load_klepto(filepath, print_msg):
    rtn = klepto.archives.file_archive(filepath)
    rtn.load()
    if print_msg:
        print('Loaded from {}'.format(filepath))
    return rtn


# protocal shouldn't be too high to avoid loading issue
def save_pickle(obj, filepath, print_msg=True, protocal=4):
    if print_msg:
        print('Saving to {}'.format(filepath))
    with open(filepath, 'wb',encoding='utf-8') as handle:
        if sys.version_info.major < 3:  # python 2
            pickle.dump(obj, handle)
        elif sys.version_info >= (3, 4):  # qilin & feilong --> 3.4
            pickle.dump(obj, handle, protocol=protocal)
        else:
            raise NotImplementedError()


def load_pickle(filepath, print_msg=True):
    if isfile(filepath):
        with open(filepath, 'rb',encoding='utf-8') as handle:
            pickle_data = pickle.load(handle)
            if print_msg:
                print(f'Loaded pickle from {filepath}')
            return pickle_data
    elif print_msg:
        print('No file {}'.format(filepath))


def proc_filepath(filepath, ext='.klepto'):
    if type(filepath) is not str:
        raise RuntimeError(
            f'Did you pass a file path to this function? type(filepath)={type(filepath)}; filepath={filepath}'
        )
    return append_ext_to_filepath(ext, filepath)


def append_ext_to_filepath(ext, fp):
    if not fp.endswith(ext):
        fp += ext
    return fp


def sorted_nicely(l, reverse=False):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    def alphanum_key(s):
        if type(s) is not str:
            raise ValueError('{} must be a string in l: {}'.format(s, l))
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    rtn = sorted(l, key=alphanum_key)
    if reverse:
        rtn = reversed(rtn)
    return rtn


global_exec_print = True


def exec_turnoff_print():
    global global_exec_print
    global_exec_print = False


def exec_turnon_print():
    global global_exec_print
    global_exec_print = True


def global_turnoff_print():
    import sys
    import os

    sys.stdout = open(os.devnull, 'w',encoding='utf-8')


def global_turnon_print():
    import sys

    sys.stdout = sys.__stdout__


def exec_cmd(cmd, timeout=None, exec_print=True):
    global global_exec_print
    if not timeout:
        if global_exec_print and exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
    else:

        def kill_proc(proc, timeout_dict):
            timeout_dict["value"] = True
            proc.kill()

        def run(cmd, timeout_sec):
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            timeout_dict = {"value": False}
            timer = Timer(timeout_sec, kill_proc, [proc, timeout_dict])
            timer.start()
            stdout, stderr = proc.communicate()
            timer.cancel()
            return (
                proc.returncode,
                stdout.decode("utf-8"),
                stderr.decode("utf-8"),
                timeout_dict["value"],
            )

        if global_exec_print and exec_print:
            print('Timed cmd {} sec(s) {}'.format(timeout, cmd))
        _, _, _, timeout_happened = run(cmd, timeout)
        if global_exec_print and exec_print:
            print('timeout_happened?', timeout_happened)
        return not timeout_happened


tstamp = None


def get_ts():
    global tstamp
    if not tstamp:
        tstamp = get_current_ts()
    return tstamp


def set_ts(ts):
    global tstamp
    tstamp = ts


def get_current_ts(zone='US/Pacific'):
    return datetime.datetime.now(pytz.timezone(zone)).strftime('%Y-%m-%dT%H-%M-%S.%f')


class timeout:
    """
    https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# def get_user():
#     try:
#         home_user = expanduser("~").split('/')[-1]
#     except:
#         home_user = 'user'
#     return home_user

def get_user():
    """
    返回当前用户的登录名，确保它是干净的字符串，不包含路径分隔符。
    兼容 Windows 和类 Unix 系统。
    """
    username = None

    # 1. 尝试从环境变量获取 (在 Windows 上通常是 USERNAME, 在 Linux/macOS 上是 USER)
    username = os.environ.get('USERNAME') # 适用于 Windows
    if username is None:
        username = os.environ.get('USER') # 适用于 Linux/macOS

    # 2. 如果环境变量没有，尝试 os.getlogin()
    if username is None:
        try:
            username = os.getlogin()
        except OSError:
            # os.getlogin() 在某些非交互式环境或服务中可能失败
            pass

    # 3. 如果以上方法都未获取到，从用户主目录路径中提取最后一个部分作为用户名
    if username is None:
        try:
            # expanduser("~") 返回主目录，例如 C:\Users\huijie 或 /home/huijie
            # basename 会正确处理 Windows 和 Unix 路径分隔符
            username = os.path.basename(expanduser("~"))
        except:
            # 如果获取主目录也失败，提供一个默认值
            username = 'unknown_user'

    # 4. 最后，清理用户名，确保它不包含任何文件系统不允许的字符
    if username:
        # 移除冒号和所有类型的斜杠，用下划线替换
        username = username.replace(':', '_').replace('\\', '_').replace('/', '_')
    else:
        username = 'unknown_user' # 如果最终还是空，给一个默认值

    return username

def get_host():
    host = environ.get('HOSTNAME')
    if host is not None:
        return host
    rtn = gethostname()
    return rtn.replace('.cs.ucla.edu', '')



def assert_valid_nid(nid, g):
    assert type(nid) is int and (0 <= nid < g.number_of_nodes())


def assert_0_based_nids(g):
    for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
        assert_valid_nid(n, g)
        assert i == n  # 0-based consecutive node ids


def format_str_list(sl):
    assert type(sl) is list
    if not sl:
        return 'None'
    else:
        return ','.join(sl)


class C(object):  # counter
    def __init__(self):
        self.count = 0

    def c(self):  # count and increment itself
        self.count += 1
        return self.count

    def t(self):  # total
        return self.count

    def reset(self):
        self.count = 0


class OurTimer(object):
    def __init__(self):
        self.t = time()
        self.durations_log = OrderedDict()

    def time_and_clear(self, log_str='', only_seconds=False):
        duration = self._get_duration_and_reset()
        if log_str:
            if log_str in self.durations_log:
                raise ValueError(
                    'log_str {} already in log {}'.format(log_str, self.durations_log)
                )
            self.durations_log[log_str] = duration
        if only_seconds:
            rtn = duration
        else:
            rtn = format_seconds(duration)
        # print(log_str, '\t\t', rtn)
        return rtn

    def start_timing(self):
        self.t = time()

    def print_durations_log(self, print_func):
        if not self.durations_log:
            raise RuntimeError(f'self.durations_log is empty')
        s_start = '*' * 59
        print_func(f'Timer log {s_start}')
        rtn = []
        tot_duration = sum([sec for sec in self.durations_log.values()])
        print_func(f'Total duration {format_seconds(tot_duration)}')
        lss = np.max([len(s) for s in self.durations_log.keys()])
        for log_str, duration in self.durations_log.items():
            s = '{0}{1} : {2} ({3:.2%})'.format(
                log_str,
                ' ' * (lss - len(log_str)),
                format_seconds(duration),
                duration / tot_duration,
            )
            rtn.append(s)
            print_func(s)
        print_func(f'Timer log {s_start}')
        self.durations_log = OrderedDict()  # reset
        return rtn

    def _get_duration_and_reset(self):
        now = time()
        duration = now - self.t
        self.t = now
        return duration

    def get_duration(self):
        now = time()
        duration = now - self.t
        return duration

    def reset(self):
        self.t = time()


def format_seconds(seconds):
    """
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    periods = [
        ('year', 60 * 60 * 24 * 365),
        ('month', 60 * 60 * 24 * 30),
        ('day', 60 * 60 * 24),
        ('hour', 60 * 60),
        ('min', 60),
        ('sec', 1),
    ]

    if seconds <= 1:
        return '{:.3f} msecs'.format(seconds * 1000)

    strings = []
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            if period_name == 'sec':
                period_value = seconds
                has_s = 's'
            else:
                period_value, seconds = divmod(seconds, period_seconds)
                has_s = 's' if period_value > 1 else ''
            strings.append('{:.3f} {}{}'.format(period_value, period_name, has_s))

    return ', '.join(strings)


def random_w_replacement(input_list, k=1):
    return [random.choice(input_list) for _ in range(k)]


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.items():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            if n == m:
                assoc[a2idx[a], b2idx[b]] = assoc[b2idx[b], a2idx[a]] = 1.0
            else:
                assoc[a2idx[a], b2idx[b]] = 1
    assoc = sp.csr_matrix(assoc)
    return assoc


def prompt(str, options=None):
    while True:
        t = input(str + ' ')
        if options:
            if t in options:
                return t
        else:
            return t


def prompt_get_cpu():
    from os import cpu_count

    while True:
        num_cpu = prompt('{} cpus available. How many do you want?'.format(cpu_count()))
        num_cpu = parse_as_int(num_cpu)
        if num_cpu and num_cpu <= cpu_count():
            return num_cpu


def parse_as_int(s):
    try:
        rtn = int(s)
        return rtn
    except ValueError:
        return None


computer_name = None


def prompt_get_computer_name():
    global computer_name
    if not computer_name:
        computer_name = prompt('What is the computer name?')
    return computer_name


def node_has_type_attrib(g):
    for n, d in g.nodes(data=True):
        if 'type' in d:  # TODO: needs to be fixed
            return True
    return False


def print_g(label, g, print_func=print):
    print_func(
        f'{label} {g.number_of_nodes()} nodes {g.number_of_edges()} edges ({type(g)})'
    )


class MLP(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(
        self,
        input_dim,
        output_dim,
        activation_type='relu',
        num_hidden_lyr=2,
        hidden_channels=None,
        bn=False,
        batched_num=None,
    ):
        super().__init__()
        self.out_dim = output_dim
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the lengh of hidden_channels"
            )
        dims = hidden_channels + [output_dim]
        if batched_num is not None:
            assert type(batched_num) is int and batched_num > 0
            dims = [x * batched_num for x in dims]
        self.layer_channels = [input_dim] + dims
        self.activation = create_act(activation_type)
        self.layers = nn.ModuleList(
            list(
                map(
                    self.weight_init,
                    [
                        nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
                        for i in range(len(self.layer_channels) - 1)
                    ],
                )
            )
        )
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x, *_, **kwargs):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
        return layer_inputs[-1]


class MLP_multi_objective(nn.Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(
        self,
        input_dim,
        output_dim,
        activation_type='relu',
        objectives=None,
        num_common_lyr=0,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        self.out_dim = output_dim

        if hidden_channels:
            self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        else:
            self.layer_channels = [input_dim] + [output_dim]
        self.activation = create_act(activation_type)
        self.num_common_lyr = num_common_lyr

        self.layers_common = nn.ModuleList(
            list(
                map(
                    self.weight_init,
                    [
                        nn.Linear(input_dim, input_dim) for i in range(num_common_lyr)
                    ],  # same dim
                )
            )
        )
        self.MLP_heads = nn.ModuleDict()
        self.objectives = objectives
        for obj in self.objectives:
            self.MLP_heads[obj] = nn.ModuleList(
                list(
                    map(
                        self.weight_init,
                        [
                            nn.Linear(
                                self.layer_channels[i], self.layer_channels[i + 1]
                            )
                            for i in range(0, len(self.layer_channels) - 1)
                        ],
                    )
                )
            )
        self.bn = bn
        if self.bn:
            self.bn = torch.nn.BatchNorm1d(output_dim)

    def weight_init(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers_common:
            input = layer_inputs[-1]
            # always apply activation on common layers
            layer_inputs.append(self.activation(layer(input)))
        out_common_layers = layer_inputs[-1]
        out_MLP = {}
        for obj in self.objectives:
            out_MLP[obj] = out_common_layers
            for layer_ind, layer in enumerate(self.MLP_heads[obj]):
                if layer_ind + self.num_common_lyr == len(self.layer_channels) - 1:
                    out_MLP[obj] = layer(out_MLP[obj])
                    if self.bn:
                        out_MLP[obj] = self.bn(out_MLP[obj])
                else:
                    out_MLP[obj] = self.activation(layer(out_MLP[obj]))
        # model.store_layer_output(self, layer_inputs[-1])
        return out_MLP


def create_act(act, num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return nn.ReLU()
    elif act == 'prelu':
        return nn.PReLU(num_parameters)
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'identity' or act == 'None':

        class Identity(nn.Module):
            def forward(self, x):
                return x

        return Identity()
    elif act == 'elu' or act == 'elu+1':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))


def print_stats(li, name, saver=None):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    if len(li) == 0:
        func(f'empty li {name}')
        return
    stats = OrderedDict()
    stats['#'] = len(li)
    stats['Avg'] = np.mean(li)
    stats['Std'] = np.std(li)
    stats['Min'] = np.min(li)
    stats['Max'] = np.max(li)
    stats['Median'] = np.median(li)
    stats['Sum'] = np.sum(li)

    try:
        stats['Mode'] = scipy.stats.mode(li, axis=None, keepdims=True)[0][0]
    except Exception as e:
        saver.log_info(f'Try pip install scipy==1.11.4')
        raise e
    func(name)
    for k, v in stats.items():
        func(f'\t{k}:\t{v:.4f}')


def plot_dist(data, label, save_dir, saver=None, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label, kde=False, norm_hist=False)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()


def _analyze_dist(saver, label, data):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join(['{:10.4f}'.format(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:10.4f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\t#     \t {len(data)}')
    func(f'\tMin   \t {np.min(data)}')
    func(f'\tMax   \t {np.max(data)}')
    func(f'\tMean  \t {np.mean(data)}')
    func(f'\tStd   \t {np.std(data)}')
    func(f'\tMedian\t {np.median(data)}')
    func(f'\tMode  \t {scipy.stats.mode(data, axis=None, keepdims=True)[0][0]}')


def get_model_info_as_str(FLAGS):
    rtn = []
    d = vars(FLAGS)
    for k in d.keys():
        v = str(d[k])
        # if type(d[k]) is list:
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        # else:
        #     vsplit = v.split(',')
        #     assert len(vsplit) >= 1
        #     for i, vs in enumerate(vsplit):
        #         if i == 0:
        #             ks = k
        #         else:
        #             ks = ''
        #         if i != len(vsplit) - 1:
        #             vs = vs + ','
        #         s = '{0:26} : {1}'.format(ks, vs)
        #         rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def extract_config_code():
    with open(join(get_src_path(), 'config.py')) as f:
        return f.read()


def plot_scatter_line(data_dict, label, save_dir):
    fn = f'scatter_{label}_iterations.png'
    ss = ['rs-', 'b^-', 'g^-', 'c^-', 'm^-', 'ko-', 'yo-']
    cs = [s[0] for s in ss]
    plt.figure()
    i = 0

    # min_size = min([len(x['incumbent_data']) for x in data_dict.values()])
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li, y_li = [], []

        # min_len = float('inf')
        # for x in data_dict_elt['incumbent_data']:
        #     if x[1] < min_len:
        #         min_len = x[1]

        for x in data_dict_elt['incumbent_data']:
            # if x[1] > FLAGS.recursion_threshold:
            #     break
            x_li.append(x[1])
            y_li.append(x[0])
        plt.scatter(
            np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)]
        )
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    plt.figure()
    fn = f'scatter_{label}_time.png'
    i = 0
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li = [x[2] for x in data_dict_elt['incumbent_data']]
        y_li = [x[0] for x in data_dict_elt['incumbent_data']]
        plt.scatter(
            np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)]
        )
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    # plt.close()


POINTS_MARKERS = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
POINTS_COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "pink",
    "black",
    "orange",
    "purple",
    "beige",
    "brown",
    "gray",
    "cyan",
    "magenta",
]


def plot_points(points_dict, label, save_dir):
    i = 0
    for pname, points in points_dict.items():
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        plt.plot(
            xs,
            ys,
            POINTS_MARKERS[i % len(POINTS_MARKERS)],
            color=POINTS_COLORS[i % len(POINTS_COLORS)],
            label=f'{pname}_{label}',
        )
        i += 1
    plt.legend(loc='best')
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def multi_plot_dimension(target_list):
    num_figure = len(target_list)
    if num_figure == 1:
        y_dim = 1
        x_dim = 1
    elif num_figure == 2:
        y_dim = 1
        x_dim = 2
    elif num_figure == 3:
        y_dim = 1
        x_dim = 3
    elif num_figure == 4:
        y_dim = 2
        x_dim = 2
    elif num_figure == 5 or num_figure == 6:
        y_dim = 2
        x_dim = 3
    else:
        assert False
    # points_dict = {}
    return num_figure, x_dim, y_dim


def plot_scatter_with_subplot(
    points_dict_multi_target, label, save_dir, target_list, connected=True
):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list)
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        points_dict[f'p'] = points_dict_multi_target[target]['pred']
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax = plt.subplot(y_dim, x_dim, idx + 1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for (
            pname,
            points_,
        ) in points_dict.items():  # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(
                    np.array(x_li),
                    np.array(y_li),
                    label=f'{gname}-{pname}',
                    color=cs[i % len(cs)],
                    marker=m[pname],
                )
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                # plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                #         color=POINTS_COLORS[i % len(POINTS_COLORS)],
                #         label=f'{pname}')
                i += 1
        plt.legend(loc='best')
        plt.title(f'{target}')
        plt.grid(True)
        plt.axis('on')
        points_dict = {}

    plt.suptitle(f'{label}')
    fn = f'points_{label}.png'
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()


def plot_scatter_with_subplot_trend(
    points_dict_multi_target, label, save_dir, target_list, connected=True
):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list)
    num_figure, x_dim, y_dim = 1, 1, 1
    points_dict = {}
    ss = ['r-', 'b-', 'g-', 'c-', 'm-', 'k-', 'y-', 'w-']
    cs = [s[0] for s in ss]
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(18)
    fig.set_figwidth(24)
    m = {'p': 'o', 't': 'x'}
    for idx, target in enumerate(target_list):
        if 'perf' not in target:
            continue
        # points_dict[f'p'] = points_dict_multi_target[target]['pred']
        points_dict[f't'] = points_dict_multi_target[target]['true']
        ax = plt.subplot(y_dim, x_dim, idx + 1)
        ax.set_facecolor('xkcd:gray')
        i = 0
        for (
            pname,
            points_,
        ) in points_dict.items():  # dict (true/pred) of dict (name: points)
            for gname, points in points_.items():
                if len(points) <= 1:
                    continue
                fig = plt.figure()
                x_li = [str(int(point[0])) for point in sorted(points)]
                y_li = [round(float(point[1]), 2) for point in sorted(points)]
                plt.scatter(
                    np.array(x_li),
                    np.array(y_li),
                    label=f'{gname}-{pname}',
                    color=cs[i % len(cs)],
                    marker=m[pname],
                )
                if connected:
                    plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
                # plt.plot(xs, ys, POINTS_MARKERS[i % len(POINTS_MARKERS)],
                #         color=POINTS_COLORS[i % len(POINTS_COLORS)],
                #         label=f'{pname}')
                i += 1
                # plt.legend(loc='best')
                plt.title(f'{target}')
                plt.grid(True)
                plt.axis('on')
                # points_dict = {}

                plt.suptitle(f'{label}')
                fn = f'points_{gname}.png'
                plt.savefig(join(save_dir, fn), bbox_inches='tight')
                plt.close()


def plot_points_with_subplot(points_dict_multi_target, label, save_dir, target_list):
    i = 0
    num_figure, x_dim, y_dim = multi_plot_dimension(target_list)
    points_dict = {}
    fig = plt.figure()
    # print(fig.get_figheight(), fig.get_figwidth())
    fig.set_figheight(7.2)
    fig.set_figwidth(10.8)
    for idx, target in enumerate(target_list):
        # points_dict[f'pred_points'] = points_dict_multi_target[target]['pred']
        # points_dict[f'true_points'] = points_dict_multi_target[target]['true']
        pdict = points_dict_multi_target[target]
        plt.subplot(y_dim, x_dim, idx + 1)
        i = 0
        # xs = pdict['true']
        # ys = pdict['pred']
        xs = [(t, t) for t in pdict['true']]
        ys = [(t, p) for t, p in zip(pdict['true'], pdict['pred'])]
        # for pname, points in points_dict_multi_target[target]['pred']:
        #     if pname == 'pred_points':
        #         xs = [point[0] for point in points]
        #         ys = [point[1] for point in points]
        plt.plot(
            xs,
            ys,
            POINTS_MARKERS[i % len(POINTS_MARKERS)],
            color=POINTS_COLORS[i % len(POINTS_COLORS)],
        )
        std = pdict.get('pred_std')
        if std is not None:
            plt.errorbar(xs, ys, std, linestyle='None')
            # i += 1

        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, color='green', lw=3, scalex=False, scaley=False)

        # plt.legend(loc='best')
        plt.title(f'{target}')
        # points_dict = {}
    plt.suptitle(f'{label}')
    fn = f'{label}_plot.png'
    fp = join(save_dir, fn)
    create_dir_if_not_exists(dirname(fp))
    plt.savefig(fp, bbox_inches='tight')
    plt.close()



def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


class TopKModelMaintainer(object):
    def __init__(self, K):
        assert K >= 1
        self.K = K
        self.li = []

    def add_model(self, model, iteration, val_loss, save_dir):

        self.li.append([val_loss, iteration, None])

        self.li.sort()

        i = self._need_save(iteration)
        if i is not None:
            from saver import saver

            saver.log_info(
                f'TopKModelMaintainer: In top {self.K}: '
                f'Saved model at iter {iteration} with val loss {val_loss:.4f}'
            )

            save_file = Path(f'{save_dir}/model_{iteration}_{val_loss:.4f}.pth')
            torch.save(model.state_dict(), save_file)
            self.li[i][2] = save_file

            if len(self.li) > self.K:
                to_remove = self.li[self.K]
                remove(to_remove[2])
                saver.log_info(
                    f'TopKModelMaintainer: ' f'Removed {to_remove[2]} from save folder'
                )

    def _need_save(self, iteration):
        if len(self.li) > self.K:
            for i in range(self.K):
                if self.li[i][1] == iteration:
                    # Current model val_loss is in top K.
                    return i
            return None
        else:
            for i in range(len(self.li)):
                if self.li[i][1] == iteration:
                    return i
            assert False


def get_gname(data):
    return data.gname.split('_')[0]


def check_prepend_root_folder(path):
    if path and path[0] != '/':
        return join(get_root_path(), 'logs', path)
    else:
        return path


def get_best_gpu(option, no_touch_gpus, print_func=print):
    gpus = get_gpu_info()

    for free_mem, gpu in gpus:
        nts = ''
        if gpu in no_touch_gpus:
            nts = ' (no touch)'
        print_func(f'gpu {gpu} free {format_file_size(free_mem)}{nts}')

    gpus_new = []
    for free_mem, gpu in gpus:
        if gpu not in no_touch_gpus:
            gpus_new.append((free_mem, gpu))
    gpus = gpus_new

    if option == 'user_input':
        done = False
        while not done:
            choose = input("Enter Integer Number!\n")
            print_func(f"You entered {choose}")
            try:
                choose = int(choose)
                if 0 <= choose <= 7:
                    done = True
                else:
                    print_func(f"Not a valid chosen gpu id")
            except ValueError:
                print_func(f"Not an integer")

    elif option == 'auto':
        cur_max = -np.inf
        choose_from = []
        for free_mem, gpu in gpus:
            if free_mem < cur_max:
                break
            else:
                cur_max = max(cur_max, free_mem)
                choose_from.append(gpu)
        print_func(f'Choose gpu from {choose_from}')
        choose = random.Random(datetime.datetime.now()).choice(choose_from)
        # choose = gpus[0][1]
        print_func(f'Choose gpu {choose}')
    else:
        raise NotImplementedError()
    return choose


def format_file_size(size, decimals=2, binary_system=True):
    if binary_system:
        units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB']
        largest_unit = 'YiB'
        step = 1024
    else:
        units = ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']
        largest_unit = 'YB'
        step = 1000

    for unit in units:
        if size < step:
            return ('%.' + str(decimals) + 'f %s') % (size, unit)
        size /= step

    return ('%.' + str(decimals) + 'f %s') % (size, largest_unit)


def get_gpu_info():
    global initial_gpu_free_mem_info

    import nvidia_smi

    nvidia_smi.nvmlInit()
    total_num_gpus = nvidia_smi.nvmlDeviceGetCount()

    import torch

    if not torch.cuda.is_available():
        return -1

    if total_num_gpus > torch.cuda.device_count():
        total_num_gpus = torch.cuda.device_count()
    gpus = []
    for gpu in range(total_num_gpus):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
        # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

        # print("Total memory:", info.total)
        # print("Free memory:", info.free)
        # print("Used memory:", info.used)

        nvidia_smi.nvmlShutdown()
        gpus.append((info.free, gpu))

    assert len(gpus) == total_num_gpus
    gpus.sort(reverse=True)

    return gpus


def print_gpu_free_mem_info(gpus, saver):
    saver.log_info(f'Save GPU free memory information', silent=True)
    for free_mem, gpu in gpus:
        saver.log_info(f'gpu {gpu} free {format_file_size(free_mem)}', silent=True)


def create_pred_dict(target_list, extra_entries=None):
    rtn = OrderedDict()
    for target_name in target_list:
        rtn[target_name] = {'true': [], 'pred': []}
        if extra_entries is not None:
            for e in extra_entries:
                rtn[target_name][e] = []
    return rtn


def create_edge_index(g):  # TODO: check directions
    # g = nx.convert_node_labels_to_integers(g, ordering='sorted')
    # critical tn ensure g.edges ordering agrees with edge_attr generation code

    # edge_index = torch.LongTensor(list(g.edges)).t().contiguous()
    from config import FLAGS

    edge_index = torch.LongTensor(list(g.edges)).t().contiguous().to(FLAGS.device)

    if edge_index.shape[0] == 3:  # to handle 3d-rendering
        edge_index = edge_index[0:2]
        # edge_index = torch.concat([edge_index[:, 0:8434], edge_index[:, 8435:8454], edge_index[:, 8455:]], 1)
        # assert edge_index.shape[1] == 9826
        # assert len(g) == 2685

    return edge_index


NON_OPT_PRAGMAS = ['LOOP_TRIPCOUNT', 'INTERFACE', 'INTERFACE', 'KERNEL']
WITH_VAR_PRAGMAS = ['DEPENDENCE', 'RESOURCE', 'STREAM', 'ARRAY_PARTITION']
# TARGET = ['perf', 'util-DSP', 'util-BRAM', 'util-LUT', 'util-FF']


def check_any_in_str(li, s):
    for li_item in li:
        if li_item in s:
            return True
    return False


def coo_to_sparse(coo, device):
    coo = sp.coo_matrix(coo.todense())
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = coo.shape

    rtn = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return rtn


def estimate_model_size(model, label, saver):
    def _to_MB(s):
        return s / 1024**2

    param_size = 0
    num_params = 0
    sizes = []
    for name, param in model.named_parameters():
        try:
            param_size_this = param.nelement() * param.element_size()
        except Exception as e:
            saver.log_info(f'{name} got an error {e}!')
            param_size_this = 0
            # raise e
        sizes.append((param_size_this, name))
        param_size += param_size_this
        num_params += 1
    sizes.sort(reverse=True)
    saver.log_info('Printing top 10 large model parameters:')
    for size, name in sizes[0:10]:
        saver.log_info(f'\t{name} size: {_to_MB(size):.3f}MB')

    buffer_size = 0
    num_buffers = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        num_buffers += 1

    size_all_mb = _to_MB(param_size + buffer_size)
    saver.log_info(
        f'The {label} has {num_params} parameters, {num_buffers} buffers of (estimated) size: {size_all_mb:.3f}MB'
    )


def create_loss_dict_get_target_list(FLAGS, task):
    loss_dict = {}
    # i = 0
    _target_list = FLAGS.target
    if not isinstance(FLAGS.target, list):
        _target_list = [FLAGS.target]
    if task == 'regression':
        target_list = [
            'actual_perf' if FLAGS.encode_log and t == 'perf' else t
            for t in _target_list
        ]
    else:
        target_list = [_target_list[0]]  # only keep the perf target
        assert target_list == ['perf']
    # target_list = ['perf', 'util-LUT', 'util-FF', 'util-DSP']
    if hasattr(FLAGS, 'node_token_alignment_loss') and FLAGS.node_token_alignment_loss:
        loss_dict['loss_pc_links'] = 0.0
    if hasattr(FLAGS, 'gs_alignment_loss') and FLAGS.gs_alignment_loss:
        loss_dict['loss_gs_alignment'] = 0.0
    for t in target_list:
        loss_dict[t] = 0.0
    return loss_dict, target_list


def update_loss_dict(loss_dict, loss_dict_, target_list, FLAGS, data):
    for t in target_list:
        if t in loss_dict_:
            loss_dict[t] += loss_dict_[t].item()  # * get_num_graphs(data)
    for k, v in loss_dict_.items():
        if k not in target_list:
            if k not in loss_dict:
                loss_dict[k] = (
                    0  # lenient code: handle any loss components that are NOT defined previously but somehow added by model forward pass eg pairwise class loss components
                )
            loss_dict[k] += loss_dict_[k].item()  # * get_num_graphs(data)
    if hasattr(FLAGS, 'node_token_alignment_loss') and FLAGS.node_token_alignment_loss:
        loss_dict['loss_pc_links'] += loss_dict_[
            'loss_pc_links'
        ].item()  # * get_num_graphs(data)
    if hasattr(FLAGS, 'gs_alignment_loss') and FLAGS.gs_alignment_loss:
        loss_dict['loss_gs_alignment'] += loss_dict_[
            'loss_gs_alignment'
        ].item()  # * get_num_graphs(data)
    if 'guide_loss' in loss_dict_:
        if 'guide_loss' not in loss_dict:
            loss_dict['guide_loss'] = 0
        loss_dict['guide_loss'] += loss_dict_[
            'guide_loss'
        ].item()  # * get_num_graphs(data)
    return loss_dict


def get_num_graphs(data):
    if hasattr(data, 'num_graphs_'):
        return data.num_graphs_
    else:
        return data.num_graphs


def deduce_load_model_path(task, FLAGS):
    if task == 'regression':
        load_model = FLAGS.load_model
    else:
        assert task == 'class'
        if hasattr(FLAGS, 'load_model_class') and FLAGS.load_model_class:
            load_model = FLAGS.load_model_class
            assert FLAGS.task == 'regression'
        else:
            load_model = FLAGS.load_model
            assert FLAGS.task == 'class'
    return load_model


def get_flag_handling_HARP(FLAGS, key_name):
    rtn = getattr(FLAGS, key_name, None)
    if rtn is not None:
        return rtn
    if key_name == 'pragma_scope':
        return 'block'
    else:
        raise NotImplementedError()


def HARP_model_loaded(FLAGS):
    return (
        hasattr(FLAGS, 'load_model_class')
        and FLAGS.load_model_class is not None
        and 'atefeh' in FLAGS.load_model_class
    )


def report_save_dir(save_dir):
    folder_path = Path(save_dir)
    files = list(folder_path.rglob('*'))
    total_size = sum(f.stat().st_size for f in files if f.is_file())

    rtn = f"Total number of files: {len([f for f in files if f.is_file()])}; Total folder size: {format_file_size(total_size)}"
    return rtn


def get_file_size_str(file):
    f = Path(file)
    if f.is_file():
        total_size = f.stat().st_size
        rtn = f"of size {format_file_size(total_size)}"
    else:
        rtn = 'not a file'

    return rtn


def print_list_of_numbers_with_perc(num_li, name, print_func):
    assert type(num_li) is list and len(num_li) > 0
    s = sum(num_li)
    perc_li = []
    for num in num_li:
        perc_li.append(f'{num/s:.4%}')
    print_func(f'{name}: {num_li} (sum={s}; %li={perc_li})')


def format_loss_dict(loss_dict):
    return {k: f'{v:.4f}' for k, v in loss_dict.items()}

