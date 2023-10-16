# -*- coding: utf-8 -*-
# @Time    : 18/05/2023 14:15
# @Function:
# -*- coding: utf-8 -*-
# @Time    : 17/04/2023 16:31
# @Function:
import copy

import matplotlib.pyplot as plt
import re
import os
import glob
import pandas as pd
import ast
import numpy as np
import logging
import json
import seaborn as sns
from lib.visualization import plot_heat_map, plot_line_bond,plot_box


def get_path(dir_path):
    conditions = []
    para_name = ['model.type','data.loader_size','train.graph_lr','train.epochs',
            'model.normal_coef','model.kl_1_coef','model.kl_2_coef','model.GC_delta_coef']
    dirs = [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]
    for dir in dirs:
        #dirs: all subfolder
        condition = {}
        subdir_path = os.path.join(dir_path, dir)
        # method_name = dir.split('_')
        condition["method_name"] = dir #'GIN-ba_2motifs-128-0.001-150-0.0-0.001-0.001-0.0'
        condition["method_path"] = subdir_path # '../result/results/GIN-ba_2motifs-128-0.001-150-0.0-0.001-0.001-0.0'
        #
        sub_dirs = [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))]
        paras_paths = []
        paras_dict= []
        for p_v in sub_dirs:
            paras_path = os.path.join(subdir_path, p_v)
            paras_paths.append(paras_path)
            paras_val = p_v.split('-')
            paras = {}
            num = min(len(para_name),len(paras_val))
            for n,para_val in zip(para_name[:num],paras_val[:num]):
                paras[n]=para_val
            paras_dict.append(paras)
        condition["paras_paths"] = paras_paths
        condition["paras_dict"] = paras_dict
        conditions.append(condition)
    # print(conditions)
    return conditions




def res_extrat(log_content):
    # 去除换行符和空格
    # 将单引号替换为双引号
    text_with_double_quotes = log_content.replace("'", "\"")

    # 将清理后的文本解析为Python字典
    data = json.loads(text_with_double_quotes)
    return data

def get_init_res(results, metrics):
    metrics = [item for item in metrics if item not in ('fpr', 'tpr')]
    edge_res = {key: [] for key in metrics}
    g_res = {key: [] for key in metrics}
    init_res = {'best_edge_res': edge_res, 'best_g_res': g_res}
    for result in results:
        for init_element_key, element_key in zip(init_res, result):
            # init_value = init_res[init_element_key]
            element_value = result[element_key]
            for m in metrics:
                init_res[init_element_key][m].append(element_value[m])
    return init_res

def comp_avg_pm(results, metrics):
    init_res = get_init_res(results, metrics)

    for outer_key in init_res:
        outer_value = init_res[outer_key]
        if isinstance(outer_value, dict):
            # 遍历内层字典的键值对
            for inner_key, inner_value in outer_value.items():
                mean = np.mean(inner_value)
                stddev = np.std(inner_value)
                init_res[outer_key][inner_key] = str(round(mean * 100, 2)) + '$\pm$' + str(round(stddev * 10, 2))
    return init_res


def comp_avg(results, metrics):
    init_res = get_init_res(results, metrics)
    # init_res = init_res[element]
    for outer_key in init_res:
        outer_value = init_res[outer_key]
        if isinstance(outer_value, dict):
            # 遍历内层字典的键值对
            for inner_key, inner_value in outer_value.items():
                mean = np.mean(inner_value)
                stddev = np.std(inner_value)
                init_res[outer_key][inner_key] =  round(mean * 100, 2)
    return init_res




def excel_res(conditions,framework,data_name = 'mutag', element = 'best_edge_res', metrics = ['roc_auc','acc']):
    one_data_df =  pd.DataFrame(columns=[data_name])
    # metrics = ['roc_auc', 'acc', 'f1', 'recall', 'recall_macro', 'recall_weight', 'fpr', 'tpr']
    # metrics = ['roc_auc', 'acc']
    for condition in conditions:
        if condition['method_name'] in framework:
            if condition['method_name'] == 'InsGNN':
                focus_params = {'model.type': ['PNA', 'GIN'], 'train.graph_lr': ['0.01'],
                                    'data.loader_size': ['128', '512'],
                                    'model.normal_coef': ['0.0', '0.0001', '0.001', '0.01', '0.1', '1.0'],
                                    'model.kl_1_coef': ['0.0', '0.0001', '0.001', '0.01', '0.1', '1.0'],
                                    'model.kl_2_coef': ['0.0'],
                                    'model.GC_delta_coef': ['0.0', '0.0001', '0.001', '0.1', '1.0']}
            else:
                focus_params = {'model.type': ['PNA', 'GIN'], 'train.graph_lr': ['0.01', '0.001', '0.0001', '0.003'],
                                'data.loader_size': ['128', '512'],
                                'model.normal_coef': ['0.0'],
                                'model.kl_1_coef': ['0.0'],
                                'model.kl_2_coef': ['0.0'],
                                'model.GC_delta_coef': ['0.0']}

            for para_path, paras_dict in zip(condition['paras_paths'], condition['paras_dict']):
                edge_res = {}
                # whether focus_param
                key_match = all(key in paras_dict and paras_dict[key] in value_list for key, value_list in focus_params.items())
                if key_match:
                    results = []
                    round_dirs = [name for name in os.listdir(para_path) if
                                  os.path.isdir(os.path.join(para_path, name))]
                    for round_name in round_dirs:
                        round_dir = os.path.join(para_path, round_name)
                        if os.path.exists(os.path.join(round_dir, 'best_res.log')):
                            # print(round_dir )
                            with open(os.path.join(round_dir, 'best_res.log'), 'r') as f:
                                log_content = f.read()
                                # remove nan
                                log_content = log_content.replace('nan', '0.5')
                                # m = ['test_acc']
                                result = res_extrat(log_content)
                                results.append(result)
                    # average
                    result_dic = comp_avg_pm(results, metrics)
                    # save parameter as name
                    name = para_path.split('/')[-1]
                    parts =  name.split('-')
                    name = '-'.join(parts[-4:])
                    name = condition['method_name'] + '-' +name
                    # parts =  name.split('-')
                    # name = '-'.join(parts[2:])  # 从第三个元素开始拼接
                    edge_res[name] = result_dic[element][metrics[0]]
                    result_df = pd.DataFrame.from_dict(edge_res, orient='index',columns=[data_name])
                    one_data_df  = pd.concat([one_data_df,result_df], axis=0, ignore_index=False)

    # print(my_df)
    # data_name = focus_param['data.data_name'][0]
    data_size = focus_params['data.loader_size'][0]

    one_data_df.to_excel(dir_path + f'/{data_name}-{element}.xlsx')
    # print(my_df)
    return one_data_df


#
def hot_map(conditions, condition_param,axis_param ,data_name, metric='roc_auc',element='best_edge_res'):
    my_df = pd.DataFrame()
    metrics = ['roc_auc', 'acc']

    for condition in conditions:
        if condition['method_name'] == 'InsGNN':
            for para_path, paras_dict in zip(condition['paras_paths'], condition['paras_dict']):
                edge_res = {}
                # whether focus_param
                key_match = all(
                    key in paras_dict and paras_dict[key] in value_list for key, value_list in condition_param.items())

                if key_match:
                    params_values = [paras_dict.get(key) for key in axis_param]
                    results = []
                    round_dirs = [name for name in os.listdir(para_path) if
                                  os.path.isdir(os.path.join(para_path, name))]
                    for round_name in round_dirs:
                        round_dir = os.path.join(para_path, round_name)
                        if os.path.exists(os.path.join(round_dir, 'best_res.log')):
                            with open(os.path.join(round_dir, 'best_res.log'), 'r') as f:
                                log_content = f.read()
                                # remove nan
                                log_content = log_content.replace('nan', '0.5')
                                # m = ['test_acc']
                                result = res_extrat(log_content)
                                results.append(result)
                    # average
                    result_dic = comp_avg(results, metrics)[element]
                    my_df.at[params_values[1],params_values[0]] = result_dic[metric]
                # print("test")

    my_df.sort_index(axis=0, inplace=True, ascending=False)  # 按行排序
    my_df.sort_index(axis=1, inplace=True, ascending=True)
    print(my_df)
    data_size = condition_param['data.loader_size'][0]
    new_axis_param = []
    for para in axis_param:
        param = para.split(".")[-1]
        new_axis_param.append(param)
    fig_path = dir_path + f'/{data_name}-{metric}-{new_axis_param[0]}-{new_axis_param[1]}.pdf'
    plot_heat_map(my_df,fig_path,new_axis_param)
    plt.show()
    plt.clf()

def roc_lines(conditions,framework,focus_parm,pose, condition_param,data_name):
    one_data_df = pd.DataFrame(columns=[data_name])
    line_dict = {'method_name': [], 'fpr': [], 'tpr': []}
    for condition in conditions:
        if condition['method_name'] in framework:
            for para_path, paras_dict in zip(condition['paras_paths'], condition['paras_dict']):
                edge_res = {}
                # whether focus_param
                key_match = all(
                    key in paras_dict and paras_dict[key] in value_list for key, value_list in condition_param.items())
                if key_match:
                    results = []
                    round_dirs = [name for name in os.listdir(para_path) if
                                  os.path.isdir(os.path.join(para_path, name))]
                    for round_name in round_dirs:
                        round_dir = os.path.join(para_path, round_name)
                        if os.path.exists(os.path.join(round_dir, 'best_res.log')):
                            with open(os.path.join(round_dir, 'best_res.log'), 'r') as f:
                                log_content = f.read()
                                # remove nan
                                log_content = log_content.replace('nan', '0.5')
                                # m = ['test_acc']
                                result = res_extrat(log_content)
                                results.append(result)
                        # average
                    name = para_path.split('/')[-1]
                    name = name.split('-')[pose]
                    line_dict = comp_auc(results, line_dict,name)
                    # result_dic = comp_avg_pm(results, metrics)
                    # save parameter as name

                    # parts =  name.split('-')
                    # name = '-'.join(parts[2:])  # 从第三个元素开始拼接
    #                 edge_res[name] = result_dic[element][metrics[0]]
    #                 result_df = pd.DataFrame.from_dict(edge_res, orient='index', columns=[data_name])
    #                 one_data_df = pd.concat([one_data_df, result_df], axis=0, ignore_index=False)
    # one_data_df.to_excel(dir_path + f'/{data_name}-{element}.xlsx')
    line_df = pd.DataFrame(line_dict)
    method_name = list(focus_parm.keys())[0]
    if method_name=='graph_lr':
        legend_name = 'learning rate'
    else:
        legend_name = 'batch size'
    fig_path = dir_path + f'/{data_name}-{method_name}.pdf'
    order = list(focus_parm.values())[0]
    plot_line_bond(line_df, fig_path,order,legend_name)

def comp_auc(results,line_dict,name):
    metrics = ['fpr', 'tpr']
    res = {key: [] for key in metrics}
    # init_res = {'best_edge_res': edge_res, 'best_g_res': g_res}
    for result in results:
        result = result['best_edge_res']
        for m in metrics:
            res[m].append(result[m])

    for f, t in zip(res['fpr'], res['tpr']):
        # sample

        f = sequential_sampling(f, 100)
        t = sequential_sampling(t, 100)
        line_dict['method_name'] += [name] * 100

        line_dict['fpr'] += f
        line_dict['tpr'] += t
        break
    return line_dict

def sequential_sampling(data, num_samples):
    # 计算采样间隔
    interval = len(data) // num_samples

    # 进行顺序采样
    sampled_data = [data[i * interval] for i in range(num_samples)]

    return sampled_data

def find_focus_parm(condition_param):
    # 创建一个空字典来存储结果
    result_dict = {}
    pose = 0
    # 遍历字典的键值对
    for i,(key, value) in   enumerate(condition_param.items()) :
        # 如果值列表的长度大于1，将键值对添加到结果字典中
        if len(value) > 1:
            key = key.split('.')[-1]
            result_dict[key] = value
            pose = i
            break
    return result_dict,pose

def find_map_param(condition_param,static_parm):
    # 存储只有一个值的键的列表
    new_condition = {**condition_param, **static_parm}
    keys_to_remove = ['model.normal_coef','model.kl_1_coef','model.GC_delta_coef']
    axis_param = [key for key in keys_to_remove if key not in static_parm]
    return new_condition,axis_param

def box_map(conditions, maintain_box_dict, flips_box_dict ):
    for condition in conditions:
        if condition['method_name'] == 'InsGNN':

            condition_param = {'model.type': ['GIN'], 'train.graph_lr': ['0.01'],
                               'data.loader_size': ['128', '512'],
                               'model.normal_coef': ['0.0'],
                               'model.kl_1_coef': ['0.001'],
                               'model.kl_2_coef': ['0.0'],
                               'model.GC_delta_coef': ['0.0']}
        else:
            condition_param = {'model.type': ['PNA', 'GIN'], 'train.graph_lr': ['0.01'],
                               'data.loader_size': ['128', '512'],
                               'model.normal_coef': ['0.0'],
                               'model.kl_1_coef': ['0.0'],
                               'model.kl_2_coef': ['0.0'],
                               'model.GC_delta_coef': ['0.0']}

        if condition['method_name'] in framework:
            for para_path, paras_dict in zip(condition['paras_paths'], condition['paras_dict']):

                key_match = all(
                    key in paras_dict and paras_dict[key] in value_list for key, value_list in condition_param.items())
                if key_match:
                    log_files = glob.glob(os.path.join(para_path, "*.log"))
                    for log_file in log_files:
                        file_name = os.path.basename(log_file)
                        r_name = file_name.split('_')[0]
                        print(log_file)
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            result_dict =  eval(log_content)
                            for round_dict in result_dict.values():
                                maintain_box_dict["method_names"].append(condition['method_name'])
                                flips_box_dict["method_names"].append(condition['method_name'])
                                maintain_box_dict["rate"].append(r_name)
                                flips_box_dict["rate"].append(r_name)
                                maintain_box_dict['values'].append(round_dict['maintain'])
                                flips_box_dict['values'].append(round_dict['decision_flips'])

    return maintain_box_dict, flips_box_dict

def sort_df(dict,order):
    df = pd.DataFrame(dict)
    df['method_names'] = pd.Categorical(df['method_names'], categories=order,ordered=True)
    df = df.sort_values(by=['method_names', 'rate'])
    return df





if __name__ == '__main__':
    framework = ['InsGNN']#,'GSAT','GAT','GNNExplainer','PGExplainer']
    results = pd.DataFrame()
    metrics =['roc_auc'] # ['acc','roc_auc']
    element = 'best_edge_res'
    data_names = ['ogbg_molbbbp']
    maintain_box_dict = {'method_names': [], 'rate': [], 'values': []}
    flips_box_dict = {'method_names': [], 'rate': [], 'values': []}
    #['ogbg_molbace', 'ogbg_molbbbp', 'ogbg_molhiv']#['spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9']
    for data_name in  data_names:
        dir_path = '../result/'+ data_name
        conditions = get_path(dir_path)
        if 'ogbg' in data_name:
            element = 'best_g_res'
        ############# if xlsx
        # one_data_df = excel_res(conditions, framework, data_name=data_name, element=element, metrics=metrics)
        # results = pd.concat([results,one_data_df], axis=1, ignore_index=False)
    # results.to_excel('../result/all/' + f'{framework}_{metrics[0]}.xlsx')

        ############# if hotmap
        condition_param = {'model.type': ['PNA', 'GIN'], 'train.graph_lr': ['0.01', '0.003'],
                           'data.loader_size': ['128', '512'],
                           'model.normal_coef': ['0.0', '0.0001', '0.001', '0.01', '0.1', '1.0'],
                           'model.kl_1_coef': ['0.0', '0.0001', '0.001', '0.01', '0.1', '1.0'],
                           'model.kl_2_coef': ['0.0'],
                           'model.GC_delta_coef': ['0.0', '0.0001', '0.001', '0.01', '0.1', '1.0']}
        static_parm = {'model.kl_1_coef':['0.0']}

        # axis_param = ['model.normal_coef','model.kl_1_coef','model.GC_delta_coef']
        new_condition,axis_param = find_map_param(condition_param,static_parm)
        hot_map(conditions, new_condition, axis_param, data_name, metric=metrics[0], element=element)

        static_parm = {'model.normal_coef': ['0.01']}
        # condition_param = {'model.type': ['PNA','GIN'], 'train.graph_lr': ['0.01', '0.001', '0.0001', '0.003'],
        #                    'data.loader_size': ['128','512'],
        #                    'model.normal_coef':  ['0.0', '0.0001', '0.001','0.01', '0.1', '1.0'],  # 0
        #                    'model.kl_1_coef': ['0.0', '0.0001', '0.001','0.01', '0.1', '1.0'],  # 1
        #                    'model.kl_2_coef': ['0.0'],
        #                    'model.GC_delta_coef': ['0.0', '0.0001', '0.001','0.01', '0.1', '1.0']}  # 2
        new_condition,axis_param = find_map_param(condition_param,static_parm)

        hot_map(conditions, new_condition,axis_param, data_name, metric=metrics[0],element= element)

        ########## if roc_auc bond
        # condition_param = {'model.type': ['GIN'],'data.loader_size': ['64', '128', '256', '512'],
        #                    'train.graph_lr': ['0.001'], 'train.epochs':['150'],
        #                    'model.normal_coef': ['0.0001'],
        #                    'model.kl_1_coef': [ '0.0001'],
        #                    'model.kl_2_coef': ['0.0'],
        #                    'model.GC_delta_coef': ['1.0']}
        # focus_parm,pose = find_focus_parm(condition_param)
        # roc_lines(conditions,  framework,focus_parm,pose, condition_param,data_name)

        ############ if box
        # maintain_box_dict, flips_box_dict = box_map(conditions,maintain_box_dict, flips_box_dict)
        # order = ['InsGNN', 'GSAT','GAT', 'GNNExplainer', 'PGExplainer']
        # maintain_box_df  = sort_df(maintain_box_dict, order=order)
        #
        # order = ['InsGNN', 'GSAT','GAT', 'GNNExplainer', 'PGExplainer']
        # flips_box_df = sort_df(flips_box_dict, order=order)
        #
        # fig_path = os.path.join('../result/all/','maintain.pdf')
        # plot_box(maintain_box_df,'rate','values','method_names',fig_path,framework )
        # fig_path = os.path.join('../result/all/', 'flips.pdf')
        # plot_box(flips_box_df, 'rate', 'values', 'method_names',fig_path,framework )













