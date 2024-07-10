# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:17:11 2024

@author: heath
"""

import pandas as pd
import Loop_detector_data_functions_4 as LDF


# Retireve list of all files (corresponding to each sensor)

config = {
   'directory': 'C:/Users/heath/Desktop/Flow_data/',
   'time_int': '30S', 
   'sensor_dict': {'Gasselstiege':'Gassels',
                  'Kanalpromenade Abschnitt 6':'KnlPro6',
                  'Kanalpromenade Dingstiege':'KnlProD',
                  'Promenade':'Promenade'},
   'loop_info': 'C:/Users/heath/Desktop/Flow_data/Full Data/sites.csv',
   'CDF_brandenburg/observations': False,
   'CDF_all_sites': True,
   'scatter_speed_flow': True,
   'scatter_density_flow': True,
   'site_heatmaps': True,
   'combined_heatmap': True,
   'line_of_best_fit': True
   }


def load_data(config):
    sensor_data = {}
    for sensor in config['sensor_dict'].keys():
        print(sensor)
        df1 = pd.read_csv(f'Full Data/{sensor}.csv', sep=';')
        df1['timestamp'] = pd.to_datetime(df1['timestamp'], format='mixed')
        df2 = pd.read_csv(f'June Data/{sensor}.csv', sep=';')
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='mixed')
        df = pd.concat([df1, df2], ignore_index=True)
        df, l2 = LDF.remove_slow(df, 0)
        sensor_data[config['sensor_dict'][sensor]] = df    

    return sensor_data


def run_analyses(data, config):
    aggregation = config['time_int']
    loop_info = pd.read_csv(config['loop_info'], sep=';')
    
    if config['scatter_speed_flow']:
        obs_x, obs_y, all_obs_x, all_obs_y = [],[],[],[]
    if config['scatter_density_flow']:
        obs_x_den, obs_y_den, all_obs_x_den, all_obs_y_den = [], [], [], []
    if config['combined_heatmap'] or config['line_of_best_fit']:    
        list_sublane_densities = {'in':[],'out':[]}
    
    for key, value in config['sensor_dict'].items():
        print(value)
        width = LDF.get_width(loop_info, key)
        orders = LDF.get_order(loop_info, key)  
        df = data[value]
        try:
            flows = pd.read_csv(config['directory']+value+aggregation+'.csv', sep=',')
        except:
            flows = LDF.get_flow(value, aggregation, config['directory'], df, width, orders)
        
        for direction in ['in','out']:
            order = orders[direction]     
            if config['scatter_speed_flow']:
                x_fs, y_fs, all_x_fs, all_y_fs = LDF.collect_observations_lane(flows, order, value, direction, 'ave_speed')
                obs_x.append(x_fs)
                obs_y.append(y_fs)
                all_obs_x.append(all_x_fs)
                all_obs_y.append(all_y_fs)
            if config['scatter_density_flow']:
                x_fd, y_fd, all_x_fs_den, all_y_fs_den = LDF.collect_observations_lane(flows, order, value, direction,'density/m')
                obs_x_den.append(x_fd)
                obs_y_den.append(y_fd)
                all_obs_x_den.append(all_x_fs_den)
                all_obs_y_den.append(all_y_fs_den)
            if config['site_heatmaps']:
                LDF.plot_heatmap_together(flows, direction, value+aggregation, order, width, aggregation)
            if config['combined_heatmap'] or config['line_of_best_fit']:
                list_sublane_densities[direction].append(LDF.get_dataframe_sublane_1m_densities(flows, direction, order, width))
    
    if config['CDF_brandenburg/observations']:
        LDF.plot_speed_CDF(data)   
    if config['CDF_all_sites']:  
        LDF.plot_speed_CDF_multiple(data, config['sensor_dict'].values())   
    if config['scatter_speed_flow']:
        LDF.plot_scatter_speed_flow_lane(obs_x, obs_y)
        LDF.plot_scatter_speed_flow(all_obs_x, all_obs_y)
    if config['scatter_density_flow']:
        LDF.plot_scatter_density_flow_lane(obs_x_den, obs_y_den)
        LDF.plot_scatter_density_flow(all_obs_x_den, all_obs_y_den)
    if config['combined_heatmap'] or config['line_of_best_fit']:
        means, stds, N = LDF.combine_sublane1m(list_sublane_densities, 500)
        if config['combined_heatmap']:
            LDF.heatmap_sublanes_densities(means, 70, 'mean', N)
            LDF.heatmap_sublanes_densities(stds, 20, "std. dev.", N)
        if config['line_of_best_fit']:
            LDF.lines_of_best_fit(means, [10,20,30,40], r'density $\overline{k_l}$ [bicycles/km/m]')
    return


data = load_data(config)
run_analyses(data, config)







        




