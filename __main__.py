# -*- coding: utf-8 -*-
"""
file: run loop detector analysis

author: Heather Kaths
"""

import argparse
import pandas as pd
import LoopDetectorData as LDF


# Parameters
def create_args():
    parser = argparse.ArgumentParser(prog = 'loop detector data',
                                 description = 'investigating bicycle traffic flow using loop detector data')
    parser.add_argument('--directory', default="./data",
                        help='Path to the directory that contains the raw data from inductive loop detectors')
    parser.add_argument('--time_int', default='30S',
                        help='Duration of time intervals used to analyse flow data')    
    parser.add_argument('--sensor_dict', default={'Gasselstiege':'Gassels',
                                               'Kanalpromenade Abschnitt 6':'KnlPro6',
                                               'Kanalpromenade Dingstiege':'KnlProD',
                                               'Promenade':'Promenade'},
                        help='Long and short labels for the sites.')
    parser.add_argument('--loop_info', default="./data/sites.csv",
                        help='information about the width and order of the inductive loops for each direction')  
    parser.add_argument('--CDF_brandenburg/observations', default=False)      
    parser.add_argument('--CDF_all_sites', default=False)  
    parser.add_argument('--scatter_speed_flow', default=True)  
    parser.add_argument('--scatter_density_flow', default=True)  
    parser.add_argument('--site_heatmaps', default=False)  
    parser.add_argument('--combined_heatmap', default=True)  
    parser.add_argument('--line_of_best_fit', default=True)  
    parser.add_argument('--min_speed', default=6,
                        help='all speed recordings less than or equal to this value will be removed from the dataset'),
    parser.add_argument('--max_counterflow', default=0,
                        help='all time intervals with counterflow greater than this value will be removed from the dataset')    
    return vars(parser.parse_args())


def load_data(config):
    sensor_data = {}
    directory = config['directory']
    for sensor in config['sensor_dict'].keys():
        print(sensor)
        df1 = pd.read_csv(f'{directory}/May_Data/{sensor}.csv', sep=';')
        df1['timestamp'] = pd.to_datetime(df1['timestamp'], format='mixed')
        df2 = pd.read_csv(f'{directory}/June_Data/{sensor}.csv', sep=';')
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], format='mixed')
        df = pd.concat([df1, df2], ignore_index=True)
        df, l2 = LDF.remove_slow(df, config['min_speed'])
        sensor_data[config['sensor_dict'][sensor]] = df 
    return sensor_data


def run_analyses(data, config):
    aggregation = config['time_int']
    loop_info = pd.read_csv(config['loop_info'], sep=';')
    directory = config['directory']
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
            flows = pd.read_csv(f'{directory}/flow_data/{value}{aggregation}.csv', sep=',')
            print(f'{value} loaded')
        except:
            print(df)
            flows = LDF.get_flow(value, aggregation, config['directory'], df, width, orders)
            flows.to_csv(f'{directory}/flow_data/{value}{aggregation}.csv', index=False)
        
        for direction in ['in','out']:
            order = orders[direction]     
            if config['scatter_speed_flow']:
                x_fs, y_fs, all_x_fs, all_y_fs = LDF.collect_observations_lane(flows, order, value, direction, 'ave_speed', config['max_counterflow'])
                obs_x.append(x_fs)
                obs_y.append(y_fs)
                all_obs_x.append(all_x_fs)
                all_obs_y.append(all_y_fs)
            if config['scatter_density_flow']:
                x_fd, y_fd, all_x_fs_den, all_y_fs_den = LDF.collect_observations_lane(flows, order, value, direction,'density/m', config['max_counterflow'])
                obs_x_den.append(x_fd)
                obs_y_den.append(y_fd)
                all_obs_x_den.append(all_x_fs_den)
                all_obs_y_den.append(all_y_fs_den)
            if config['site_heatmaps']:
                LDF.plot_heatmap_together(flows, direction, value+aggregation, order, width, aggregation, config['max_counterflow'])
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
        means, stds, N = LDF.combine_sublane1m(list_sublane_densities, config['max_counterflow'])
        if config['combined_heatmap']:
            LDF.heatmap_sublanes_densities(means, 70, 'mean', N)
            LDF.heatmap_sublanes_densities(stds, 20, "std. dev.", N)
        if config['line_of_best_fit']:
            LDF.lines_of_best_fit(means, [10,20,30,40], r'density $\overline{k_l}$ [bicycles/km/m]')
    return

def main():
    config = create_args()
    data = load_data(config)
    run_analyses(data, config)

if __name__ == '__main__':
    main()





        




