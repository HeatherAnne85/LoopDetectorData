# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:07:07 2024

@author: heath
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d 
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

def map_laneid_to_separate_lane_columns(input_df):
    # This function receives a dataframe with a column called 'laneid' with entries ranging from 1-n
    # and maps this columns to n one-hot columns (n columns)
    for lane in range(1, input_df.lane_id.max() + 1):
        lane_str = f"lane{lane}"
        input_df[lane_str] = np.where(input_df.lane_id == lane, 1, 0)
    return input_df

def map_speed_to_separate_lane_columns(input_df):
    # This function creates a one-hot mapping of the 'speed' column of the received input_df 
    # !!! The received input_df must have been already processed by map_laneid_to_separate_lane_columns()
    for lane in range(1, input_df.lane_id.max() + 1):
        lane_str  = f"lane{lane}"
        speed_str = f"speed{lane}"
        input_df[speed_str] = input_df[lane_str] * input_df['speed']
    return input_df

def remove_slow(df, speed_min):
    #This function removes observations with a speed less than the limit from the database
    l1 = len(df['speed'].values)
    df = df[df['speed'] > speed_min]
    l2 = len(df['speed'].values)
    print(f'removed {l1-l2} of {l1} observations or {(l1-l2)*100/l1}%')
    return df, l2

def ECDF(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum/cusum[-1]

def time_space_headway(input_df):
    #calculate the delta T and the delta speed between the observed cyclist and the one travelling before
    input_df['time_headway'] = input_df['timestamp'].diff() 
    input_df['speed_diff'] = input_df['speed'].diff()
    inx=list(input_df.index)
    for i in range(1,len(inx)-1):
        # space_headway calculated from the speed_b-1 (leader) and the time headway
        input_df.at[inx[i],'space_headway']=input_df.at[inx[i-1], 'speed'] * input_df.at[inx[i], 'time_headway'].total_seconds() / 3.6  
    return input_df

def get_midpoint(interval):
    return int((interval.left + interval.right) / 2)

def get_order(info_df, site):
    orders = {}
    for direction in ['in','out']:
        left = info_df.loc[info_df["Site"] == site, direction+'_left'].values[0]
        center = info_df.loc[info_df["Site"] == site, direction+'_center'].values[0]
        right = info_df.loc[info_df["Site"] == site, direction+'_right'].values[0]
        orders[direction] = [left, center, right]
    return orders

def get_width(info_df, site):
    return info_df.loc[info_df["Site"] == site].values[0][-2]/100

def get_flow(site, aggregation, directory, df, width, orders):
    df_in = calculate_flow_direction(df, aggregation, 'in', width, orders['in'])
    df_out = calculate_flow_direction(df, aggregation, 'out', width, orders['out'])
    flows =  pd.concat([df_in, df_out], axis=1)   
    
    return flows

def get_dataframe_sublane_1m_densities(data, direction, order, loop_width):
    #The function converts densities observed by inductive loops of different widths to 4 one-meter sublanes
    data['D_sublane1_'+direction] = 0
    data['D_sublane2_'+direction] = 0
    data['D_sublane3_'+direction] = 0
    data['D_sublane4_'+direction] = 0
    
    for flow in data.index:  # iterate over the index of the grouped DataFrame
        density_right = data.loc[flow, order[2] + '_density/m_' + direction]
        density_mid = data.loc[flow, order[1] + '_density/m_' + direction]
        density_left = data.loc[flow, order[0] + '_density/m_' + direction]
        
        data.loc[flow, 'D_sublane1_'+direction] = density_right
        data.loc[flow, 'D_sublane2_'+direction] = ((loop_width - 1) * density_right) + ((2 - loop_width) * density_mid)
        data.loc[flow, 'D_sublane3_'+direction] = ((2 * loop_width - 2) * density_mid) + ((3 - (2 * loop_width)) * density_left)
        data.loc[flow, 'D_sublane4_'+direction] = density_left
    return data

def calculate_flow_direction(input_df, aggregation, direction, width, order):
    #This function processes the raw sensor data and returns speed, density and flow information in give time intervals
    input_df = map_speed_to_separate_lane_columns(map_laneid_to_separate_lane_columns(input_df))
    agg_dict = {'15S':140, '30S': 120, '1Min': 60, '5Min': 12, '10Min': 6, '15Min':4, 'H': 1, 'D': 1}
    input_df = time_space_headway(input_df)
    input_df[aggregation+'_interval'] = input_df['timestamp'].dt.floor(aggregation)

    df=pd.DataFrame()
    df[aggregation+'_interval'] = input_df[aggregation+'_interval'].unique()                   
    
    dir_df = input_df[input_df.direction == direction]        
    
    # Group by the aggregation interval
    for time in dir_df[aggregation+'_interval'].unique():
        dir_df2 = dir_df.loc[dir_df[aggregation+'_interval'] == time]
        number = dir_df2[aggregation+'_interval'].apply(lambda x: 1 if x == time else 0).sum()
        speed_inverse = dir_df2['speed'].apply(lambda x: 1/x).sum()
        total_time_headway = dir_df2['time_headway'].apply(lambda x: x.total_seconds()).sum()
        condition = (df[aggregation+'_interval'] == time)

        df.loc[condition, 'number'] = number
        df.loc[condition, 'ave_speed'] = number / speed_inverse
        df.loc[condition, 'time_headway'] =  total_time_headway / number
        df.loc[condition, 'flow'] = number * agg_dict[aggregation]
        df.loc[condition, 'flow/m'] = (number * agg_dict[aggregation])/(width*3)
        df.loc[condition, 'density'] = df.loc[condition, 'flow'] / df.loc[condition, 'ave_speed'] 
        df.loc[condition, 'density/m'] = df.loc[condition, 'flow/m'] / df.loc[condition, 'ave_speed'] 
        
        for lane in range(1, input_df.lane_id.max() + 1):
            lane_str = f"lane{lane}"
            dir_df3 = dir_df2.loc[dir_df2[lane_str] == 1]
            if len(dir_df3['speed'].values) == 0:
                continue
            number = dir_df3[aggregation+'_interval'].apply(lambda x: 1 if x == time else 0).sum()
            speed_inverse = dir_df3['speed'].apply(lambda x: 1/x).sum()
            total_time_headway = dir_df3['time_headway'].apply(lambda x: x.total_seconds()).sum()
            
            df.loc[condition, lane_str+'_number'] = number
            df.loc[condition, lane_str+'_ave_speed'] = number / speed_inverse
            df.loc[condition, lane_str+'_time_headway'] =  total_time_headway / number
            df.loc[condition, lane_str+'_flow'] = number * agg_dict[aggregation]
            df.loc[condition, lane_str+'_flow/m'] = (number * agg_dict[aggregation])/width
            df.loc[condition, lane_str+'_density'] = df.loc[condition, lane_str+'_flow'] / df.loc[condition, lane_str+'_ave_speed'] 
            df.loc[condition, lane_str+'_density/m'] = df.loc[condition, lane_str+'_flow/m'] / df.loc[condition, lane_str+'_ave_speed'] 
            
            df[[lane_str+'_number',lane_str+'_flow',lane_str+'_density',lane_str+'_density/m']] = df[[lane_str+'_number',lane_str+'_flow',lane_str+'_density',lane_str+'_density/m']].fillna(0)
        
    df[['number','flow','density','density/m']] = df[['number','flow','density','density/m']].fillna(0)
    df = df.rename(columns={col: col + '_'+direction for col in df.columns})           
    df = get_dataframe_sublane_1m_densities(df, direction, order, width)
    
    return df

def plot_speed_CDF(speed_dfs):
    #Paper figure 2 - A
    df = speed_dfs['Promenade']
    
    #observed data
    x,y = ECDF(df['speed'])
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)
    
    #distribution from Brandenberg et al. 
    brand_data = np.sort(np.random.normal(18.8, 3.7, 10000))
    x_b, y_b = ECDF(brand_data)
    x_b = np.insert(x_b, 0, x_b[0])
    y_b = np.insert(y_b, 0, 0.)  
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.step(x, y, where='post', color='black', label='Promenade')
    plt.plot(x_b, y_b, color='black', linestyle='--', label='Brandenburg et al.')
    
    legend = plt.legend(loc='lower right')
    legend.get_frame().set_edgecolor('none')
    plt.xlim(5, 35)
    plt.xlabel('speed [km/h]')
    plt.ylabel('cumulative probability')
    plt.tight_layout()
    plt.savefig('figures/Figure_2_A.png', dpi=300, bbox_inches='tight')
    plt.show()
    return

def plot_speed_CDF_multiple(speed_dfs, location_list):
    #Paper figure 2 - B
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    labels = ['Gasselstiege','Kanalpromenade A','Kanalpromenade B','Promenade']
    plt.figure(figsize=(7, 6), dpi=300)
    
    for i, location in enumerate(location_list):
        df = speed_dfs[location]
        x,y = ECDF(df['speed'])
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0.)
        plt.step(x, y, where='post', linestyle = line_styles[i], color='black', label=f'{labels[i]}')
    
    legend = plt.legend(loc='upper left')
    legend.get_frame().set_edgecolor('none')
    legend.get_frame().set_facecolor('none') 

    plt.xlim(5, 35)
    plt.xlabel('speed [km/h]')
    plt.ylabel('cumulative probability')
    plt.tight_layout()
    plt.savefig('figures/Figure_2_B.png', dpi=300, bbox_inches='tight')
    plt.show()
    return

def collect_observations_lane(df_org, right, intersection, direction, label, max_flow): 
    labs = ['inductive loop left', 'inductive loop center', 'inductive loop right']
    obs_x = {}
    obs_y = {}
    
    #if label == 'density/m':
    if direction == 'in':
        opposite = 'out'
    else:
        opposite = 'in'
        
    df_org = df_org[df_org[f'flow_{opposite}'] <= max_flow]
    
    df = df_org
    df = df.dropna(subset=[f'{label}_{direction}', f'flow_{direction}'])
    df = df.sort_values(by=f'flow_{direction}')
    x_s = np.array(df[f'flow_{direction}'].values).reshape(-1, 1)
    y_s = df[f'{label}_{direction}'].values
    
    for lane in [2,1,0]:
        df = df_org
        df = df.dropna(subset=[f'{right[lane]}_{label}_{direction}', f'flow_{direction}'])
        df = df.sort_values(by=f'flow_{direction}')
        
        obs_x[labs[lane]] = np.array(df[f'flow_{direction}'].values).reshape(-1, 1)
        obs_y[labs[lane]] = df[f'{right[lane]}_{label}_{direction}'].values
    
    return obs_x, obs_y, x_s, y_s


def plot_scatter_density_flow(obs_x, obs_y):
    #Paper figure 5 - A
    x = []
    y = []
    for i, intersection in enumerate(obs_x):  
        x.extend(obs_x[i])
        y.extend(obs_y[i])
    
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
           
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    trendline = model.predict(x)
    
    jitter_amount = 15 
    x_jittered = x.flatten() + np.random.normal(0, jitter_amount, len(x))  
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.scatter(x_jittered, y, s=1, color='black') 
    plt.plot(x, trendline, color='white', linewidth=2.7)
    plt.plot(x, trendline, color='black', linewidth=2.0)
        
    plt.xlim(0, 3000)
    plt.ylim(-2, 70)
    plt.xlabel('$q$ [bicycles/h]')
    plt.ylabel('$k$ [bicycles/km/m]')
    plt.text(50, 65, f'n = {len(x.flatten())}', color='black', bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.2'))
    plt.savefig('figures/Figure_5_A.png', dpi=300, bbox_inches='tight')
    plt.show()
    return  
           
def plot_scatter_density_flow_lane(obs_x, obs_y):  
    #Paper figure 5 - B, C, and D
    labs = ['inductive loop right', 'inductive loop center', 'inductive loop left']
    labs_short = ['right', 'center', 'left']
    fig_lab = ['B', 'C', 'D']

    for lane in [0, 1, 2]:
        plt.figure(figsize=(7, 6), dpi=300)
        x = []
        y = []
        for i, intersection in enumerate(obs_x):
            x.extend(obs_x[i][labs[lane]])
            y.extend(obs_y[i][labs[lane]])
        
        x = np.array(x).reshape(-1, 1)
        y = np.array(y)
        
        model = LinearRegression(fit_intercept=True)
        model.fit(x, y)
        trendline = model.predict(x)
        
        jitter_amount = 15 
        x_jittered = x.flatten() + np.random.normal(0, jitter_amount, len(x))  
        
        plt.scatter(x_jittered, y, s=1, color='black')
        plt.plot(x, trendline, color='white', linewidth=2.7)
        plt.plot(x, trendline, color='black', linewidth=2.0)
        
        plt.xlim(0, 3000)
        plt.ylim(-2, 70)
        plt.xlabel('$q$ [bicycles/h]')
        plt.ylabel(r'$k_{' + labs_short[lane] + r'}$ [bicycles/km/m]')
        plt.text(50, 65, f'n = {len(x.flatten())}', color='black', bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.2'))
        plt.savefig(f'figures/Figure_5_{fig_lab[lane]}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return    

def fit_linear_models(x,y,split):
    mask = x < split  
    x1 = x[mask]
    y1 = y[mask]
    x2 = x[~mask]
    y2 = y[~mask]
    
    if len(y1) == 0:
        model = LinearRegression(fit_intercept=False)
        model.fit(x2.reshape(-1, 1), y2)      
        y2_fit = model.predict(x2.reshape(-1, 1))
        return y2_fit      
    
    y1_median = np.mean(y1)
    y1_fit = np.full_like(x1, y1_median)  # Create a horizontal line based on the median value    
    
    if len(y2) == 0:
        return y1_fit
    
    X_prime = x2 - split
    X_prime = X_prime.reshape(-1, 1)
    Y_prime = y2 - y1_median
    model = LinearRegression(fit_intercept=False)
    model.fit(X_prime, Y_prime) 
    y2_fit = model.predict(X_prime)+y1_median
    
    y_piecewise_fit = np.concatenate((y1_fit, y2_fit))

    return y_piecewise_fit
    
def plot_scatter_speed_flow(obs_x, obs_y):
    #Paper figure 4 - A
    x = []
    y = []
    
    for i, intersection in enumerate(obs_x):  
        x.extend(obs_x[i])
        y.extend(obs_y[i])
    
    x = np.array(x).flatten()
    y = np.array(y)
    
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    
    r2s = []
    
    for split in range(360, 2000, 120):   
        y_fit = fit_linear_models(x,y,split)
        r2s.append(r2_score(y, y_fit))

    split = range(360, 2000, 120)[r2s.index(max(r2s))]
    y_fit = fit_linear_models(x,y,split)           
    
    jitter_amount = 15 
    x_jittered = x + np.random.normal(0, jitter_amount, len(x))    
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.scatter(x_jittered, y, s=1, color="black")
    plt.plot(x, y_fit, color='white', linewidth=2.7)
    plt.plot(x, y_fit, color='black', linewidth=2)
        
    plt.xlim(0, 3000)
    plt.ylim(5, 35)
    plt.xlabel('$q$ [bicycles/h]')
    plt.ylabel('$\overline{v}$ [km/h]')
    plt.text(2250, 33, f'n = {len(x)}', color='black', bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.2'))
    plt.savefig('figures/Figure_4_A.png', dpi=300, bbox_inches='tight')
    plt.show()
    return 
     
def plot_scatter_speed_flow_lane(obs_x, obs_y):
    # Paper figure 4 - B, C, D
    labs = ['inductive loop right', 'inductive loop center', 'inductive loop left']
    labs_short = ['right', 'center', 'left']
    fig_lab = ['B', 'C', 'D']

    for lane in [0, 1, 2]:
        r2s = []
            
        x, y = [], []
        for i, intersection in enumerate(obs_x):
            x.extend(obs_x[i][labs[lane]])
            y.extend(obs_y[i][labs[lane]])

        x = np.array(x).flatten()  # Ensure x is 1D
        y = np.array(y)

        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        
        for split in range(0, 2000, 120):   
            y_fit = fit_linear_models(x,y,split)
            r2s.append(r2_score(y, y_fit))
    
        # Plotting best fit
        split = range(0, 2000, 120)[r2s.index(max(r2s))]
        y_fit = fit_linear_models(x,y,split)
        plt.figure(figsize=(7, 6), dpi=300)
        print(f"lane {labs[lane]}\n: split = {split}, r2 = {max(r2s):.5f}")
        
        jitter_amount = 15 
        x_jittered = x + np.random.normal(0, jitter_amount, len(x))
        
        plt.scatter(x_jittered, y, s=1, color='black')
        plt.plot(x, y_fit, color='white', linewidth=2.7)
        plt.plot(x, y_fit, color='black', linewidth=2)

        plt.xlim(0, 3000)
        plt.ylim(5, 35)
        plt.xlabel('$q$ [bicycles/h]')
        plt.ylabel(r'$\overline{v}_{' + labs_short[lane] + r'}$ [km/h]')
        plt.text(2250, 33, f'n = {len(x)}', color='black', bbox=dict(facecolor='white', edgecolor='white', boxstyle='square, pad=0.2'))
        plt.savefig(f'figures/Figure_4_{fig_lab[lane]}.png', dpi=300, bbox_inches='tight')
        plt.show()

    return

    
def plot_heatmap_together(df, direction, site, order, width, aggregation, max_flow):
    #Paper figure 5
    if direction == 'in':
        opposite = 'out'
    else:
        opposite = 'in'
        
    df = df[df[f'flow_{opposite}'] <= max_flow]
    
    N = len(df['flow_'+direction].values)
    flow_direction_bins = range(0,2880,480)
    df['flow_direction_bin'] = pd.cut(df['flow_'+direction], bins=flow_direction_bins, right=False)
    grouped = df.groupby('flow_direction_bin')
    filtered_grouped = grouped.filter(lambda x: len(x) >= 5)
    heatmap_data = filtered_grouped.groupby('flow_direction_bin')[[f'{order[2]}_density/m_{direction}', f'{order[1]}_density/m_{direction}', f'{order[0]}_density/m_{direction}']].mean()
    
    x_labs = [agg.right for agg in heatmap_data.index]
    heatmap_data = heatmap_data.T

    fig, ax = plt.subplots(figsize=(7, 4))
    norm = Normalize(vmin=0, vmax=70)
    cax = ax.matshow(heatmap_data, cmap='jet', aspect='auto', norm=norm)
    fig.colorbar(cax, label=r'$\overline{k_l}$ [bicycles/km/m]')
    
    ax.set_xticks([x + 0.48 for x in range(heatmap_data.shape[1])])
    ax.set_xticklabels(x_labs, rotation=0)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel('$q$ [bicycles/h]')
    ax.invert_yaxis()
    ax.yaxis.set_visible(False)

    ax2 = ax.twinx() 
    ax2.set_ylim(0, width*3) 
    ax2.set_yticks([0, width, 2*width, width*3])  
    ax2.set_yticklabels(['0', str(round(width,1)),str(round(2*width,1)), str(round(width*3,1))])  
    ax2.yaxis.set_ticks_position('left')
    ax2.yaxis.set_label_position('left')
    ax2.set_ylabel('distance from right edge [m]')
    ax2.text(-1, width / 2, 'right\nloop', rotation=90, va='center', fontsize=16)
    ax2.text(-1, 1.5 * width, 'center\nloop', rotation=90, va='center', fontsize=16)
    ax2.text(-1, 2.5 * width, 'left\nloop', rotation=90, va='center', fontsize=16)
        
    ax.annotate('', xy=(0.2, 1.05), xytext=(0, 1.05), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(0.22, 1.05, 'cycling direction', rotation=0, va='center', ha='left', transform=ax.transAxes) 
    ax.text(0.75, 1.05, f'n={N}', rotation=0, va='center', ha='left', transform=ax.transAxes)     
    plt.savefig(f'figures/Figure_6_{site}_{direction}.png', dpi=300, bbox_inches='tight')

    plt.show()
    return   


def combine_sublane1m(list_locs, lim):      
    ins = pd.concat(list_locs['in'])
    ins = ins[ins['flow_out'] <= lim]
    dir_in = ins[['flow_in', 'D_sublane1_in', 'D_sublane2_in', 'D_sublane3_in', 'D_sublane4_in']]
    dir_in.rename(columns={'flow_in': 'flow', 'D_sublane1_in': 'D_sublane1', 'D_sublane2_in': 'D_sublane2', 'D_sublane3_in': 'D_sublane3', 'D_sublane4_in': 'D_sublane4'}, inplace=True)
    
    outs = pd.concat(list_locs['out'])
    outs = outs[outs['flow_in'] <= lim]
    dir_out = outs[['flow_out', 'D_sublane1_out', 'D_sublane2_out', 'D_sublane3_out', 'D_sublane4_out']]
    dir_out.rename(columns={'flow_out': 'flow', 'D_sublane1_out': 'D_sublane1', 'D_sublane2_out': 'D_sublane2', 'D_sublane3_out': 'D_sublane3', 'D_sublane4_out': 'D_sublane4'}, inplace=True)

    df = pd.concat([dir_in, dir_out])
    
    N = len(df['flow'].values)

    flow_direction_bins = range(0,2880,480)
    df['flow_direction_bin'] = pd.cut(df['flow'], bins=flow_direction_bins, right=False)
    
    grouped = df.groupby('flow_direction_bin')
    filtered_grouped = grouped.filter(lambda x: len(x) >= 5)
    mean_df = filtered_grouped.groupby('flow_direction_bin')[['D_sublane1', 
                                                        'D_sublane2', 
                                                        'D_sublane3',
                                                        'D_sublane4']].mean().reset_index()
    std_df = filtered_grouped.groupby('flow_direction_bin')[['D_sublane1', 
                                                        'D_sublane2', 
                                                        'D_sublane3',
                                                        'D_sublane4']].std().reset_index()  
    
    return mean_df, std_df, N

    
def heatmap_sublanes_densities(df, lim, label, N):
    #Figure 7
    heatmap_data = df[['flow_direction_bin','D_sublane1', 'D_sublane2', 'D_sublane3', 'D_sublane4']]
    heatmap_data.set_index('flow_direction_bin', inplace=True)
    y_labs = []
    for agg in df['flow_direction_bin'].unique():
        y_labs.append(agg.right)
        
    heatmap_data = heatmap_data.T 
    fig, ax = plt.subplots(figsize=(7, 4))
    norm = Normalize(vmin=0, vmax=lim)
    cax = ax.matshow(heatmap_data, cmap='jet', aspect='auto', norm=norm)
    if label == 'std. dev.':  
        fig.colorbar(cax, label=label+r' $k_{sl}$ [bicycles/km/m]')
    else:
        fig.colorbar(cax, label='$\overline{k_{sl}}$ [bicycles/km/m]')
    
    # Set the labels
    ax.set_xticks([x + 0.48 for x in range(heatmap_data.shape[1])])
    ax.set_xlabel('$q$ [bicycles/h]')
    ax.invert_yaxis()
    ax.set_yticks([x + 0.48 for x in range(heatmap_data.shape[0])])
    ax.set_yticklabels([1,2,3,4])
    ax.set_xticklabels(y_labs)
    ax.set_ylabel('distance from right edge [m]')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    
    ax.annotate('', xy=(0.2, 1.05), xytext=(0, 1.05), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(0.22, 1.05, 'cycling direction', rotation=0, va='center', ha='left', transform=ax.transAxes)
    ax.text(0.75, 1.05, f'n={N}', rotation=0, va='center', ha='left', transform=ax.transAxes)   
    plt.savefig(f'figures/Figure_7_{label}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return 


def lines_of_best_fit_flow_sublane(df, values): 
    #Figure 8
    flow_dics = {values[0]:[],values[1]:[],values[2]:[],values[3]:[]}
    x_dics = {values[0]:[],values[1]:[],values[2]:[],values[3]:[]}
    y = []
    for flow in df['flow_direction_bin'].unique():
        y.append(get_midpoint(flow))
    y = np.array(y)
    df = df[['D_sublane1', 'D_sublane2', 'D_sublane3', 'D_sublane4']]
    for line_value in values:
        y_values = []
        for lanes in [1,2,3]:
            sublane_x = np.array(df[f'D_sublane{lanes}'].values)
            mask = ~np.isnan(sublane_x)
            sublane_x = sublane_x[mask]
            y_new = y[mask]
            interpolation_function = interp1d(sublane_x, y_new, fill_value='extrapolate')
            print(line_value)
            y_values.append(interpolation_function(line_value))
        flow_dics[line_value].extend(y_values)
        x_dics[line_value].extend([1,2,3])
    
    print(flow_dics)
    print(x_dics)
    marker_styles = ['o', 's', 'D', '^']
    line_styles = ['-', '--', '-.', ':']
    colors = ['midnightblue', 'darkorange', 'teal', "black"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, line_value in enumerate(values):
        x = np.array(x_dics[line_value])
        y = np.array(flow_dics[line_value])
        mask = ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        x = x.reshape(-1, 1)
        ax.scatter(y, x, color=colors[i], s=20, marker=marker_styles[i])
        
        model = LinearRegression(fit_intercept=True)
        model.fit(x, y)
        trendline = model.predict(np.array([1,2,3,4]).reshape(-1, 1))
        ax.plot(trendline, np.array([1,2,3,4]).reshape(-1, 1), label=f'{line_value} bicycles/km/m', linestyle=line_styles[i], color=colors[i])
    
    ax.set_yticks([0.5, 1, 2, 3, 4, 4.5])
    ax.set_yticklabels(['','1', '2', '3', '4', ''])
    ax.set_xlim(-100, 4000)
    ax.set_xlabel('$q$ [bicycles/h]')
    ax.set_ylabel('distance from right edge [m]')
    ax.legend(loc='lower right') 
    ax.annotate('', xy=(0.2, 1.05), xytext=(0, 1.05), xycoords='axes fraction', arrowprops=dict(arrowstyle='->', color='black'))
    ax.text(0.22, 1.05, 'cycling direction', rotation=0, va='center', ha='left', transform=ax.transAxes)
    ax.grid(True)
    plt.savefig('figures/Figure_8.png', dpi=300, bbox_inches='tight')

    plt.show()
    return

def lines_of_best_fit_flow_density(df, densities):
    # Setup data structures to store values for each width
    flow_dics = {density: [] for density in densities}
    x_dics = {density: [] for density in densities}
    y = []
    
    # Calculate midpoints of `flow_direction_bin` and store in `y`
    for flow in df['flow_direction_bin'].unique():
        y.append(get_midpoint(flow))
    y = np.array(y)
    
    # Focus only on density columns for each sublane
    df_sublanes = df[['D_sublane1', 'D_sublane2', 'D_sublane3', 'D_sublane4']]
    
    # Extract flow values for each density level on each sublane
    for density in densities:
        y_values = []
        for lane in [1, 2, 3]:
            # Extract data for the current sublane
            sublane_x = np.array(df_sublanes[f'D_sublane{lane}'].values)
            mask = ~np.isnan(sublane_x)
            sublane_x = sublane_x[mask]
            y_new = y[mask]
            # Interpolate to find flow values for the given density
            interpolation_function = interp1d(sublane_x, y_new, fill_value='extrapolate')
            interpolated_values = interpolation_function(density)
            y_values.append(interpolated_values)
        
        # Store the interpolated flow values and corresponding lane indices
        flow_dics[density].extend(y_values)
        x_dics[density].extend([1, 2, 3]) 
       
    # Plotting   
    line_colors = ['#F08080', '#DC143C', '#4169E1', '#87CEEB']  # Crimson, Light Coral, Royal Blue, Sky Blue
    segment_colors = ['#FFE4E1', '#FFB6C1', '#B0C4DE', '#B0E0E6']  # Light Crimson, Very Light Coral, Light Steel Blue, Powder Blue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Store trendlines for each sublane
    trendlines = []
    
    # Loop through each sublane to plot scatter points and trendlines
    for i in range(3):  # Assuming three sublanes (1, 2, 3)
        x = []
        y = list(flow_dics.keys())  # Density levels as y-values
        
        # Gather flow values for the current sublane across densities
        for density in densities:
            x.append(flow_dics[density][i])
        
        # Fit and plot trendline
        model = LinearRegression(fit_intercept=True)
        model.fit(np.array(y).reshape(-1, 1), x)
        trendline = model.predict(np.array(densities).reshape(-1, 1))
        trendlines.append(trendline)
        
        if i == 2:
            linewidth = 1.5
        else:
            linewidth = 0.75
        ax.plot(trendline, np.array(densities).reshape(-1, 1), linewidth=linewidth, color=line_colors[i])

    # Adding colored segments to indicate lane requirements between each pair of trendlines
    densities_array = np.array(densities).reshape(-1, 1)
    
    ax.fill_betweenx(densities_array.flatten(), 0, trendlines[0], color=segment_colors[0], alpha=0.3, label='1 sublane')

    for i in range(len(trendlines) - 1):
        ax.fill_betweenx(densities_array.flatten(), trendlines[i], trendlines[i + 1],
                         color=segment_colors[i+1], alpha=0.3, label=f'{i+2} sublanes')
    
    ax.fill_betweenx(densities_array.flatten(), trendlines[-1], 4000, color=segment_colors[-1], alpha=0.3, label='>3 sublanes')

    # Legend, labels, and grid
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.set_xlim(-100, 4000)
    ax.set_xlabel('$q$ [bicycles/h]')
    ax.set_ylabel('$k_{max}$ [bicycles/km/m]')
    ax.text(250, 30, "supply-oriented design", color='black',
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square, pad=0.3'))
    ax.text(2150, 25, "demand-oriented design", color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))

    # Display and save the plot
    plt.savefig('figures/Figure_9.png', dpi=300, bbox_inches='tight')
    plt.show()
    return

