# LoopDetectorData
I wrote this code to analyse raw detector data collected a four sites in Muenster, Germany. The data were collected using the Zelt<sup>Evo</sup> from EcoCounter. 

The analysis are published in the following paper: 
  - Kaths, H., Roosta, A., Fischer, J., Kathmann, T., & Pusica, A. (2024). "Propagation of density across the width of a bicycle path", submitted for consideration in the Transportation Research Record

Please cite the paper if you use the code or data. 

## Analyses:
We are interested in the flow characteristics of bicycle traffic and define the following parameters for the entire cross-section and each inductive loop independently. First, the number of cyclists $N_l$ crossing each inductive loops $l$ in a given time interval $T$ is determined. The average bicycle traffic flow $\overline{q_l}$ passing inductive loop $l$ during the time interval $T$ is calculated. The harmonic mean of the speeds $\overline{v_l}$ of all cyclists $i$ that pass loop $l$ in time interval $T$ is calculated. The average density $\overline{k_l}$ at inductive loop $l$ is derived from the average speed $\overline{v_l}$ and the average flow $\overline{q_l}$ and the width of the inductive loop $w_l$. The total number of cyclists detected over the entire cross-section N is the sum across all loops. The corresponding parameters across the entire cross-section, $\overline{q}$, $\overline{v}$ and $\overline{k}$ are calculated using $N$, $v_i$, the speed of all cyclists, and $w$, the total width of the bicycle path. More information about the calculation of speed, density and flow can be found in the paper. 

It is possible to make the following figures with the code:
 - Cumulative density function of all crossings at all site 
 - Cumulative density function of observations at Promenade site compared to distribution from Brandenburg et al. (2022)
 - Scatter plot of speed and flow observations for the entire cross-section and each inductive loop including trendline
 - Scatter plot of density and flow observations for the entire cross-section and each inductive loop including trendline
 - Heatmaps of the density by flow by inductive loop for each site
 - Heatmap of the density by flow for each 1 m sublane for all sites combined
 - Lines showing the development of density across the cross-section of the bicycle lane

Please note that the code is quite slow, especially when first processing the raw data. I would be very grateful if a pro could speed it up!

