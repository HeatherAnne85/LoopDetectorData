# LoopDetectorData
I wrote this code to analyse raw detector data collected a four sites in Muenster, Germany. The data were collected using the Zelt<sup>Evo</sup> from EcoCounter. 

The analysis are published in the following paper: 
  - Kaths, H., Roosta, A., Fischer, J., Kathmann, T., & Pusica, A. (2024). "Propagation of density across the width of a bicycle path", submitted for consideration in the Transportation Research Record
Please cite the paper if you use the code or data. 

Analyses:
We are interested in the flow characteristics of bicycle traffic and define the following parameters for the entire cross-section and each inductive loop independently. First, the number of cyclists $N_l$ crossing each inductive loops $l$ in a given time interval $T$ is determined. The average bicycle traffic flow $\overline{q_l}$ passing inductive loop $l$ during the time interval $T$ is calculated. The harmonic mean of the speeds $\overline{v_l}$ of all cyclists $i$ that pass loop $l$ in time interval $T$ is calculated. The average density $\overline{k_l}$ at inductive loop $l$ is derived from the average speed $\overline{v_l}$ and the average flow $\overline{q_l}$ and the width of the inductive loop $w_l$. The total number of cyclists detected over the entire cross-section N is the sum across all loops. 

  $\overline{q_l} = N_{l}/T$

  &#772;v<sub>l</sub> = N<sub>l</sub> (∑<sub>i=0</sub><sup>N<sub>l</sub></sup> v<sub>i,l</sub><sup>-1</sup> )<sup>-1</sup>

  $\overline{k_l} = \frac{\overline{q_l}}{\overline{v_l} \cdot w_l}$

  $N = \sum_{l=1}^3 N_l$

The corresponding parameters across the entire cross-section, $\overline{q}$, $\overline{v}$ and $\overline{k}$ are calculated using $N$, $v_i$, the speed of all cyclists, and $w$, the total width of the bicycle path. 
