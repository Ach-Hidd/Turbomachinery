# Assignment 1: Power cycles


A simple Brayton cycle consists of:
* Compression
* Heat addition
* Expansion

![simple_brayton_cycle](./images/simple_brayton_cycle.svg)

<br/><br/>

There are ways to increase specific work and/or thermodynamic efficiency.
For example:

![three_type_cycles](./images/three_type_cycles.svg)


## Tasks

Groups of two students: use student ID number to select which cycle to use.<br>
Calculate numpy.mod(studentID1 + studentID2, 2) + 1

If results is:<br>
$\textbf{1}$ Inter cooling + recuperation<br>
$\textbf{2}$ Reheat + recuperation

**Implement a cycle into this Jupyter Notebook and discuss the following:**
1. What is the impact over the cycle, if losses are included? Which
turbomachinery losses have a greater impact on the performance of
the power cycle and why?
2. How does it compare to the basic Brayton cycle?
3. What is the effect of changing the pressure ratio and/or temperature
ratio on the cycle performance?
4. What are the advantages and disadvantages of the new configuration?

**Hand in:**
* Report (no longer than 4 pages). Use plots from the code below (jupyter notebook).
* Python functions for the cycle.

**Assumptions:**
* No pressure losses in combustion chamber: $PR_{cc}$=1
* For all compressors use polytropic efficiency $\eta_{p,comp}$
* For all turbines use polytropic efficiency $\eta_{p,turb}$
* Isentropic exponent is constant across compression and expansion

Use the code below to explore the performance of the different cycles and implement the cycle requested in the assignment.
Click the {fa}`rocket` --> {guilabel}`Live Code` button above on this page, and run the code below.
It is also possible to download the jupyter notebook and run it locally on your computer. 


## Code

The code consists of 4 blocks:
* 3 functions for the cycles: cycBasic (complete), cycRecup (complete) and cycAssign (needs to be adjusted at designated locations).
* 1 block to start the interactive plot.


**Function basic Brayton cycle:**

def cycBasic(PR, TR, gam_c, gam_t, R, etap_c, etap_t, PR_cc):
    
    # cc .. combustion chamber
    # c  .. compressor 
    # t  .. turbine 
    
    # Inputs, gam_c = gamma_air, gam_t = gamma_flg
    T02 = 293.15
    cp_c = R*gam_c/(gam_c-1)
    cp_t = R*gam_t/(gam_t-1)
    
    # Compressor stage 
    TR_c = PR**((gam_c-1)/gam_c/etap_c)
    T03 = T02*TR_c

    # Combustion stage
    T04 = T02*TR

    # Stuff for pressure loss 
    PR_t = PR*PR_cc

    # Turbine stage
    TRt = (1/PR_t)**((gam_t-1)/gam_t*etap_t)
    T05 = TRt*T04
    
    # Cycle
    w_t = cp_t*(T04-T05)
    w_c = cp_c*(T03-T02)
    hin = (cp_c+cp_t)/2*(T04-T03)
    eta = (w_t-w_c)/hin
    specw = (w_t-w_c)/(cp_c*T02)
    
    # Entropy and temperature
    ds_3 = cp_c*np.log(T03/T02) - R*np.log(PR) # compression
    ds_4 = 0.5*(cp_c+cp_t)*np.log(T04/T03) - R*np.log(PR_cc) # heat injection
    ds_5 = cp_t*np.log(T05/T04) - R*np.log(1/PR_t) # expansion
    
    s_2 = 0
    s_3 = ds_3
    s_4 = s_3 + ds_4
    s_5 = s_4 + ds_5
    
    # Concatenate tuples
    entr = [s_2, s_3, s_4, s_5, s_2] # s_2 = 0, reference!
    temp = [T02, T03, T04, T05, T02]
    
    return specw, eta, entr, temp

**Function basic Brayton cycle with recuperation :**

def cycRecup(PR, TR, gam_c, gam_t, R, etap_c, etap_t, PR_cc):
    
    # cc .. combustion chamber
    # c  .. compressor 
    # t  .. turbine 
    
    # Inputs, gam_c = gamma_air, gam_t = gamma_flg
    T02 = 293.15
    cp_c = R*gam_c/(gam_c-1)
    cp_t = R*gam_t/(gam_t-1)
    
    # Compressor stage 
    TR_c = PR**((gam_c-1)/gam_c/etap_c)
    T03 = T02*TR_c

    # Combustion stage
    T04 = T02*TR

    # Stuff for pressure loss 
    PR_t = PR*PR_cc

    # Turbine stage
    TRt = (1/PR_t)**((gam_t-1)/gam_t*etap_t)
    T05 = TRt*T04
    
    # Cycle
    w_t = cp_t*(T04-T05)
    w_c = cp_c*(T03-T02)
    if T05 > T03:
        hin = (cp_c+cp_t)/2*(T04-T05)
    else:
        hin = (cp_c+cp_t)/2*(T04-T03)
        
    eta = (w_t-w_c)/hin
    specw = (w_t-w_c)/(cp_c*T02)
    
    # Entropy and temperature
    ds_3 = cp_c*np.log(T03/T02) - R*np.log(PR) # compression
    ds_4 = 0.5*(cp_c+cp_t)*np.log(T04/T03) - R*np.log(PR_cc) # heat injection
    ds_5 = cp_t*np.log(T05/T04) - R*np.log(1/PR_t) # expansion
    
    s_2 = 0
    s_3 = ds_3
    s_4 = s_3 + ds_4
    s_5 = s_4 + ds_5
    
    # Concatenate tuples
    entr = [s_2, s_3, s_4, s_5, s_2] # s_2 = 0, reference!
    temp = [T02, T03, T04, T05, T02]
    
    return specw, eta, entr, temp

**Function assignment:**

def cycAssign(PR, TR, gam_c, gam_t, R, etap_c, etap_t, PR_cc):
    
    # cc .. combustion chamber
    # c  .. compressor 
    # t  .. turbine 
    
    # Inputs, gam_c = gamma_air, gam_t = gamma_flg
    T02 = 293.15
    cp_c = R*gam_c/(gam_c-1)
    cp_t = R*gam_t/(gam_t-1)
    
    #------------------------------
    # plug in your cycle model here:
    
    s_2 = 0
    s_3 = ds_3
    s_4 = s_3 + ds_4
    s_5 = s_4 + ds_5
    s_6 = s_5 + ds_6
    s_7 = s_6 + ds_7
    
    # Concatenate tuples
    entr = [s_2, s_3, s_4, s_5, s_6, s_7, s_2] # s_2 = 0, reference!
    temp = [T02, T03, T04, T05, T06, T07, T02]
    
    return specw, eta, entr, temp

**Main:**

import ipywidgets as widgets
from ipywidgets import interact, interactive, interact_manual, interactive_output, Label
import numpy as np
from matplotlib import pyplot as plt

# Define initial parameters
R = 287 # universal gas constant [J kg^-1 K^-1]
init_PR = 8 # Initial pressure ratio
init_TR = 5 # Initial temperature ratio
init_PR_cc = 1 # Initial pressure ratio over combustion chamber
init_etap_c = 1 # Initial polytropic efficiency compressor
init_etap_t = 1 # Initial polytropic efficiency turbine
init_gam_c = 1.4 # Initial heat capacity ratio combustor
init_gam_t = 1.4 # Initial heat capacity ratio turbine

# Define plot parameters
scale_fig1 = 1.5  
fontsize = 16
color = ['b', 'r']
        
def solve_cycle(PR1=init_PR, TR1=init_TR, gam_c1=init_gam_c, gam_t1=init_gam_t, etap_c1=1, etap_t1=1, PR_cc1=init_PR_cc, PR2=init_PR, TR2=init_TR, gam_c2=init_gam_c, gam_t2=init_gam_t, etap_c2=1, etap_t2=1, PR_cc2=init_PR_cc, semilogy=True, cycle1='Basic', cycle2='Basic'): 
    
    # Initialize figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(scale_fig1*6.4*2, scale_fig1*4.8))
    
    # Set parameters left subplot
    ax1.set_xlabel(r'Entropy [$J.K^{-1}kg^{-1}$]', fontsize=fontsize)
    ax1.set_ylabel(r'Temperature [$K$]', fontsize=fontsize)
    ax1.set_xlim(-650, 2000)
    ax1.set_ylim(200, 2300)
    
    y_ticks_list1 = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200]
    y_ticks_list2 = [200, 400, 600, 800, 1000, 1200, 1400, 1800, 2200]
    y_ticks_list3 = [str(y_tick) for y_tick in y_ticks_list2]
    
    # Check if left subplot uses linear or semilogical scale on the y-axis
    if semilogy == True:
        ax1.set_yscale("log")
        ax1.set_yticks(y_ticks_list2)
        ax1.set_yticklabels(y_ticks_list3)
    elif semilogy == False:
        ax1.set_yscale("linear")
        ax1.set_yticks(y_ticks_list1) 
        
    # Set parameters right subplot
    ax2.set_xlabel('Specific work', fontsize=fontsize)
    ax2.set_ylabel('Efficiency', fontsize=fontsize)
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 1)
    
    # Lists of cycle parameters
    cycle = [cycle1, cycle2]
    PR = [PR1, PR2]
    TR = [TR1, TR2]
    gam_c = [gam_c1, gam_c2]
    gam_t = [gam_t1, gam_t2]
    etap_c = [etap_c1, etap_c2]
    etap_t = [etap_t1, etap_t2]
    PR_cc = [PR_cc1, PR_cc2]
    
    # Initialize arrays for contour plots
    PR_range = np.arange(2., 64.01, 2.0)
    TR_range = np.arange(3., 7.01, 0.5)
    specw_list = np.zeros((2, len(PR_range), len(TR_range)))
    eta_list = np.zeros((2, len(PR_range), len(TR_range)))
    P = np.zeros((len(PR_range), len(TR_range)))
    T = np.zeros((len(PR_range), len(TR_range)))
    
    # PLOTS
    for nr in range(len(cycle)):
        
        i = 0
        j = 0
        
        # LEFT PLOT: cycle 1 and 2
        if cycle[nr] == 'Basic': my_func = cycBasic
        if cycle[nr] == 'Recuperated': my_func = cycRecup
        if cycle[nr] == 'Assignment': my_func = cycAssign
        
        [specw, eta, entr0, temp0] = my_func(PR[nr], TR[nr], gam_c[nr], gam_t[nr], R, etap_c[nr], etap_t[nr], PR_cc[nr])
        line_left, = ax1.plot(entr0, temp0, lw=2, ls='-', c=color[nr])
        line_right, = ax2.plot(specw, eta, lw=2, c=color[nr], marker='o', ms=12)
        
        # RIGHT PLOT: Plot pressure and temperature contours
        for i in range(len(PR_range)):
            for j in range(len(TR_range)):
                if cycle[nr] == 'Basic': my_func = cycBasic
                if cycle[nr] == 'Recuperated': my_func = cycRecup
                if cycle[nr] == 'Assignment': my_func = cycAssign
                
                [specw_test, eta_test, entr0_test, temp0_test] = cycBasic(PR_range[i], TR_range[j], gam_c[nr], gam_t[nr], R, etap_c[nr], etap_t[nr], PR_cc[nr])
                specw_list[nr][i][j] = specw_test
                eta_list[nr][i][j] = eta_test
                P[i][j] = PR_range[i]
                T[i][j] = TR_range[j]
                
                j +=1
            i += 1
            
        CS_P = ax2.contour(specw_list[nr], eta_list[nr], P, levels=[2, 4, 8, 16, 32, 63.9999], colors=color[nr], linestyles='--', linewidths=1)
        ax2.clabel(CS_P, fontsize=fontsize, inline=True)
        
        CS_T = ax2.contour(specw_list[nr], eta_list[nr], T, levels=[3, 4, 5, 6, 6.9999], colors=color[nr], linestyles='--', linewidths=1)
        ax2.clabel(CS_T, fontsize=fontsize, inline=True)
        
        CS_P_current = ax2.contour(specw_list[nr], eta_list[nr], P, levels=[PR[nr]], colors=color[nr], linestyles='-', linewidths=3)
        ax2.clabel(CS_P_current, fontsize=fontsize, inline=True)
        
        CS_T_current = ax2.contour(specw_list[nr], eta_list[nr], T, levels=[TR[nr]], colors=color[nr], linestyles='-', linewidths=3)
        ax2.clabel(CS_T_current, fontsize=fontsize, inline=True)
        
        ### Show minimum, maximum temperature, cycle efficiency and specific work
        if nr == 0:
            p1.value = str(round(np.min(temp0), 2))
            p2.value = str(round(np.max(temp0), 2))
            p3.value = str(round(100*np.min(eta), 2))
            p4.value = str(round(specw, 2))
        else:
            q1.value = str(round(np.min(temp0), 2))
            q2.value = str(round(np.max(temp0), 2))
            q3.value = str(round(100*np.min(eta), 2))
            q4.value = str(round(specw, 2))
    
    # LEFT PLOT: Isobars
    cp = R*gam_c1/(gam_c1-1) # Specific heat capacity
    isobars = [1, 2, 4, 8, 16, 32, 64]
    power_isobars = [isobar ** ((gam_c1-1)/gam_c1) for isobar in isobars]
    T02isob = [power_isobar*293.15 for power_isobar in power_isobars]
    s = np.linspace(-650,2000)
    TRisob = np.exp(s/cp)

    for isobar in T02isob:
        ax1.plot(s, isobar*TRisob, 'k', lw=0.5) 

    # Turn on grid
    ax1.grid()
    ax2.grid()
    
    # Set fontsize x- and y- tick labels
    ax1.xaxis.set_tick_params(labelsize=fontsize)
    ax1.yaxis.set_tick_params(labelsize=fontsize)
    ax2.xaxis.set_tick_params(labelsize=fontsize)
    ax2.yaxis.set_tick_params(labelsize=fontsize)
    
    # Update plot
    plt.show()

# Define interactive widgets
cycle1 = widgets.Dropdown(options=['Basic','Recuperated','Assignment'], description=r'\(\color{blue} {' + 'Cycle\ 1'  + '}\)')
cycle2 = widgets.Dropdown(options=['Basic','Recuperated','Assignment'], description=r'\(\color{red} {' + 'Cycle\ 2'  + '}\)')
a = widgets.FloatLogSlider(value=init_PR, base=2, min=1, max=6, step=0.1, description='PR1', style = {'handle_color': 'blue'})
b = widgets.FloatLogSlider(value=init_PR, base=2, min=1, max=6, step=0.1, description='PR2', style = {'handle_color': 'red'})
c = widgets.FloatSlider(value= init_TR, min=3, max=7, step=0.5, description='TR1', style = {'handle_color': 'blue'})
d = widgets.FloatSlider(value= init_TR, min=3, max=7, step=0.5, description='TR2', style = {'handle_color': 'red'})
e = widgets.FloatSlider(value= init_gam_c, min=1., max=1.8, step=0.01, description=r'$\gamma_{comp,1}$', style = {'handle_color': 'blue'})
f = widgets.FloatSlider(value= init_gam_c, min=1., max=1.8, step=0.01, description=r'$\gamma_{comp,2}$', style = {'handle_color': 'red'})
g = widgets.FloatSlider(value= init_gam_t, min=1., max=1.8, step=0.01, description=r'$\gamma_{turb,1}$', style = {'handle_color': 'blue'})
h = widgets.FloatSlider(value= init_gam_t, min=1., max=1.8, step=0.01, description=r'$\gamma_{turb,2}$', style = {'handle_color': 'red'})
i = widgets.FloatSlider(value= init_PR_cc, min=0.01, max=1., step=0.01, description=r'$PR_{cc,1}$', style = {'handle_color': 'blue'})
j = widgets.FloatSlider(value= init_PR_cc, min=0.01, max=1., step=0.01, description=r'$PR_{cc,2}$', style = {'handle_color': 'red'})
k = widgets.FloatSlider(value= init_etap_c, min=0.01, max=1., step=0.01, description=r'$\eta_{p,comp1}$', style = {'handle_color': 'blue'})
l = widgets.FloatSlider(value= init_etap_c, min=0.01, max=1., step=0.01, description=r'$\eta_{p,comp2}$', style = {'handle_color': 'red'})
m = widgets.FloatSlider(value= init_etap_t, min=0.01, max=1., step=0.01, description=r'$\eta_{p,turb1}$', style = {'handle_color': 'blue'})
n = widgets.FloatSlider(value= init_etap_t, min=0.01, max=1., step=0.01, description=r'$\eta_{p,turb2}$', style = {'handle_color': 'red'})
o = widgets.Checkbox(True,  description='Plot in semilogy')

# Define labels cycle 1
p1 = widgets.Text(value='', description=r'$\color{blue} {' + 'T_{min} [K] ='  + '}$', disabled=True)
p2 = widgets.Text(value='', description=r'$\color{blue} {' + 'T_{max} [K] ='  + '}$', disabled=True)
p3 = widgets.Text(value='', description=r'$\color{blue} {' + '\eta [\%] ='  + '}$', disabled=True)
p4 = widgets.Text(value='', description=r'$\color{blue} {' + 'w_{s} ='  + '}$', disabled=True)

# Define labels cycle 2
q1 = widgets.Text(value='', description=r'$\color{red} {' + 'T_{min} [K] ='  + '}$', disabled=True)
q2 = widgets.Text(value='', description=r'$\color{red} {' + 'T_{max} [K] ='  + '}$', disabled=True)
q3 = widgets.Text(value='', description=r'$\color{red} {' + '\eta [\%] ='  + '}$', disabled=True)
q4 = widgets.Text(value='', description=r'$\color{red} {' + 'w_{s} ='  + '}$', disabled=True)
    
# Layout of widgets
ui0 = widgets.HBox([cycle1, cycle2])
ui1 = widgets.HBox([a, b, p1, q1])
ui2 = widgets.HBox([c, d, p2, q2])
ui3 = widgets.HBox([e, f, p3, q3])
ui4 = widgets.HBox([g, h, p4, q4])
ui5 = widgets.HBox([i, j])
ui6 = widgets.HBox([k, l])
ui7 = widgets.HBox([m, n])
ui8 = widgets.HBox([o])
ui = widgets.VBox([ui0, ui1, ui2, ui3, ui4, ui5, ui6, ui7, ui8])

# Activate interactivity with plot!
out = widgets.interactive_output(solve_cycle, {'PR1': a, 'PR2': b, 'TR1': c, 'TR2': d, 'gam_c1': e, 'gam_c2': f, 'gam_t1': g, 'gam_t2': h, 'PR_cc1': i, 'PR_cc2': j, 'etap_c1': k, 'etap_c2': l, 'etap_t1': m, 'etap_t2': n, 'semilogy':o, 'cycle1': cycle1, 'cycle2': cycle2})
display(ui, out)


