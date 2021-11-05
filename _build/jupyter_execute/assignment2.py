# Assignment 2: Blade Design

## Tasks
**An axial turbine stage with the following costumer specifications has to be designed:**

* A power output of 4 MW.
* The expanding mass flow [$R_{gas}$=287 J/(kgK) and isentropic exponent $\gamma$=1.334] is $\dot{m}$=40 kg/s.
* Total conditions at the inlet of the turbine: $p_{01}$=2.42 bar and $T_{01}$=800 K.
* Shaft speed 12000 rpm
* The hub radius at the inlet of the stator vane ($R_{i}$ in figure) is 30 cm.
* Assume that the axial velocity across the whole stage is constant and $c_{x}$=150 m/s.

__Perform the following tasks:__

1. Design the blade at mid-span: calculate velocities and blade angles.
2. Blade height at positions 1, 2 and 3.
3. Area of the passage at positions 1, 2 and 3.
4. Static pressure ratio across the stage.
5. Three-dimensional blade shape using free vortex design
6. Plot the radial distribution of the loading factor, flow coefficient and degree of reaction.
7. Discuss possible drawbacks of this design: Comment on degree of reaction and highest Mach (relative and absolute)
    
**Bonus point:** approximate 3D design (several assumption need to be made,
such as blade thickness, number of stator vanes and rotor blades, etc.)
Deliver a report: Around 4 pages. Group of 2 students.

## Code
**Function turbine blades:**

def turbine_blades(phi, psi, R):
    
    c_x = 1.0
    U = c_x/phi
    
    A = np.array([[1, 1], [1, -1]])
    b = np.array([(1-2*R)/phi, (psi+1)/phi])
    x = np.linalg.solve(A, b)
    
    alfa2 = np.arctan(x[0])*(180/np.pi)
    beta2 = np.arctan(x[0] - U)*(180/np.pi)
    
    alfa3 = np.arctan(x[1] + U)*(180/np.pi)
    beta2 = np.arctan(x[1])*(180/np.pi)
    
    # Bezier 2nd order for stator and rotor blades
    t = np.linspace(0, 1, 21)
    F = np.column_stack(((1-t)**2, 2*t*(1-t),  t**2))
    
    # Profile thickness
    prof_th = 1.
    
    # Camber line stator
    stat = np.array([[U + x[1], 0.], [0, -1], [-x[0], -2.]])
    bez_stator = F.dot(stat)     
    
    # Stator profile pressure side (PS)
    statorPS = np.array([[U + x[1], 0.], [-prof_th/2, -1.], [-x[0], -2.]])
    bez_statorPS = F.dot(statorPS)
    
    # Stator profile suction side (SS)
    statorSS = np.array([[U + x[1], 0.], [prof_th/2., -1.], [-x[0], -2.]])
    bez_statorSS = F.dot(statorSS)
    
    stator = np.vstack((bez_statorSS, bez_statorPS[::-1]))
    
    # Camber line rotor
    rot = np.array([[0., -3.], [-x[0] + U , -4.], [-x[0] + U - x[1], -5.]])
    bez_stator = F.dot(rot)     
    
    # Rotor profile pressure side (PS)
    rotorPS = np.array([[0., -3.], [-x[0] + U - prof_th/2., -4.], [-x[0] + U - x[1], -5.]])
    bez_rotorPS = F.dot(rotorPS)
    
    # Rotor profile suction side (SS)
    rotorSS = np.array([[0., -3.], [-x[0] + U + prof_th/2., -4.], [-x[0] + U - x[1], -5.]])
    bez_rotorSS = F.dot(rotorSS)
    
    rotor = np.vstack((bez_rotorSS, bez_rotorPS[::-1]))
    
    return stator, rotor, x, U

**Main:**

import ipywidgets as widgets
from ipywidgets import interact, interactive, interact_manual, interactive_output, Label
import numpy as np
from matplotlib import pyplot as plt

# Define initial parameters
init_phi = 0.4 # Flow coefficient
init_psi = 1 # Loading factor
init_R = 0.5 # Degree of reaction
init_pitch_stat = 3
init_pitch_rot = 3

# Define plot parameters
scale_fig1 = 1.5  
fontsize = 16

def turbine_gui(phi, psi, R, pitch_stat, pitch_rot, show_velocity=True): 
    
    
    offset_y = 0 if show_velocity == True else 0.25

    # Initialize figure 
    fig1, ax1 = plt.subplots(figsize=(scale_fig1*6.4, scale_fig1*4.8))

    # Set parameters left subplot
    ax1.set_xlabel('x', fontsize=fontsize)
    ax1.set_ylabel('y', fontsize=fontsize)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-7, 2)
    
    [stator, rotor, x, U] = turbine_blades(phi, psi, R)
    
    for i in range(-10, 11):
        
        ax1.plot(stator[:,0] + i*pitch_stat, stator[:,1] + 0.25 - offset_y, lw=3, c='r')
        ax1.plot(rotor[:,0] + i*pitch_rot, rotor[:,1] - 0.25 + offset_y, lw=3, c='b')
    
    if show_velocity == True:
        
        offset = 0
        head_length=0.3
        head_width=0.3
        # Draw velocity profiles
        # Stator inlet
        ax1.arrow(offset, 1.5, offset-x[1]-U, -1, color='r', head_length=head_length, head_width=head_width, length_includes_head=True)  # C1

        # Stator outlet / rotor inlet
        ax1.arrow(offset, -2, -x[0], -1, color='r', head_length=head_length, head_width=head_width, length_includes_head=True, label='absolute velocity')  # C2
        ax1.arrow(offset, -2, -x[0]+U, -1, color='b', head_length=head_length, head_width=head_width, length_includes_head=True, label='relative velocity')  # W2
        ax1.arrow(offset-x[0]+U, -3, -U, 0, color='g', head_length=head_length, head_width=head_width, length_includes_head=True, label='rotational velocity')  # U

        # Rotor outlet
        ax1.arrow(offset, -5.5, -x[1]-U, -1, color='r', head_length=head_length, head_width=head_width, length_includes_head=True)  # C3
        ax1.arrow(offset, -5.5, -x[1], -1, color='b', head_length=head_length, head_width=head_width, length_includes_head=True)  # W3
        ax1.arrow(offset-x[1], -6.5, -U, 0, color='g', head_length=head_length, head_width=head_width, length_includes_head=True)  # U
        
        ax1.legend()
        
    # Update plot
    
    plt.show()

# Define interactive widgets
a = widgets.FloatSlider(value=init_phi, min=0, max=2, step=0.01, description=r'$\phi$')
b = widgets.FloatSlider(value=init_psi, min=0, max=5, step=0.01, description=r'$\psi$')
c = widgets.FloatSlider(value=init_R, min=0, max=2, step=0.01, description=r'$R$')
d = widgets.FloatSlider(value=init_pitch_stat, min=0, max=10, step=0.01, description='Pitch stator')
e = widgets.FloatSlider(value=init_pitch_rot, min=0, max=10, step=0.01, description='Pitch rotor')
f = widgets.Checkbox(True,  description='Show velocity triangles')

    
# Layout of widgets
ui0 = widgets.HBox([a, d])
ui1 = widgets.HBox([b, e])
ui2 = widgets.HBox([c, f])

ui = widgets.VBox([ui0, ui1, ui2])

# Activate interactivity with plot!
out = widgets.interactive_output(turbine_gui, {'phi': a, 'psi': b, 'R': c, 'pitch_stat': d, 'pitch_rot': e, 'show_velocity': f})
display(ui, out)


**Function 3D blade design:**

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ipyvolume as ipv

nb = 20
phi = 0.1
LF = np.linspace(2., 1., nb)
DR = np.linspace(0., 0.5 ,nb)

[sta1, rot1, A, U] = turbine_blades(phi, LF[0], DR[0])

n = sta1.shape[0]

Xs = sta1[:,0]
Ys = sta1[:,1]
Zs = 10*np.ones((n, 1))

Xr = rot1[:,0]
Yr = rot1[:,1]
Zr = 10*np.ones((n, 1))


fig = ipv.figure()
ipv.style.use("light") # looks better

for i in range(1,nb):
    
    [sta1, rot1, A, U] = turbine_blades(phi, LF[i], DR[i]);

    Xs = np.column_stack((Xs, sta1[:,0]))
    Ys = np.column_stack((Ys, sta1[:,1]))
    Zs = np.column_stack((Zs, 10 + 10*(i-1)/9*np.ones((n, 1))))
    
    Xr = np.column_stack((Xr, rot1[:,0]))
    Yr = np.column_stack((Yr, rot1[:,1]))
    Zr = np.column_stack((Zr, 10 + 10*(i-1)/9*np.ones((n, 1))))

for j in range(0, nb):
    
    th = 20*(j-1)*np.pi/180;

    Xn = Xs*np.cos(th) - Zs*np.sin(th)
    Zn = Xs*np.sin(th) + Zs*np.cos(th)
    ipv.plot_surface(Xn, Ys*5, Zn, color="red")
    ipv.plot_wireframe(Xn,  Ys*5, Zn, color="red")
    
    Xn = Xr*np.cos(th) - Zr*np.sin(th)
    Zn = Xr*np.sin(th) + Zr*np.cos(th)
    ipv.plot_surface(Xn, Yr*5, Zn, color="blue")
    ipv.plot_wireframe(Xn, Yr*5, Zn, color="blue")
    

ipv.show()

