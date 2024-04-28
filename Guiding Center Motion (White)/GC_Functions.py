import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#____________________________________params/init________________________

def get_params(a, b):
    global params, init
    params, init = a, b

#_____________________________________ODE System________________________

def dSdt(t, S, mu = None):
    theta, psi, z, Pz, rho = S
    mu = params["mu"]
    # Intermediate values
    q = 1
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    r = np.sqrt(2*psi)
    B = 1 - r*cos_theta
    par = mu + rho**2 * B

    theta_dot   =   1/q * (rho*B**2 - par*cos_theta/r)
    psi_dot     = - 1/q * par*r*sin_theta
    z_dot       =   rho*B**2
    pz_dot      =   0
    rho_dot     =   psi_dot

    return np.array([theta_dot, psi_dot, z_dot, pz_dot, rho_dot])

#_______________________________________Solver__________________________

def motion(init, tspan, params):
    sol = odeint(dSdt, y0 = init, t=tspan, tfirst=True)
    theta_sol   = sol.T[0]
    psi_sol     = sol.T[1]
    z_sol       = sol.T[2]
    Pz_sol      = sol.T[3]
    rho_sol     = sol.T[4]
    return np.array([theta_sol, psi_sol, z_sol, Pz_sol, rho_sol])

#_______________________________Conversion to canonical_________________

def canonical(sol):
    theta_sol, psi_sol, z_sol, Pz_sol, rho_sol = sol
    g = params["g"]

    theta = theta_sol
    Ptheta = psi_sol
    z = z_sol
    Pz = rho_sol*g - psi_sol # = psip_sol
    return np.array([theta, Ptheta, z, Pz])

#_____________________________________Plotters__________________________

def time_plots(tspan, sol):
    theta_sol, psi_sol, z_sol, Pz_sol, rho_sol = sol

    scatter_kw = {"s" : 0.08, "color" : "blue"}
    ylabel_kw  = {"rotation" : 0, "fontsize" : 10}
    fig, ax = plt.subplots(4,1, sharex = True)

    ax[0].scatter(tspan, theta_sol, **scatter_kw);
    ax[0].set_ylabel("$\\theta(t)$", **ylabel_kw);
    ax[1].scatter(tspan, z_sol, **scatter_kw);
    ax[1].set_ylabel("$\\zeta(t)$\t", **ylabel_kw);
    ax[2].scatter(tspan, psi_sol, **scatter_kw);
    ax[2].set_ylabel("$\\psi(t)$\t", **ylabel_kw);
    ax[3].scatter(tspan, rho_sol, **scatter_kw);
    ax[3].set_ylabel("$\\rho(t)$", **ylabel_kw);

    plt.xlabel("$t$");

def drift_plot(theta, Ptheta, z, Pz, mod = False):
    scatter_kw = {"s" : 0.08, "color" : "blue"}
    ylabel_kw  = {"rotation" : 0, "fontsize" : 10}
    fig, ax = plt.subplots(1,2, figsize = (12,5))

    if mod: theta = np.mod(theta, 2*np.pi)

    ax[0].scatter(theta, Ptheta, **scatter_kw);
    ax[0].set_xlabel("$\\theta$");
    ax[0].set_ylabel("$P_\\theta$", **ylabel_kw);
    ax[1].scatter(z, Pz, **scatter_kw);
    ax[1].set_xlabel("$\\zeta$");
    ax[1].set_ylabel("$P_ζ$", **ylabel_kw);
    plt.sca(ax[0])
    plt.xticks(np.linspace(-np.pi, np.pi, 5), ["-π", "-π/2", "0", "π/2", "π"]);

#_____________________________________Contours__________________________

def contour3d(limits):
    # Calculate W values
    theta_min, theta_max, psi_min, psi_max, Pz_min, Pz_max = limits
    theta3d, psip3d, Pz3d = np.meshgrid(np.linspace(theta_min, theta_max, 100), 
                                        np.linspace(psi_min, psi_max, 100), 
                                        np.linspace(Pz_min, Pz_max, 100))
    data = calcW(theta3d, psip3d, Pz3d)
    W_span = [data.min(), data.max()]

    # Create a figure with 3D ax
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(40, -30, 0)
    ax.set_box_aspect(None, zoom=1)
    ax.set_facecolor('white')
    kw = {'vmin': W_span[0], 'vmax': W_span[1], 
          'levels': np.linspace(W_span[0], W_span[1], 40), "cmap" : plt.cm.plasma}
    

    # Plot each contour
    _ = ax.contourf(theta3d[:, :, 0], psip3d[:, :, 0], data[:, :, -1],  
                    zdir='z', offset=Pz_max, **kw)
    _ = ax.contourf(theta3d[0, :, :], data[0, :, :], Pz3d[0, :, :],     
                    zdir='y', offset=psi_min, **kw)
    C = ax.contourf(data[:, -1, :], psip3d[:, -1, :], Pz3d[:, -1, :],   
                    zdir='x', offset=theta_max, **kw)

    # Set limits of the plot from coord limits
    ax.set(xlim=[theta_min, theta_max], ylim=[psi_min, psi_max], zlim=[Pz_min, Pz_max])

    # Plot edges
    edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
    ax.plot([theta_max, theta_max], [psi_min, psi_max], Pz_max, **edges_kw)
    ax.plot([theta_min, theta_max], [psi_min, psi_min], Pz_max, **edges_kw)
    ax.plot([theta_max, theta_max], [psi_min, psi_min], [Pz_min, Pz_max], **edges_kw)

    # Set labels and zticks
    ax.set(xlabel='$\\theta$', ylabel='$\\psi_p$', zlabel='$P_ζ$');
    ax.tick_params(axis="x", which='major', labelsize=6);
    ax.tick_params(axis="y", which='major', labelsize=6);
    ax.tick_params(axis="z", which='major', labelsize=6);
    plt.xticks(np.linspace(-np.pi, np.pi, 5), ["-π", "-π/2", "0", "π/2", "π"])

    #Plot colorbar
    fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2);

def contour(limits):
    # Calculate W values
    theta_min, theta_max, psi_min, psi_max, Pz0 = limits
    theta2d, psip2d = np.meshgrid(np.linspace(theta_min, theta_max, 100), 
                                        np.linspace(psi_min, psi_max, 100))
    data = calcW(theta2d, psip2d, Pz0)
    W_span = [data.min(), data.max()]

    # Create figure
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    kw = {'vmin': W_span[0], 'vmax': W_span[1], 'levels': 30,
          "cmap" : plt.cm.plasma};
    
    # Contour plot
    C = ax.contourf(theta2d, psip2d, data, **kw);
    ax.set(xlim=[theta_min, theta_max], ylim=[psi_min, psi_max])
    ax.set_xlabel("$\\theta$");
    ax.set_ylabel("$\\psi_p$", rotation = 0);
    #plt.xticks(np.linspace(-np.pi, np.pi, 5), ["-π", "-π/2", "0", "π/2", "π"]);
    fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2);

#_________________________________ Constant of Motion ___________________

def calcW(theta, psi, Pz):
    mu = params["mu"]
    g = params["g"]

    psip = psi
    r = np.sqrt(2*psi)
    B = 1 - r*np.cos(theta)

    return (Pz + psip)**2 * B**2/(2*g**2) + mu*B

#_____________________________Multiple Orbits & contour___________________

def orbits_contour(theta0s, psip0s, tspan, init, params):

    *_, z0, Pz0, rho0 = init

    for i in range(len(psip0s)):

        # Initial conditions
        psip0 = psip0s[i]
        theta0 = theta0s[i]
        rho0 = Pz0 + psip0

        # Update parameters
        init = np.array([theta0, psip0, z0, Pz0, rho0])
        get_params(params, init)

        # Motion
        sol = motion(init, tspan, params)
        theta_sol, Ptheta_sol, *_ = sol

        # Move theta to [-π,π]
        theta_mod = np.mod(theta_sol, 2*np.pi)
        theta_mod = theta_mod - 2*np.pi*(theta_mod > np.pi)

        # If Ptheta starts increasing too much, replace all values after that 
        # point with 0, else the system returns random numbers
        if any(Ptheta_sol[Ptheta_sol > 2]): 
            idx = np.argmax(Ptheta_sol > 2)
            Ptheta_sol[idx:] = 0
        
        plt.scatter(theta_mod, Ptheta_sol, s = 0.08, c = "r")
        plt.scatter(theta0s, psip0s, s = 20, c = "k");