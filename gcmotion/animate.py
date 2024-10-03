import multiprocessing as mp
import vpython as vp
import numpy as np
from scipy.signal import argrelextrema as ex
from tqdm import tqdm

spawn_ctx = mp.get_context("spawn")


def animate(cwp, params: dict = {}):
    proc = spawn_ctx.Process(target=run, args=(cwp, params), daemon=True)
    proc.start()
    try:
        proc.join()
    finally:
        proc.kill()


def run(cwp, params: dict = {}):

    # Grab configuration
    Config = cwp.Config
    defaults = Config.defaults
    print(Config.torus_color)

    for key in defaults.keys():
        if key not in params:
            params[key] = defaults[key]

    percentage = params["percentage"]
    truescale = params["truescale"]
    min_step = params["min_step"]
    seconds = params["seconds"]

    R, a, r_torus, theta_torus, z_torus = cwp.toruspoints(
        percentage=percentage, truescale=truescale
    )
    # Cartesian (y and z are switched in vpython!)
    x = (R + r_torus * np.cos(theta_torus)) * np.cos(z_torus)
    z = (R + r_torus * np.cos(theta_torus)) * np.sin(z_torus)
    y = r_torus * np.sin(theta_torus)
    rate = 30
    running = 0

    print(f"Minimum step size:\t{np.around(min_step,5)}.")
    print(f"Animation duration:\t{seconds} seconds.")

    def dist_compress(min_step: None):

        n = len(x)  # number of steps

        step = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 + (z[1:] - z[:-1]) ** 2)

        # Create arrays used for plotting
        plotx, ploty, plotz = [np.zeros(n) for _ in range(3)]
        plotx[0], ploty[0], plotz[0] = x[0], y[0], z[0]

        step_buffer = 0
        skipped = 0
        j = 0
        for i in range(n - 2):
            if step_buffer > min_step:
                plotx[j], ploty[j], plotz[j] = x[i], y[i], z[i]
                j += 1
                step_buffer = 0
                skipped = 0
            else:
                step_buffer += step[i]
                skipped += 1

        plotx = np.trim_zeros(plotx, trim="b")
        ploty = np.trim_zeros(ploty, trim="b")
        plotz = np.trim_zeros(plotz, trim="b")

        compression = int((x.shape[0] - plotx.shape[0]) / x.shape[0] * 100)
        print(f"\nCompression level: -{compression}%\n")

        return plotx, ploty, plotz

    def adjust_rate(x, seconds=seconds):
        rate = int(x.shape[0] / seconds)
        print(f"Rate = {rate}/s")
        return rate

    def _setup_scene(truescale: bool = True):

        # Canvas
        width = 1920
        height = 850
        scene = vp.canvas(width=width, height=height, background=vp.color.white)
        scene.userpan = False
        scene.center = vp.vector(0, -0.55 * R, 0)

        # Vertical axis
        height = 1 * (2 * R)
        pos = [vp.vector(0, -height / 2, 0), +vp.vector(0, height / 2, 0)]
        vaxis = vp.curve(pos=pos, color=eval("vp.color." + Config.vaxis_color), radius=0.004 * R)

        # Torus walls
        shape = vp.shapes.circle(radius=float(a), np=60)
        path = vp.paths.circle(radius=float(R), np=60)
        torus = vp.extrusion(
            pos=vp.vector(0, 0, 0),
            shape=shape,
            path=path,
            color=eval("vp.color." + Config.torus_color),
            opacity=0.4,
        )

        return scene, vaxis, torus

    def _flux_surface():

        # Get theta of 1 period
        if cwp.t_or_p == "Trapped":
            span = ex(cwp.theta, np.greater)[0][:2]
            theta = cwp.theta[span[0] : span[1]]

        if cwp.t_or_p == "Passing":
            condition = (np.abs(cwp.theta) > cwp.theta0) & (
                np.abs(cwp.theta) < cwp.theta0 + 2 * np.pi
            )
            theta = cwp.theta[condition]

        r = r_torus[: theta.shape[0]]
        xflux = r * np.cos(theta)
        zflux = r * np.sin(theta)
        zero = np.zeros(theta.shape[0])
        points = np.vstack((xflux, zflux, zero)).T
        vectors = []

        for i in range(len(points)):
            vectors.append(vp.vector(points[i, 0], points[i, 1], points[i, 2]))

        shape = np.vstack((xflux, zflux)).T.tolist()
        shape.append(shape[0])
        path = vp.paths.circle(radius=float(R), np=60)
        flux_surface = vp.extrusion(
            pos=vp.vector(0, 0, 0),
            shape=shape,
            path=path,
            color=eval("vp.color." + Config.flux_surface_color),
            opacity=Config.flux_surface_opacity,
        )

        return flux_surface

    def _setup_particle():

        # Particle
        pos = vp.vector(x[0], y[0], z[0])
        p = vp.sphere(
            pos=pos,
            radius=a / 20,
            color=eval("vp.color." + Config.particle_color),
            make_trail=True,
            trail_radius=R / 1000,
            interval=1,
        )

        return p

    class buttons:
        def runAnimation(foo):
            nonlocal running
            running = not running
            if running:
                foo.text = "Pause"
            else:
                foo.text = "Run"

        def ptrail(foo):
            nonlocal p
            p.make_trail = foo.checked
            if not foo.checked:
                p.clear_trail()

        def rate_slider(foo):
            nonlocal rate
            rate = foo.value

        def restart(foo):
            nonlocal i, p, pbar
            i = 0
            p.clear_trail()
            p.visible = False
            p = _setup_particle()
            pbar.reset()

        def setup():
            vp.radio(bind=buttons.runAnimation, text="Run")
            vp.checkbox(bind=buttons.ptrail, text="Show Trail", checked=True)
            vp.slider(bind=buttons.rate_slider, text="Rate", min=30, max=10000, step=5, value=rate)
            vp.button(bind=buttons.restart, text="Restart")

    x, y, z = dist_compress(min_step)
    rate = adjust_rate(x, seconds=seconds)
    scene, vaxis, torus = _setup_scene()
    # flux = _flux_surface()
    p = _setup_particle()
    buttons.setup()

    i = 0
    with tqdm(total=len(x), leave=True) as pbar:
        while True:
            vp.rate(rate)

            if running:
                p.pos = vp.vector(x[i], y[i], z[i])
                pbar.update(1)
                i += 1

            if i == len(x):
                pbar.refresh()
                scene.waitfor("click")
                buttons.restart("foo")

        pass
