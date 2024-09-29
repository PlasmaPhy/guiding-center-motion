import multiprocessing as mp
import vpython as vp
import numpy as np
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

    cwp.toruspoints(percentage=100, truescale=True)
    R = cwp.Rtorus
    a = cwp.atorus
    rate = 30
    running = 1

    # Grab configuration
    Config = cwp.Config
    defaults = Config.defaults
    print(Config.torus_color)

    if "min_step" in params:
        min_step = R * params["min_step"]
    else:
        min_step = R * defaults["min_step"]

    if "seconds" in params:
        seconds = params["seconds"]
    else:
        seconds = defaults["seconds"]

    print(f"Minimum step size:\t{np.around(min_step,5)}.")
    print(f"Animation duration:\t{seconds} seconds.")

    def dist_compress(min_step: None):

        # Grab particle's data
        x = cwp.cartx
        y = cwp.cartz
        z = cwp.carty
        n = len(cwp.tspan)  # number of steps

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

    def _setup_torus(truescale: bool = True):

        # Canvas
        width = 1920
        height = 850
        scene = vp.canvas(width=width, height=height, background=vp.color.white)
        scene.userpan = False
        scene.center = vp.vector(0, -2.5, 0)

        # Vertical axis
        height = 1 * (2 * R)
        pos = [vp.vector(0, -height / 2, 0), +vp.vector(0, height / 2, 0)]
        vaxis = vp.curve(pos=pos, color=eval("vp.color." + Config.vaxis_color), radius=0.03)

        # Torus walls
        shape = vp.shapes.circle(radius=a, np=60)
        path = vp.paths.circle(radius=R, np=60)
        torus = vp.extrusion(
            pos=vp.vector(0, 0, 0),
            shape=shape,
            path=path,
            color=eval("vp.color." + Config.torus_color),
            opacity=0.4,
        )

        return scene, vaxis, torus

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
    scene, vaxis, torus = _setup_torus()
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
