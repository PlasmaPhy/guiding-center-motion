import multiprocessing as mp
import vpython as vp

spawn_ctx = mp.get_context("spawn")


def animate(cwp):
    proc = spawn_ctx.Process(target=run, args=(cwp,), daemon=True)
    proc.start()
    try:
        proc.join()
    finally:
        proc.kill()


def run(cwp):

    R = cwp.Rtorus
    a = cwp.rtorus
    x = cwp.cartx
    z = cwp.carty
    y = cwp.cartz

    # ---------------------------------------------------------------------------

    # Canvas
    v_offset = 2.5  # Should find out how to move the actual camera
    y += v_offset
    width = 1920
    height = 850
    scene = vp.canvas(width=width, height=height, background=vp.color.white)

    # Vertical axis
    height = 1 * (2 * R)
    pos = [vp.vector(0, v_offset - height / 2, 0), +vp.vector(0, v_offset + height / 2, 0)]
    axis = vp.vector(0, 1, 0)
    vaxis = vp.curve(pos=pos, color=vp.color.red, radius=0.03)

    # Torus walls
    shape = vp.shapes.circle(radius=a, np=100)
    path = vp.paths.circle(radius=R, np=100)
    torus = vp.extrusion(
        pos=vp.vector(0, v_offset, 0),
        shape=shape,
        path=path,
        color=vp.color.cyan,
        opacity=0.4,
    )

    # Particle
    pos = vp.vector(x[0], y[0], z[0])
    p = vp.sphere(
        pos=pos,
        radius=a / 20,
        color=vp.color.red,
        make_trail=True,
        trail_radius=a / 300,
        # interval=1,
        pps=10,
    )

    # -----------------------------------------------------------------------

    # Run button

    rate = 30
    running = 0
    show_trail = 1

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
        nonlocal i, p
        i = 1
        p.pos = vp.vector(z[0], y[0], z[0])
        p.clear_trail()

    vp.radio(bind=runAnimation, text="Run")
    vp.checkbox(bind=ptrail, text="Show Trail", checked=True)
    vp.slider(bind=rate_slider, text="Rate", min=30, max=5000, step=10, value=rate)
    vp.button(bind=restart, text="Restart")

    # Run
    i = 1
    while True:
        vp.rate(rate)

        if running:
            p.pos = vp.vector(x[i], y[i], z[i])
            i += 1

        if i == len(x):
            scene.pause()
        pass
