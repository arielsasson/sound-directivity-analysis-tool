from vedo import Sphere, Plotter
import time

class MouseFollower:
    def __init__(self, plotter, mesh):
        self.plotter = plotter
        self.mesh = mesh
        self.cursor = None
        self.locked = False  # Start unlocked so it follows after first click
        self._last_move_time = 0
        self._move_interval = 1/60  # 30 FPS = 33ms

    def on_mouse_move(self, evt):
        if self.locked or not self.cursor:
            return
        
        now = time.time()
        if (now - self._last_move_time < self._move_interval):
            # Skip update, too soon
            return
        self._last_move_time = now

        picked = evt.picked3d
        actor = evt.actor

        if picked is not None and (actor == self.mesh or actor == self.cursor):
            self.cursor.pos(picked)
            self.plotter.render()

    def on_left_click(self, evt):
        picked = evt.picked3d
        actor = evt.actor

        if picked is None or (actor != self.mesh and actor != self.cursor):
            return

        if not self.cursor:
            self.cursor = Sphere(pos=picked, r=0.02, c='red')
            self.plotter += self.cursor
            self.locked = False
        else:
            self.locked = not self.locked

        self.plotter.render()


    def on_right_click(self, evt):
        if self.cursor:
            self.plotter.remove(self.cursor)
            self.cursor = None
            self.locked = False
            self.plotter.render()

# Setup mesh and plotter
mesh = Sphere().wireframe(False).c('lightblue')
plotter = Plotter()
follower = MouseFollower(plotter, mesh)

plotter += mesh
plotter.add_callback("mouse move", follower.on_mouse_move)
plotter.add_callback("mouse left click", follower.on_left_click)
plotter.add_callback("mouse right click", follower.on_right_click)

plotter.show(interactive=True)