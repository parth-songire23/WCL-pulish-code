import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Arrow3D(FancyArrowPatch):
    """
    Custom class to draw 3D arrows in Matplotlib.
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Transform 3D coordinates to 2D projection for rendering."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = renderer.transform_3d(xs3d, ys3d, zs3d)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

class Render:
    """
    3D visualization for the UAV-RIS communication system.
    """
    def __init__(self, system):
        self.system = system
        self.fig = plt.figure(figsize=(10, 7))
        self.pause = False
        self.t_index = 0
        plt.ion()  # Enable interactive mode

    def render_pause(self):
        """Pause rendering and display the last frame."""
        self._render_scene()
        plt.ioff()  # Disable interactive mode
        plt.show()
        self.pause = False

    def render(self, interval=0.5):
        """Continuously update the 3D plot at a given time interval."""
        self._render_scene()
        plt.pause(interval)

    def _render_scene(self):
        """Internal function to configure and update the 3D scene."""
        plt.clf()
        ax = self._setup_3D_plot()
        self._plot_entities(ax)
        self._plot_channels(ax)
        self._plot_text(ax)
        plt.draw()

    def _setup_3D_plot(self):
        """Initialize the 3D plot settings."""
        ax = self.fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_xlim([-25, 25])
        ax.set_ylim([0, 50])
        ax.set_zlim([0, 60])
        ax.view_init(elev=30, azim=45)  # Better perspective
        return ax

    def _plot_entities(self, ax):
        """Plot UAV, RIS, users, and attackers in the 3D space."""
        self._plot_point(ax, self.system.UAV.coordinate, "UAV", "r")
        self._plot_point(ax, self.system.RIS.coordinate, "RIS", "g")

        for user in self.system.user_list:
            self._plot_point(ax, user.coordinate, f"User {user.index}\nCap: {user.capacity:.2f}", "b")

        for attacker in self.system.attacker_list:
            self._plot_point(ax, attacker.coordinate, f"Attacker {attacker.index}\nCap: {attacker.capacity:.2f}", "y")

    def _plot_point(self, ax, coord, label, color):
        """Helper function to plot individual entities in 3D space."""
        ax.scatter(*coord, color=color, s=50, edgecolors="k")
        ax.text(*coord, label, size=10, color=color)

    def _plot_channels(self, ax):
        """Draw communication channels between UAV, RIS, users, and attackers."""
        for channel in self.system.h_U_k + self.system.h_U_p + self.system.h_R_k + self.system.h_R_p:
            self._draw_arrow(ax, channel, "b" if "user" in channel.channel_name else "y")

        self._draw_arrow(ax, self.system.H_UR, "r")

    def _draw_arrow(self, ax, channel, color):
        """Helper function to draw 3D arrows representing wireless channels."""
        start, end = np.array(channel.transmitter.coordinate), np.array(channel.receiver.coordinate)
        mid = (start + end) / 2
        
        ax.text(*mid, f"{channel.channel_name}\nPL: {channel.path_loss_normal:.2f} dB", size=8, color=color)
        
        line = Line3DCollection([list(zip(start, end))], colors=color, linewidths=2)
        ax.add_collection3d(line)

    def _plot_text(self, ax):
        """Display simulation information in the 3D space."""
        ax.text(0, 0, 60, f"Simulation Time: {self.t_index}", size=10, color="b")

if __name__ == "__main__":
    from env import MiniSystem  # Ensure correct import

    system = MiniSystem()
    renderer = Render(system)

    for _ in range(5):  # Simulate 5 steps
        renderer.render(interval=1)
