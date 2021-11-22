import numpy as np
import matplotlib.pyplot as plt
from inventory_model import InvMdlSingle, GenDemand, GenLeadTime


# Fixing random state for reproducibility
# noinspection PyTypeChecker
class VisualizeModel:
    """Visualizing Designed for model
     """

    def __init__(self, x, y):
        self.x = x["Data"]
        self.x_label = x["Label"]
        self.y = y["Data"]
        self.y_label = y["Label"]

    def scatter_hist(self, ax, ax_histx, ax_histy):
        """reference https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery
        -lines-bars-and-markers-scatter-hist-py """

        # no labels
        ax_histx.tick_params(axis="x", labelbottom=self.x_label)
        ax_histy.tick_params(axis="y", labelleft=self.y_label)

        # the scatter plot:
        ax.scatter(self.x, self.y)

        # now determine nice limits by hand:
        range_x = np.subtract(self.x.max(), self.x.min())
        range_y = np.subtract(self.y.max(), self.y.min())
        bin_width_x = range_x / np.log(self.x.size)
        bin_width_y = range_y / np.log(self.y.size)
        bins_x = np.arange(self.x.min(), self.x.max(), bin_width_x.round())
        bins_y = np.arange(self.y.min(), self.y.max(), bin_width_y.round())
        ax_histx.hist(self.x, bins=bins_x)
        ax_histy.hist(self.y, bins=bins_y, orientation='horizontal')

    def plot_scatter_hist(self) -> object:
        # Definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        # start with a square Figure
        fig = plt.figure(figsize=(8, 8))

        ax: plt = fig.add_axes(rect_scatter)
        ax_histx: plt = fig.add_axes(rect_histx, sharex=ax)
        ax_histy: plt = fig.add_axes(rect_histy, sharey=ax)

        self.scatter_hist(ax, ax_histx, ax_histy)
        # use the previous defined function
        plt.show()

    def plot_scatter(self) -> object:
        plt.scatter(self.x, self.y)
        plt.show()


