#from PyQt5.QtWidgets import QApplication
#from PyQt5 import QtGui
#import pyqtgraph.opengl as gl
#import pyqtgraph
# pyqtgraph.setConfigOption('background', (255, 255, 200))
# pyqtgraph.setConfigOption('foreground', 'k')

#pyqtgraph.setConfigOption('background', 'w')
#pyqtgraph.setConfigOption('foreground', 'k')
#pyqtgraph.setConfigOptions(antialias=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import cv2
from .utils import load_points


def plot_calib_board(img_points, board_shape, camera_resolution, frame_fpath=None):
    corners = np.array(img_points, dtype=np.float32)
    plt.figure(figsize=(8, 4.5))
    if frame_fpath:
        plt.imshow(plt.imread(frame_fpath))
    for pts in corners:
        pts = pts.reshape(-1, 2)
        cols = board_shape[0]
        rows = board_shape[1]
        edges = []
        for r in range(rows):
            for c in range(cols - 1):
                edges.append(c + r * cols)
                edges.append(c + r * cols + 1)
        for c in range(cols):
            for r in range(rows - 1):
                edges.append(c + r * cols)
                edges.append(c + (r + 1) * cols)
        lc = mc.LineCollection(pts[edges].reshape(-1, 2, 2), color='r', linewidths=1)

        plt.gca().add_collection(lc)
        plt.gca().set_xlim((0, camera_resolution[0]))
        plt.gca().set_ylim((camera_resolution[1], 0))
    plt.show()