import sys
import argparse
import numpy as np
from collections import defaultdict
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from pyqtgraph import LegendItem
from pyqtgraph.Qt import QtCore
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class CostPlotter(Node):
    def __init__(self, topic_settings, interval):
        super().__init__('cost_plotter')
        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="MPC Cost Plotter")

        self.plots = {}          # key = (row, col)
        self.legends = {}        # key = (row, col)
        self.curves = defaultdict(dict)
        self.data_cache = defaultdict(lambda: np.zeros(1))

        for topic, row, col, ymin, ymax in topic_settings:
            key = (row, col)
            if key not in self.plots:
                plot = self.win.addPlot(row=row, col=col, title=f"Plot {row},{col}")
                plot.setYRange(ymin, ymax)
                self.plots[key] = plot

                legend = LegendItem(offset=(30, 30))
                legend.setParentItem(plot.graphicsItem())
                self.legends[key] = legend
            else:
                plot = self.plots[key]
                legend = self.legends[key]

            color = pg.intColor(len(self.curves[key]))
            curve = plot.plot(pen=color, name=topic)
            legend.addItem(curve, topic)
            self.curves[key][topic] = curve

            self.create_subscription(
                Float32MultiArray,
                topic,
                self.make_callback(topic),
                10
            )

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(int(interval * 1000))

    def make_callback(self, topic_name):
        def callback(msg):
            self.data_cache[topic_name] = np.array(msg.data)
        return callback

    def update(self):
        for key, topic_curve_map in self.curves.items():
            for topic, curve in topic_curve_map.items():
                curve.setData(self.data_cache[topic])

    def spin(self):
        self.app.exec()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', nargs=5, action='append', metavar=('NAME', 'ROW', 'COL', 'YMIN', 'YMAX'),
                        help="Add a topic to plot: NAME ROW COL YMIN YMAX", required=True)
    parser.add_argument('--interval', type=float, default=0.05, help="Refresh interval in seconds")
    return parser.parse_args()

def main():
    args = parse_args()
    topic_settings = []
    for entry in args.topic:
        name, row, col, ymin, ymax = entry
        topic_settings.append((name, int(row), int(col), float(ymin), float(ymax)))

    rclpy.init()
    plotter = CostPlotter(topic_settings, args.interval)
    plotter.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()