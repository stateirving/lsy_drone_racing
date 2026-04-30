import argparse
from typing import Dict, List
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QApplication
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

COLOR_CYCLE = [
    '#1f77b4', "#ff7f0e", '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
]

class TopicPlot:
    def __init__(self, plot_item, name: str, color):
        self.curve = plot_item.plot(pen=pg.mkPen(color=color, width=2), name=name)
        self.data = []

    def update(self, new_data: List[float]):
        self.data = new_data
        self.curve.setData(self.data)

class CostPlotter(Node):
    def __init__(self, title: str, topic_layout: Dict[str, int], ylims: Dict[int, List[float]], interval: float):
        super().__init__('cost_plotter')
        self.topic_layout = topic_layout
        self.ylims = ylims
        self.interval = interval

        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.subplots: Dict[int, pg.PlotItem] = {}
        self.plots: Dict[str, TopicPlot] = {}

        self.latest_data: Dict[str, List[float]] = {}

        color_idx = 0
        for topic, subplot_idx in topic_layout.items():
            if subplot_idx not in self.subplots:
                plot = self.win.addPlot(title=f"subplot {subplot_idx}")
                plot.showGrid(x=True, y=True)
                if self.ylims.get(subplot_idx, ['default'])[0] != 'default':
                    plot.setYRange(self.ylims[subplot_idx][0], self.ylims[subplot_idx][1])
                plot.addLegend()
                self.subplots[subplot_idx] = plot
                self.win.nextRow()
            color = COLOR_CYCLE[color_idx % len(COLOR_CYCLE)]
            color_idx += 1
            self.plots[topic] = TopicPlot(self.subplots[subplot_idx], topic, color)
            self.create_subscription(Float32MultiArray, topic, self.make_callback(topic), 10)

        # 定时器刷新界面
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(self.interval * 1000))

    def make_callback(self, topic):
        def callback(msg):
            self.latest_data[topic] = msg.data
        return callback

    def update_plot(self):
        rclpy.spin_once(self, timeout_sec=0)
        for topic, data in self.latest_data.items():
            self.plots[topic].update(data)

    def spin(self):
        self.app.exec()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default="MPC Step Cost Plot", help='Window title')
    parser.add_argument('--topic', nargs=5, action='append', metavar=('NAME', 'IDX', 'YMIN', 'YMAX', '_'),
                        help='Specify topic, subplot index, and ylim range (YMIN YMAX). Use "default" for auto.')
    parser.add_argument('--interval', type=float, default=0.1, help='Update interval in seconds')
    return parser.parse_args()

def main():
    args = parse_args()
    rclpy.init()

    topic_layout = {}
    ylims = {}
    for topic_args in args.topic:
        name, idx_str, ymin_str, ymax_str, _ = topic_args
        idx = int(idx_str)
        topic_layout[name] = idx
        if idx not in ylims:
            if ymin_str == 'default' or ymax_str == 'default':
                ylims[idx] = ['default']
            else:
                ylims[idx] = [float(ymin_str), float(ymax_str)]

    plotter = CostPlotter(args.title, topic_layout, ylims, args.interval)
    plotter.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
