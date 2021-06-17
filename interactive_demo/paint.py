import argparse
import glob
import os
import random
import warnings

from PyQt5 import QtCore
from PyQt5.QtWidgets import *

from canvas import Canvas
from display_pad import DisplayPad
from hparams import *
from main_window import Ui_MainWindow

parser = argparse.ArgumentParser()
parser.add_argument('--use_cpu', action='store_true',
                    help='whether to use cpu to render images '
                         '(TVM model only supports GPU)')
parser.add_argument('--model', type=str, default='tvm', choices=['tvm', 'compressed', 'legacy', 'original'],
                    help='which model do you use [tvm | original | legacy | compressed]')
opt = parser.parse_args()
if opt.model == 'tvm' and opt.use_cpu:
    warnings.warn('TVM model only supports gpu')


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.canvas = Canvas()
        self.canvas.initialize()
        # We need to enable mouse tracking to follow the mouse without the button pressed.
        self.canvas.setMouseTracking(True)
        # Enable focus to capture key inputs.
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.horizontalLayout.addWidget(self.canvas, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.display_pad = DisplayPad(self.canvas, opt)
        self.horizontalLayout.addWidget(self.display_pad, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.canvas.set_display_pad(self.display_pad)
        # Setup the mode buttons
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)

        self.clearButton.released.connect(self.canvas.initialize)
        for mode in MODES:
            btn = getattr(self, '%sButton' % mode)
            print(btn)
            btn.pressed.connect(lambda mode=mode: self.canvas.set_mode(mode))
            mode_group.addButton(btn)

        # Setup up action signals
        self.actionCopy.triggered.connect(self.copy_to_clipboard)

        # Initialize animation timer.
        self.timer = QTimer()
        self.timer.timeout.connect(self.canvas.on_timer)
        self.timer.setInterval(100)
        self.timer.start()

        # Setup to agree with Canvas.
        self.set_primary_color('#000000')
        self.set_secondary_color('#ffffff')

        # Signals for canvas-initiated color changes (dropper).
        self.canvas.primary_color_updated.connect(self.set_primary_color)
        self.canvas.secondary_color_updated.connect(self.set_secondary_color)

        # Setup the stamp state.
        self.current_stamp_n = -1
        # self.next_stamp()
        # self.stampnextButton.pressed.connect(self.next_stamp)

        # Menu options
        self.actionNewImage.triggered.connect(self.canvas.initialize)
        self.actionOpenImage.triggered.connect(self.open_file)
        self.actionRandomSketch.triggered.connect(self.open_random)
        self.actionSaveImage.triggered.connect(self.save_file)
        self.actionSaveGenerated.triggered.connect(self.save_generated)
        self.actionClearImage.triggered.connect(self.canvas.reset)
        self.actionInvertColors.triggered.connect(self.invert)
        self.actionFlipHorizontal.triggered.connect(self.flip_horizontal)
        self.actionFlipVertical.triggered.connect(self.flip_vertical)

        # Setup the drawing toolbar.
        self.show()
        self.canvas.update()

    def choose_color(self, callback):
        dlg = QColorDialog()
        if dlg.exec():
            callback(dlg.selectedColor().name())

    def set_primary_color(self, hex):
        self.canvas.set_primary_color(hex)
        # self.primaryButton.setStyleSheet('QPushButton { background-color: %s; }' % hex)

    def set_secondary_color(self, hex):
        self.canvas.set_secondary_color(hex)
        # self.secondaryButton.setStyleSheet('QPushButton { background-color: %s; }' % hex)

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()

        if self.canvas.mode == 'selectrect' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectrect_copy())

        elif self.canvas.mode == 'selectpoly' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectpoly_copy())

        else:
            clipboard.setPixmap(self.canvas.pixmap())

    def open_random(self):
        choices = glob.glob("sketch/*.png")
        path = random.choice(choices)
        if os.path.exists(path):
            pixmap = QPixmap()
            pixmap.load(path)

            # We need to crop down to the size of our canvas. Get the size of the loaded image.
            iw = pixmap.width()
            ih = pixmap.height()

            # Get the size of the space we're filling.
            cw, ch = CANVAS_DIMENSIONS

            if iw / cw < ih / ch:  # The height is relatively bigger than the width.
                pixmap = pixmap.scaledToWidth(cw)
                hoff = (pixmap.height() - ch) // 2
                pixmap = pixmap.copy(
                    QRect(QPoint(0, hoff), QPoint(cw, pixmap.height() - hoff))
                )

            elif iw / cw > ih / ch:  # The height is relatively bigger than the width.
                pixmap = pixmap.scaledToHeight(ch)
                woff = (pixmap.width() - cw) // 2
                pixmap = pixmap.copy(
                    QRect(QPoint(woff, 0), QPoint(pixmap.width() - woff, ch))
                )

            self.canvas.setPixmap(pixmap)
            self.canvas.update()

    def open_file(self):
        """
        Open image file for editing, scaling the smaller dimension and cropping the remainder.
        :return:
        """
        path, _ = QFileDialog.getOpenFileName(self, "Open file", "",
                                              "PNG image files (*.png); JPEG image files (*jpg); All files (*.*)")

        if path:
            pixmap = QPixmap()
            pixmap.load(path)

            # We need to crop down to the size of our canvas. Get the size of the loaded image.
            iw = pixmap.width()
            ih = pixmap.height()

            # Get the size of the space we're filling.
            cw, ch = CANVAS_DIMENSIONS

            if iw / cw < ih / ch:  # The height is relatively bigger than the width.
                pixmap = pixmap.scaledToWidth(cw)
                hoff = (pixmap.height() - ch) // 2
                pixmap = pixmap.copy(
                    QRect(QPoint(0, hoff), QPoint(cw, pixmap.height() - hoff))
                )

            elif iw / cw > ih / ch:  # The height is relatively bigger than the width.
                pixmap = pixmap.scaledToHeight(ch)
                woff = (pixmap.width() - cw) // 2
                pixmap = pixmap.copy(
                    QRect(QPoint(woff, 0), QPoint(pixmap.width() - woff, ch))
                )

            self.canvas.setPixmap(pixmap)
            self.canvas.update()

    def save_file(self):
        """
        Save active canvas to image file.
        :return:
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save sketch file", "", "PNG Image file (*.png)")

        if path:
            pixmap = self.canvas.pixmap()
            pixmap.save(path, "PNG")

    def save_generated(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save generated file", "", "PNG Image file (*.png)")

        if path:
            pixmap = self.display_pad.pixmap()
            pixmap.save(path, "PNG")

    def invert(self):
        img = QImage(self.canvas.pixmap())
        img.invertPixels()
        pixmap = QPixmap()
        pixmap.convertFromImage(img)
        self.canvas.setPixmap(pixmap)

    def flip_horizontal(self):
        pixmap = self.canvas.pixmap()
        self.canvas.setPixmap(pixmap.transformed(QTransform().scale(-1, 1)))

    def flip_vertical(self):
        pixmap = self.canvas.pixmap()
        self.canvas.setPixmap(pixmap.transformed(QTransform().scale(1, -1)))


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    if DART_THEME:
        file = QFile("qdark.stylesheet")
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())
    app.exec_()
