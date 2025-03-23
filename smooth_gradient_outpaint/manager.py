from smooth_gradient_outpaint.outpainter import Outpainter


class OutpaintManager:
    def __init__(self):
        self._outpainter = Outpainter()

    def _run(self, data):
        return self._outpainter.paint(data, self._config)
