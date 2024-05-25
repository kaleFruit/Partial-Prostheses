from lib import QObject, pyqtSignal, QRunnable


class Progress(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal()


class WorkerWrapper(QRunnable):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.run()
