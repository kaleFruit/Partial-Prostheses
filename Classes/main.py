from GUI import GUI
from lib import sys, Qt

app = Qt.QApplication(sys.argv)
window = GUI()
sys.exit(app.exec_())
