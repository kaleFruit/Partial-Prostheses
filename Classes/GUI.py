from lib import vtk, Qt, QVTKRenderWindowInteractor, QtCore, QtWidgets, np, QThreadPool
from handManipulator import HandManipulator
from handMesh import HandMesh
from styles import Styles
from Progress import WorkerWrapper


class NoRotateStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self):
        self.AddObserver("LeftButtonPressEvent", self.customLeftButtonPressEvent)
        self.AddObserver("LeftButtonReleaseEvent", self.customLeftButtonReleaseEvent)
        self.AddObserver("MouseMoveEvent", self.customMouseMoveEvent)
        self.AddObserver("RightButtonPressEvent", self.dummy)
        self.AddObserver("MiddleButtonPressEvent", self.customMiddleButtonPressEvent)
        self.AddObserver(
            "MiddleButtonReleaseEvent", self.customMiddleButtonReleaseEvent
        )
        self.AddObserver("MouseWheelForwardEvent", self.customMouseWheelForwardEvent)
        self.AddObserver("MouseWheelBackwardEvent", self.customMouseWheelBackwardEvent)
        self.isRotating = False
        self.isTranslating = False

    def customLeftButtonPressEvent(self, obj, event):
        # Only start rotation when shift is held down
        shift_pressed = self.GetInteractor().GetShiftKey()
        if shift_pressed:
            self.isRotating = True
            self.StartRotate()

    def customLeftButtonReleaseEvent(self, obj, event):
        self.isRotating = False
        self.EndRotate()

    def customMouseMoveEvent(self, obj, event):
        shift_pressed = self.GetInteractor().GetShiftKey()
        if self.isRotating and shift_pressed:
            self.RotateAction()
        elif self.isTranslating and shift_pressed:
            self.TranslateAction()
        elif not shift_pressed:
            # If you want other behaviors when shift isn't pressed, implement here
            pass

    def RotateAction(self):
        rwi = self.GetInteractor()
        self.FindPokedRenderer(rwi.GetEventPosition()[0], rwi.GetEventPosition()[1])
        super().Rotate()  # call the superclass' Rotate method
        rwi.Render()

    def TranslateAction(self):
        rwi = self.GetInteractor()
        self.FindPokedRenderer(rwi.GetEventPosition()[0], rwi.GetEventPosition()[1])
        super().Pan()  # call the superclass' Pan method (for translation)
        rwi.Render()

    def customMiddleButtonPressEvent(self, obj, event):
        # Only start translation when shift is held down
        shift_pressed = self.GetInteractor().GetShiftKey()
        if shift_pressed:
            self.isTranslating = True
            self.StartPan()

    def customMiddleButtonReleaseEvent(self, obj, event):
        self.isTranslating = False

    def customMouseWheelForwardEvent(self, obj, event):
        shift_pressed = self.GetInteractor().GetShiftKey()
        if shift_pressed:
            self.OnMouseWheelForward()

    def customMouseWheelBackwardEvent(self, obj, event):
        shift_pressed = self.GetInteractor().GetShiftKey()
        if shift_pressed:
            self.OnMouseWheelBackward()

    def dummy(self, obj, event):
        pass


class GUI(Qt.QMainWindow):
    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)

        app = Qt.QApplication.instance()
        styleSheet = Styles()
        app.setStyleSheet(styleSheet.getStyles())

        mainLayout = Qt.QHBoxLayout()
        meshBox = Qt.QVBoxLayout()
        self.pbar = QtWidgets.QProgressBar(self)
        self.meshVis = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.meshVis.GetRenderWindow().AddRenderer(self.renderer)
        self.renderWindowInteractor = self.meshVis.GetRenderWindow().GetInteractor()
        noRotateStyle = NoRotateStyle()
        self.renderWindowInteractor.SetInteractorStyle(noRotateStyle)
        self.renderer.ResetCamera()
        meshBox.addWidget(self.pbar)
        meshBox.addWidget(self.meshVis)
        mainLayout.addLayout(meshBox)

        self.renderWindowInteractor.AddObserver("LeftButtonPressEvent", self.on_click)
        self.renderWindowInteractor.AddObserver("MouseMoveEvent", self.on_drag)
        self.renderWindowInteractor.AddObserver(
            "LeftButtonReleaseEvent", self.on_release
        )

        self.handMesh = HandMesh(
            self.renderWindowInteractor, self.renderer, self.meshVis.GetRenderWindow()
        )
        self.handMesh.signals.progress.connect(self.updateProgressBar)
        self.handMesh.signals.finished.connect(self.task_finished)
        self.threadpool = QThreadPool()

        self.handManipulator = HandManipulator(
            self.renderWindowInteractor,
            self.renderer,
            self.meshVis.GetRenderWindow(),
            self.handMesh.getHandMesh(),
        )
        self.handManipulator.signals.progress.connect(self.updateProgressBar)
        self.handManipulator.signals.finished.connect(self.task_finished)

        self.socketMode = False

        self.initUI(mainLayout)

        centralWidget = Qt.QWidget(objectName="totalBackground")
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.wristPlane = None

        self.show()
        self.renderWindowInteractor.Initialize()

    def closeEvent(self, event):
        self.meshVis.Finalize()
        super().closeEvent(event)

    def on_click(self, obj, event):
        if self.socketMode:
            self.handMesh.on_click(obj, event)
        else:
            self.handManipulator.on_click(obj, event)

    def on_drag(self, obj, event):
        if self.socketMode:
            self.handMesh.on_drag(obj, event)
        else:
            self.handManipulator.on_drag(obj, event)

    def on_release(self, obj, event):
        if self.socketMode:
            self.handMesh.on_release(obj, event)
        else:
            self.handManipulator.on_release(obj, event)

    def initUI(self, mainLayout):
        tabs = Qt.QTabWidget()
        tabs.addTab(self.fingerTabUI(), "Finger Generator")
        tabs.addTab(self.socketTabUI(), "Socket Generator")
        tabs.setFixedWidth(600)
        mainLayout.addWidget(tabs)

    def socketTabUI(self):
        tab = Qt.QWidget()
        outerBox = Qt.QVBoxLayout()

        vbox = Qt.QVBoxLayout()

        header = Qt.QLabel("Socket Tools:")
        header.setAlignment(QtCore.Qt.AlignCenter)
        header.setStyleSheet("QLabel { font-size: 20px;}")
        vbox.addWidget(header)
        cb = QtWidgets.QCheckBox("Enable/Disable")
        vbox.addWidget(cb)
        cb.stateChanged.connect(self.toggleSocket)

        rbPaint = QtWidgets.QRadioButton("Paint brush")
        rbPaint.setGeometry(QtCore.QRect(180, 120, 95, 20))
        rbPaint.setChecked(True)
        vbox.addWidget(rbPaint)
        rbPaint.toggled.connect(lambda: self.socketToolSelected(0))
        rbEraser = QtWidgets.QRadioButton("Eraser")
        rbEraser.setGeometry(QtCore.QRect(180, 150, 95, 20))
        vbox.addWidget(rbEraser)
        rbEraser.toggled.connect(lambda: self.socketToolSelected(1))

        label = Qt.QLabel("Change Brush Size:")
        vbox.addWidget(label)
        brushSize = QtWidgets.QSlider()
        brushSize.setGeometry(QtCore.QRect(190, 100, 160, 16))
        brushSize.setOrientation(QtCore.Qt.Horizontal)
        brushSize.sliderMoved.connect(self.brushSizer)
        vbox.addWidget(brushSize)

        label = Qt.QLabel("Change Paint Density:")
        vbox.addWidget(label)
        density = QtWidgets.QSlider()
        density.setGeometry(QtCore.QRect(190, 100, 160, 16))
        density.setOrientation(QtCore.Qt.Horizontal)
        density.sliderMoved.connect(self.densityChanger)
        vbox.addWidget(density)

        label = Qt.QLabel("Change Hard Socket Clip Height:")
        vbox.addWidget(label)
        plane = QtWidgets.QSlider()
        plane.setGeometry(QtCore.QRect(190, 100, 160, 16))
        plane.setValue(55)
        plane.setOrientation(QtCore.Qt.Horizontal)
        plane.sliderMoved.connect(self.planeHeight)
        vbox.addWidget(plane)

        outerBox.addLayout(vbox)

        outerBox.addItem(
            QtWidgets.QSpacerItem(
                0, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )
        hLine = QtWidgets.QFrame()
        hLine.setFrameShape(QtWidgets.QFrame.HLine)
        hLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        outerBox.addWidget(hLine)
        outerBox.addItem(
            QtWidgets.QSpacerItem(
                0, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )

        buttonBox = Qt.QVBoxLayout()
        buttonBox.setAlignment(QtCore.Qt.AlignCenter)
        genSocketButton = QtWidgets.QPushButton("Generate Soft Socket")
        buttonBox.addWidget(genSocketButton)
        genSocketButton.clicked.connect(self.generateSocket)
        testSocketButton = QtWidgets.QPushButton("Generate Hardshell of Socket")
        buttonBox.addWidget(testSocketButton)
        testSocketButton.clicked.connect(self.generateHardSocket)

        outerBox.addLayout(buttonBox)

        tab.setLayout(outerBox)

        scroll = Qt.QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(600)

        return scroll

    def fingerTabUI(self):
        tab = Qt.QWidget()

        self.sliders = []
        outerBox = Qt.QVBoxLayout()

        splitBox = Qt.QVBoxLayout()

        fingerActivationBox = Qt.QVBoxLayout()
        leftHeader = Qt.QLabel("Finger/Joint Selection")
        leftHeader.setAlignment(QtCore.Qt.AlignCenter)
        leftHeader.setStyleSheet("QLabel { font-size: 20px;}")
        fingerActivationBox.addWidget(leftHeader)

        fingerSubBoxes = Qt.QHBoxLayout()
        labelNames = ["Index", "Middle", "Third", "Fourth"]

        for i in range(4):
            box = Qt.QVBoxLayout()
            label = Qt.QLabel(f"{labelNames[i]} Finger")
            box.addWidget(label)
            h_layout = Qt.QHBoxLayout()
            box.addLayout(h_layout)
            slider = QtWidgets.QSlider(objectName="fingerSelectorSlider")
            slider.setOrientation(QtCore.Qt.Vertical)
            slider.setMinimum(1)
            slider.setMaximum(4)
            slider.setSingleStep(1)
            slider.setTickInterval(1)
            slider.setMinimumHeight(100)
            slider.setValue(0)
            slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
            slider.valueChanged.connect(self.toggleJointInteraction)
            self.sliders.append(slider)
            h_layout.addWidget(slider)

            frame = Qt.QFrame(objectName="fingerJointSelector")
            frame.setFrameShape(Qt.QFrame.StyledPanel)
            # frame.setStyleSheet(
            #     f"background-color: rgb(200, {200+i*10}, 200); border-radius: 10px; "
            # )
            frame.setLayout(box)
            fingerSubBoxes.addWidget(frame)
        fingerActivationBox.addLayout(fingerSubBoxes)

        fingerSettingsBox = Qt.QVBoxLayout()

        rightHeader = Qt.QLabel("Finger Settings")
        rightHeader.setAlignment(QtCore.Qt.AlignCenter)
        rightHeader.setStyleSheet("QLabel { font-size: 20px;}")
        fingerSettingsBox.addWidget(rightHeader)

        cb = QtWidgets.QCheckBox("Add Elastic String")
        cb.stateChanged.connect(self.addElastic)
        cb.setChecked(True)
        fingerSettingsBox.addWidget(cb)

        fingerSettingsBox.addWidget(Qt.QLabel("Set Wrist Actuation Rotation:"))
        wristRotation = QtWidgets.QLineEdit()
        wristRotation.setValidator(Qt.QIntValidator(0, 90))
        wristRotation.setText("90")
        wristRotation.textChanged.connect(self.setWristRotation)
        fingerSettingsBox.addWidget(wristRotation)

        fingerSettingsBox.addWidget(Qt.QLabel("Set Finger Width:"))
        fingerWidth = Qt.QDoubleSpinBox()
        fingerWidth.setDecimals(1)
        fingerWidth.setSingleStep(0.1)
        fingerWidth.setValue(12)
        fingerWidth.textChanged.connect(self.setFingerWidth)
        fingerSettingsBox.addWidget(fingerWidth)

        fingerSettingsBox.addWidget(Qt.QLabel("Set Finger Height:"))
        fingerHeight = Qt.QDoubleSpinBox()
        fingerHeight.setDecimals(1)
        fingerHeight.setSingleStep(0.1)
        fingerHeight.setValue(12)
        fingerHeight.textChanged.connect(self.setFingerHeight)
        fingerSettingsBox.addWidget(fingerHeight)

        fingerSettingsBox.addWidget(Qt.QLabel("Set Connector Thickness:"))
        connectorThickness = Qt.QDoubleSpinBox()
        connectorThickness.setDecimals(1)
        connectorThickness.setSingleStep(0.1)
        connectorThickness.setValue(12)
        connectorThickness.textChanged.connect(self.setConnectorThickness)
        fingerSettingsBox.addWidget(connectorThickness)

        fingerSettingsBox.addWidget(Qt.QLabel("Set Connector End Radius:"))
        connectorRadius = Qt.QDoubleSpinBox()
        connectorRadius.setDecimals(1)
        connectorRadius.setSingleStep(0.1)
        connectorRadius.setValue(12)
        connectorRadius.textChanged.connect(self.setConnectorRadius)
        fingerSettingsBox.addWidget(connectorRadius)
        fingerSettingsBox.addStretch()

        splitBox.addLayout(fingerActivationBox, 1)
        splitBox.addItem(
            QtWidgets.QSpacerItem(
                0, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )
        hLine = QtWidgets.QFrame()
        hLine.setFrameShape(QtWidgets.QFrame.HLine)
        hLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        splitBox.addWidget(hLine)
        splitBox.addItem(
            QtWidgets.QSpacerItem(
                0, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )
        splitBox.addLayout(fingerSettingsBox, 1)

        outerBox.addLayout(splitBox)

        buttonBox = Qt.QVBoxLayout()
        buttonBox.setAlignment(QtCore.Qt.AlignCenter)
        genFingers = QtWidgets.QPushButton("Generate Fingers")
        buttonBox.addWidget(genFingers)
        genFingers.clicked.connect(self.genFingers)

        outerBox.addItem(
            QtWidgets.QSpacerItem(
                0, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )
        hLine = QtWidgets.QFrame()
        hLine.setFrameShape(QtWidgets.QFrame.HLine)
        hLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        outerBox.addWidget(hLine)
        outerBox.addItem(
            QtWidgets.QSpacerItem(
                0, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
        )

        tensionerPlate = QtWidgets.QPushButton("Generate String Tensioning Plate")
        buttonBox.addWidget(tensionerPlate)
        tensionerPlate.clicked.connect(self.tensionerPlate)

        saveFingerPos = QtWidgets.QPushButton("Save Finger Positions")
        buttonBox.addWidget(saveFingerPos)
        saveFingerPos.clicked.connect(self.saveFingerPositions)

        outerBox.addLayout(buttonBox)
        tab.setLayout(outerBox)
        scroll2 = Qt.QScrollArea()
        scroll2.setWidget(tab)
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(950)

        return scroll2

    def socketToolSelected(self, state):
        self.handMesh.setTool(state)

    def generateSocket(self):
        self.handMesh.process = "generateSocket"
        self.handMesh.processArgs = [self.wristPlane]
        if not self.handMesh._is_finished:
            self.handMesh.stop()
            self.threadpool.waitForDone()
        self.handMesh.reset()
        new_worker = WorkerWrapper(self.handMesh)
        self.threadpool.start(new_worker)

    def genFingers(self):
        self.handManipulator.process = "generateFingers"
        if not self.handManipulator._is_finished:
            self.handManipulator.stop()
            self.threadpool.waitForDone()
        self.handManipulator.reset()
        new_worker = WorkerWrapper(self.handManipulator)
        self.threadpool.start(new_worker)

    def tensionerPlate(self):
        self.handMesh.wristBand, self.handMesh.slots, self.wristPlane = (
            self.handManipulator.createWristBand(self.handMesh.getSocketThickness())
        )

    def saveFingerPositions(self):
        self.handManipulator.saveFingerPositions()

    def densityChanger(self, p):
        self.handMesh.densityChange(p)

    def brushSizer(self, p):
        self.handMesh.brushSizer(p)

    def planeHeight(self, p):
        self.handMesh.changePlaneHeight(p)

    def addElastic(self, state):
        self.handManipulator.elastic = state
        self.handMesh.elasticString = state

    def toggleSocket(self, state):
        self.handMesh.socketMode(state)
        self.socketMode = state

    def toggleJointInteraction(self, jointIDx):
        sender = self.sender()
        fingerIndex = self.sliders.index(sender)
        self.handManipulator.enableJoints(fingerIndex, jointIDx)

    def generateHardSocket(self):
        writer = vtk.vtkSTLWriter()
        for i, finger in enumerate(self.handManipulator.getFingerMeshes()):
            writer.SetFileName(f"toBePrinted/fingerStruct{i}.stl")
            writer.SetInputData(finger)
            writer.Write()

        appender = vtk.vtkAppendPolyData()
        for connector in self.handManipulator.getConnectors():
            appender.AddInputData(connector)
        appender.Update()
        writer.SetFileName("toBePrinted/connectorStruct.stl")
        writer.SetInputData(appender.GetOutput())
        writer.Write()

        writer.SetFileName("toBePrinted/handMesh.stl")
        writer.SetInputData(self.handMesh.getHand())
        writer.Write()

        args = (
            self.handManipulator.getJoints(),
            self.handManipulator.getFingerMeshes(),
            self.handManipulator.getFingerInfo(),
        )
        self.handMesh.process = "genHardSocket"
        self.handMesh.processArgs = args
        if not self.handMesh._is_finished:
            self.handMesh.stop()
            self.threadpool.waitForDone()
        self.handMesh.reset()
        new_worker = WorkerWrapper(self.handMesh)
        self.threadpool.start(new_worker)

        # self.handMesh.genHardSocket(
        #     self.handManipulator.getJoints(),
        #     self.handManipulator.getFingerMeshes(),
        #     self.handManipulator.getFingerInfo(),
        # )

    def setFingerWidth(self, val):
        self.handManipulator.setFingerWidth(float(val))

    def setFingerHeight(self, val):
        self.handManipulator.setFingerHeight(float(val))

    def setConnectorThickness(self, val):
        self.handManipulator.setConnectorThickness(float(val))

    def setConnectorRadius(self, val):
        self.handManipulator.setConnectorRadius(float(val))

    def setWristRotation(self, val):
        if val:
            self.handManipulator.setWristRotation(int(val) / 180 * np.pi)

    def updateProgressBar(self, value):
        self.pbar.setValue(value)
        self.pbar.setFormat(f"Generating: {value}%")

    def task_finished(self):
        self.pbar.setFormat(f"Generation Complete")
