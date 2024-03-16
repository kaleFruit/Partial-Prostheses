import numpy as np
import vtk
import pandas as pd
import pymeshlab as ml
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys
import open3d as o3d
from vtkbool.vtkBool import vtkPolyDataBooleanFilter
from plyfile import PlyData
import random
import pyvista as pv
from tqdm import tqdm
from sklearn.manifold import LocallyLinearEmbedding
from scipy.spatial import Voronoi, cKDTree, ConvexHull
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkFiltersCore import vtkAppendPolyData
import time


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


class cylinderBone:
    def __init__(self, startPoint: list, endPoint: list):
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.startSphere = None
        self.endSphere = None
        self.actor, self.transformPDCyl = self.initCylinder()

    def setStartSphere(self, startSphere):
        self.startSphere = startSphere

    def setEndSphere(self, endSphere):
        self.endSphere = endSphere

    def move(self, sphere):
        if sphere == self.startSphere:
            self.startPoint = sphere.center
        else:
            self.endPoint = sphere.center
        normalizedX = [0] * 3
        normalizedZ = [0] * 3
        normalizedY = [0] * 3
        vtk.vtkMath.Subtract(self.endPoint, self.startPoint, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)
        arbitrary = [1, 1, 1]
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])
        transform = vtk.vtkTransform()
        transform.Translate(self.startPoint)
        transform.Concatenate(matrix)
        transform.RotateZ(-90.0)
        transform.Scale(1.0, length, 1.0)
        transform.Translate(0, 0.5, 0)
        self.transformPDCyl.SetTransform(transform)
        self.transformPDCyl.Update()

    def initCylinder(self):
        cylinderSource = vtk.vtkCylinderSource()
        cylinderSource.SetResolution(15)
        cylinderSource.SetRadius(2)
        # Compute a basis
        normalizedX = [0] * 3
        normalizedY = [0] * 3
        normalizedZ = [0] * 3
        vtk.vtkMath.Subtract(self.endPoint, self.startPoint, normalizedX)
        length = vtk.vtkMath.Norm(normalizedX)
        vtk.vtkMath.Normalize(normalizedX)
        arbitrary = [1, 1, 1]
        vtk.vtkMath.Cross(normalizedX, arbitrary, normalizedZ)
        vtk.vtkMath.Normalize(normalizedZ)
        vtk.vtkMath.Cross(normalizedZ, normalizedX, normalizedY)
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(0, 3):
            matrix.SetElement(i, 0, normalizedX[i])
            matrix.SetElement(i, 1, normalizedY[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        # Apply the transforms
        transform = vtk.vtkTransform()
        transform.Translate(self.startPoint)  # translate to starting point
        transform.Concatenate(matrix)  # apply direction cosines
        transform.RotateZ(-90.0)  # align cylinder to x axis
        transform.Scale(1.0, length, 1.0)  # scale along the height vector
        transform.Translate(0, 0.5, 0)  # translate to start of cylinder

        transformPD = vtk.vtkTransformPolyDataFilter()
        transformPD.SetTransform(transform)
        transformPD.SetInputConnection(cylinderSource.GetOutputPort())

        mapper = vtk.vtkPolyDataMapper()
        cylinder = vtk.vtkActor()
        mapper.SetInputConnection(transformPD.GetOutputPort())
        cylinder.SetMapper(mapper)
        cylinder.type = "cylinder"
        return [cylinder, transformPD]


class sphereBone:
    def __init__(self, center: list, newName: str):
        self.center = center
        self.idTag = 10
        self.actor, self.sphereStartSource = self.initStartSphere()
        self.name = newName
        self.toggled = True

    def setIdTag(self, idTagNew: int):
        self.idTag = idTagNew
        self.actor.idTag = self.idTag

    def move(self, newPoint: list):
        self.sphereStartSource.SetCenter(newPoint)
        self.sphereStartSource.Update()
        self.center = newPoint

    def initStartSphere(self):
        sphereStartSource = vtk.vtkSphereSource()
        sphereStartSource.SetCenter(self.center)
        sphereStartSource.SetRadius(5)
        sphereStartMapper = vtk.vtkPolyDataMapper()
        sphereStartMapper.SetInputConnection(sphereStartSource.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(sphereStartMapper)
        actor.type = "sphere"
        return [actor, sphereStartSource]


class GUI(Qt.QMainWindow):
    def __init__(self, parent=None):
        Qt.QMainWindow.__init__(self, parent)

        app = Qt.QApplication.instance()
        app.setStyleSheet(
            "QLabel { font-size: 14px; } "
            "QPushButton {"
            "border-radius: 10px; "
            "border: 1px solid #339955;"
            "width: 200px;"
            "height: 50px;"
            "}"
        )

        mainLayout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderWindowInteractor = self.vtkWidget.GetRenderWindow().GetInteractor()
        noRotateStyle = NoRotateStyle()
        self.renderWindowInteractor.SetInteractorStyle(noRotateStyle)
        self.renderer.ResetCamera()
        mainLayout.addWidget(self.vtkWidget)

        self.renderWindowInteractor.AddObserver("LeftButtonPressEvent", self.on_click)
        self.renderWindowInteractor.AddObserver("MouseMoveEvent", self.on_drag)
        self.renderWindowInteractor.AddObserver(
            "LeftButtonReleaseEvent", self.on_release
        )

        self.handManipulator = HandManipulator(
            self.renderWindowInteractor, self.renderer, self.vtkWidget.GetRenderWindow()
        )

        self.handMesh = HandMesh(
            self.renderWindowInteractor, self.renderer, self.vtkWidget.GetRenderWindow()
        )
        self.socketMode = False

        self.initUI(mainLayout)

        centralWidget = Qt.QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.renderWindowInteractor.Initialize()
        self.show()

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
        testSocketButton.clicked.connect(self.testSocket)

        outerBox.addLayout(buttonBox)

        tab.setLayout(outerBox)

        scroll = Qt.QScrollArea()
        scroll.setWidget(tab)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(500)

        return scroll

    def fingerTabUI(self):
        tab = Qt.QWidget()

        self.checkboxes = {}
        outerBox = Qt.QVBoxLayout()

        splitBox = Qt.QVBoxLayout()

        fingerActivationBox = Qt.QVBoxLayout()
        leftHeader = Qt.QLabel("Finger/Joint Selection")
        leftHeader.setAlignment(QtCore.Qt.AlignCenter)
        leftHeader.setStyleSheet("QLabel { font-size: 20px;}")
        fingerActivationBox.addWidget(leftHeader)

        fingerSubBoxes = Qt.QHBoxLayout()
        labelNames = ["Index", "Middle", "Third", "Fourth"]
        fingerNames = [
            self.handManipulator.indexJoints.keys(),
            self.handManipulator.middleJoints.keys(),
            self.handManipulator.thirdJoints.keys(),
            self.handManipulator.fourthJoints.keys(),
        ]
        for i in range(4):
            box = Qt.QVBoxLayout()
            label = Qt.QLabel(f"{labelNames[i]} Finger")
            box.addWidget(label)
            for j, sphere in enumerate(fingerNames[i]):
                cb = QtWidgets.QCheckBox(sphere.name)
                box.addWidget(cb)
                cb.stateChanged.connect(self.toggleJointInteraction)
                self.checkboxes[cb] = sphere
            frame = Qt.QFrame()
            frame.setFrameShape(Qt.QFrame.StyledPanel)
            frame.setStyleSheet(
                f"background-color: rgb(200, {200+i*10}, 200); border-radius: 10px; "
            )
            frame.setLayout(box)
            fingerSubBoxes.addWidget(frame)
        fingerActivationBox.addLayout(fingerSubBoxes)

        fingerSettingsBox = Qt.QVBoxLayout()
        rightHeader = Qt.QLabel("Finger Settings")
        rightHeader.setAlignment(QtCore.Qt.AlignCenter)
        rightHeader.setStyleSheet("QLabel { font-size: 20px;}")
        fingerSettingsBox.addWidget(rightHeader)

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

        saveFingerPos = QtWidgets.QPushButton("Save Finger Positions")
        buttonBox.addWidget(saveFingerPos)
        saveFingerPos.clicked.connect(self.saveFingerPositions)

        outerBox.addLayout(buttonBox)
        tab.setLayout(outerBox)
        scroll2 = Qt.QScrollArea()
        scroll2.setWidget(tab)
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(500)

        return scroll2

    def socketToolSelected(self, state):
        self.handMesh.setTool(state)

    def generateSocket(self):
        self.handMesh.generateSocket()

    def genFingers(self):
        self.handManipulator.generateFingers()

    def saveFingerPositions(self):
        self.handManipulator.saveFingerPositions()

    def densityChanger(self, p):
        self.handMesh.densityChange(p)

    def brushSizer(self, p):
        self.handMesh.brushSizer(p)

    def toggleSocket(self, state):
        self.handMesh.socketMode(state)
        self.socketMode = state

    def toggleJointInteraction(self, state):
        sender = self.sender()
        joint = self.checkboxes[sender]
        if state == QtCore.Qt.Checked:
            joint.actor.PickableOff()
            joint.toggled = False
        else:
            joint.actor.PickableOn()
            joint.toggled = True

    def testSocket(self):
        writer = vtk.vtkSTLWriter()
        for i, finger in enumerate(self.handManipulator.getFingerMeshes()):
            writer.SetFileName(f"imageAnalysisGeneration/fingerStruct{i}.stl")
            writer.SetInputData(finger)
            writer.Write()

        appender = vtk.vtkAppendPolyData()
        for connector in self.handManipulator.getConnectors():
            appender.AddInputData(connector)
        appender.Update()
        writer.SetFileName("imageAnalysisGeneration/connectorStruct.stl")
        writer.SetInputData(appender.GetOutput())
        writer.Write()

        writer.SetFileName("imageAnalysisGeneration/handMesh.stl")
        writer.SetInputData(self.handMesh.getHand())
        writer.Write()

        self.handMesh.genHandPortion(
            self.handManipulator.getJoints(),
            self.handManipulator.getFingerMeshes(),
            self.handManipulator.getFingerInfo(),
        )

    def setFingerWidth(self, val):
        self.handManipulator.setFingerWidth(float(val))

    def setFingerHeight(self, val):
        self.handManipulator.setFingerHeight(float(val))

    def setConnectorThickness(self, val):
        self.handManipulator.setConnectorThickness(float(val))

    def setConnectorRadius(self, val):
        self.handManipulator.setConnectorRadius(float(val))


class HandMesh:
    def __init__(self, renderWinIn, ren, renWin):
        self.renderWindowInteractor = renderWinIn
        self.renderer = ren
        self.renderWindow = renWin
        self.maxHoleRadius = 6
        self.flexibility = self.maxHoleRadius
        self.tool = self.maxHoleRadius

        self.colors = vtk.vtkNamedColors()
        self.scalars = vtk.vtkFloatArray()
        self.actor = self.genHandView("stlfiles\partialHand1.ply")
        self.is_painting = False

        self.brush_radius = 10.0
        self.enclosed_points = vtk.vtkSelectEnclosedPoints()
        self.socketIds = [0] * self.actor.GetMapper().GetInput().GetNumberOfCells()
        self.holesInCells = [0] * self.actor.GetMapper().GetInput().GetNumberOfCells()
        self.thickness = 2

        self.fromPreviousDesign = True
        self.initTime = 0

        self.finalSocket = None

    def genHandView(self, fileName: str):
        reader = vtk.vtkPLYReader()
        reader.SetFileName(fileName)
        reader.Update()

        plydata = PlyData.read(fileName)
        normals_data = plydata["vertex"]
        normals = list(zip(normals_data["nx"], normals_data["ny"], normals_data["nz"]))
        vtk_normals = vtk.vtkDoubleArray()
        vtk_normals.SetNumberOfComponents(3)
        vtk_normals.SetName("normals")
        for normal in normals:
            norm = (normal[0], normal[1], normal[2])
            vtk_normals.InsertNextTuple(norm)

        reader.Update()
        poly = reader.GetOutput()
        poly.GetPointData().SetNormals(vtk_normals)

        reverseNormals = vtk.vtkReverseSense()
        reverseNormals.SetInputData(poly)
        reverseNormals.ReverseCellsOn()
        reverseNormals.ReverseNormalsOn()
        reverseNormals.Update()
        poly = reverseNormals.GetOutput()

        status, message = self.check_constraints(poly)
        print(message)

        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(poly)
        featureEdges.BoundaryEdgesOff()
        featureEdges.FeatureEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOn()
        featureEdges.Update()

        # Map the extracted edges to graphics primitives
        edgeMapper = vtk.vtkPolyDataMapper()
        edgeMapper.SetInputConnection(featureEdges.GetOutputPort())

        # Create an actor for the edges
        edgeActor = vtk.vtkActor()
        edgeActor.SetMapper(edgeMapper)
        edgeActor.GetProperty().SetColor(0, 1, 0)
        self.renderer.AddActor(edgeActor)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        reader.Update()
        num_cells = poly.GetNumberOfCells()
        for _ in range(num_cells):
            self.scalars.InsertNextValue(0.0)
        mapper.GetInput().GetCellData().SetScalars(self.scalars)
        actor.SetMapper(mapper)

        lookupTable = vtk.vtkLookupTable()
        lookupTable.SetNumberOfTableValues(256)
        for i in range(256):
            r = 1.0 - i / 255.0
            g = 1.0 - i / 255.0
            b = 1.0
            lookupTable.SetTableValue(i, r, g, b, 1.0)
        mapper.SetLookupTable(lookupTable)
        mapper.SetScalarRange(0, self.maxHoleRadius)

        actor.GetProperty().SetRepresentationToWireframe()
        actor.PickableOff()
        actor.GetProperty().SetOpacity(0.1)
        self.renderer.AddActor(actor)

        return actor

    def getHand(self):
        return self.actor.GetMapper().GetInput()

    def check_constraints(self, polydata):
        """Check if the mesh meets the constraints of a closed surface without boundary or non-manifold edges."""
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(polydata)
        featureEdges.BoundaryEdgesOn()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOn()
        featureEdges.FeatureEdgesOff()
        featureEdges.Update()

        edge_data = featureEdges.GetOutput()
        num_boundary_edges = edge_data.GetNumberOfCells()

        featureEdges.BoundaryEdgesOff()
        featureEdges.NonManifoldEdgesOn()
        featureEdges.Update()

        non_manifold_data = featureEdges.GetOutput()
        num_non_manifold_edges = non_manifold_data.GetNumberOfCells()

        is_closed_surface = num_boundary_edges == 0
        has_no_non_manifold_edges = num_non_manifold_edges == 0

        if is_closed_surface and has_no_non_manifold_edges:
            return True, "Mesh meets all constraints."
        else:
            message = "Issues found: "
            if not is_closed_surface:
                message += f"{num_boundary_edges} boundary edges. "
            if not has_no_non_manifold_edges:
                message += f"{num_non_manifold_edges} non-manifold edges."
            return False, message

    def setTool(self, tool):
        if tool == 0:
            self.tool = self.flexibility
        elif tool == 1:
            self.tool = 0

    def socketMode(self, state):
        if state:
            self.actor.PickableOn()
            self.actor.GetProperty().SetRepresentationToSurface()
            self.actor.GetProperty().SetOpacity(1)
        else:
            self.actor.PickableOff()
            self.actor.GetProperty().SetRepresentationToWireframe()
            self.actor.GetProperty().SetOpacity(0.1)
        self.renderWindow.Render()

    def resize(self, factor):
        scalar = np.log(factor + 3)
        self.actor.SetScale(scalar, scalar, scalar)
        self.renderWindow.Render()

    def brushSizer(self, factor):
        self.brush_radius = factor / 4

    def densityChange(self, factor):
        self.flexibility = factor / 100 * 6

    def on_click(self, obj, event):
        self.is_painting = True
        click_pos = self.renderWindowInteractor.GetEventPosition()
        self.paintCellsAroundPoint(click_pos[0], click_pos[1])

    def on_drag(self, obj, event):
        if not self.is_painting:
            return
        drag_pos = self.renderWindowInteractor.GetEventPosition()
        self.paintCellsAroundPoint(drag_pos[0], drag_pos[1])

    def on_release(self, obj, event):
        self.is_painting = False

    def paintCellsAroundPoint(self, x, y):
        picker = vtk.vtkCellPicker()
        picker.Pick(x, y, 0, self.renderer)
        world_pos = picker.GetPickPosition()
        transform = vtk.vtkTransform()
        transform.SetMatrix(self.actor.GetMatrix())
        transform.Inverse()
        world_pos = transform.TransformPoint(world_pos)

        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(world_pos)
        sphere.SetRadius(self.brush_radius)
        sphere.Update()

        self.enclosed_points.SetInputData(self.actor.GetMapper().GetInput())
        self.enclosed_points.SetSurfaceData(sphere.GetOutput())
        self.enclosed_points.Update()

        polydata = self.actor.GetMapper().GetInput()
        for i in range(polydata.GetNumberOfPoints()):
            if self.enclosed_points.IsInside(i):
                cellIds = vtk.vtkIdList()
                polydata.GetPointCells(i, cellIds)
                for j in range(cellIds.GetNumberOfIds()):
                    cellId = cellIds.GetId(j)
                    if self.tool == 0:
                        self.socketIds[cellId] = 0
                        self.scalars.SetTuple1(cellId, 0)
                    else:
                        self.socketIds[cellId] = self.flexibility
                        self.scalars.SetTuple1(cellId, self.flexibility)

        self.actor.GetMapper().GetInput().Modified()
        self.renderWindow.Render()

    def plotVectors(self, polydata, everyOther, h, s, v):
        pointsTest = vtk.vtkPoints()
        normalsTest = vtk.vtkFloatArray()
        normalsTest.SetNumberOfComponents(3)

        # Check if point normals are available
        if polydata.GetPointData().GetNormals():
            for i in range(polydata.GetNumberOfPoints()):
                if i % everyOther == 0:
                    pointsTest.InsertNextPoint(polydata.GetPoint(i))
                    normalsTest.InsertNextTuple(
                        polydata.GetPointData().GetNormals().GetTuple(i)
                    )
        # If point normals are not available, check for cell normals
        elif polydata.GetCellData().GetNormals():
            for i in range(polydata.GetNumberOfCells()):
                if i % everyOther == 0:
                    cell = polydata.GetCell(i)
                    centroid = [0, 0, 0]
                    for j in range(cell.GetNumberOfPoints()):
                        point = polydata.GetPoint(cell.GetPointId(j))
                        centroid[0] += point[0]
                        centroid[1] += point[1]
                        centroid[2] += point[2]
                    centroid = [c / cell.GetNumberOfPoints() for c in centroid]
                    pointsTest.InsertNextPoint(centroid)
                    normalsTest.InsertNextTuple(
                        polydata.GetCellData().GetNormals().GetTuple(i)
                    )
        else:
            print("No point or cell normals available in the provided polydata.")
            return

        arrows_polydata = vtk.vtkPolyData()
        arrows_polydata.SetPoints(pointsTest)
        arrows_polydata.GetPointData().SetNormals(normalsTest)

        arrow_source = vtk.vtkArrowSource()

        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrow_source.GetOutputPort())
        glyph.SetVectorModeToUseNormal()
        glyph.SetInputData(arrows_polydata)
        glyph.SetScaleFactor(0.5)
        glyph.OrientOn()
        glyph.Update()

        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyph.GetOutputPort())

        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(h, s, v)

        self.renderer.AddActor(glyph_actor)

    def generateSocket(self):
        self.initTime = time.time()

        socketShell = vtk.vtkPolyData()
        if self.fromPreviousDesign:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName("oldDesigns/1stgen.vtk")
            reader.Update()
            socketShell = reader.GetOutput()
        else:
            newPoints = vtk.vtkPoints()
            newCells = vtk.vtkCellArray()
            point_map = {}

            transform = vtk.vtkTransform()
            transform.SetMatrix(self.actor.GetMatrix())

            normals = vtk.vtkFloatArray()
            normals.SetNumberOfComponents(3)

            old_scalars = self.actor.GetMapper().GetInput().GetCellData().GetScalars()
            new_scalars = vtk.vtkFloatArray()
            new_scalars.SetNumberOfComponents(old_scalars.GetNumberOfComponents())
            new_scalars.SetName(old_scalars.GetName())

            new_cell_map = {}
            new_cell_counter = 0
            for cellId in range(len(self.socketIds)):
                if self.socketIds[cellId] > 0:
                    cell = self.actor.GetMapper().GetInput().GetCell(cellId)
                    pointIds = cell.GetPointIds()
                    new_pointIds = []

                    for i in range(pointIds.GetNumberOfIds()):
                        pointId = pointIds.GetId(i)
                        if pointId not in point_map:
                            x, y, z = (
                                self.actor.GetMapper().GetInput().GetPoint(pointId)
                            )
                            x, y, z = transform.TransformPoint(x, y, z)
                            new_id = newPoints.InsertNextPoint(x, y, z)

                            normal = (
                                self.actor.GetMapper()
                                .GetInput()
                                .GetPointData()
                                .GetNormals()
                                .GetTuple(pointId)
                            )
                            normals.InsertNextTuple([n for n in normal])

                            point_map[pointId] = new_id
                        new_pointIds.append(point_map[pointId])
                    newCells.InsertNextCell(len(new_pointIds), new_pointIds)
                    new_cell_map[cellId] = new_cell_counter
                    scalar_value = old_scalars.GetTuple(cellId)
                    new_scalars.InsertNextTuple(scalar_value)
                    new_cell_counter += 1

            socketShell.SetPoints(newPoints)
            socketShell.SetPolys(newCells)
            socketShell.GetPointData().SetNormals(normals)
            socketShell.GetCellData().SetScalars(new_scalars)
            socketShell = self.cleanSocketShell(socketShell)

            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName("oldDesigns/1stgen.vtk")
            writer.SetInputData(socketShell)
            writer.Write()

        finalSocket = self.extrusion(socketShell)
        self.finalSocket = finalSocket
        writer = vtk.vtkSTLWriter()
        writer.SetFileName("finalSocket.stl")
        writer.SetInputData(finalSocket)
        writer.Write()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(finalSocket)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.colors.GetColor3d("Red"))
        actor.PickableOff()
        actor.GetProperty().SetOpacity(1)
        # actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(actor)
        self.renderWindow.Render()

    def samplePoints(self, polydata, radius, bdpts):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName("inputSampling.stl")
        writer.SetInputData(polydata)
        writer.Write()
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(polydata)
        locator.BuildLocator()
        mesh = o3d.io.read_triangle_mesh("inputSampling.stl")
        target_radius = radius
        surface_area = np.asarray(mesh.get_surface_area()).sum()
        estimated_points = int(surface_area / (np.pi * target_radius * target_radius))
        pcd = mesh.sample_points_poisson_disk(number_of_points=estimated_points)
        pcd = np.asarray(pcd.points)
        boundaryTree = cKDTree(bdpts)
        thresholdDistance = self.maxHoleRadius
        indices = boundaryTree.query_ball_point(pcd, r=thresholdDistance)
        pcd = np.array([point for i, point in enumerate(pcd) if not indices[i]])
        normals = []
        scals = []
        cellNormals = polydata.GetCellData().GetNormals()
        scalars = polydata.GetCellData().GetScalars()
        for point in pcd:
            cellId = locator.FindCell(point)
            normal = cellNormals.GetTuple(cellId)
            normals.append(normal)
            scals.append(scalars.GetValue(cellId))
        return pcd, normals, scals

    def genHoles(self, pts, nms, scals, polydata):
        writer = vtk.vtkSTLWriter()
        writer.SetFileName("temp_input.stl")
        writer.SetInputData(polydata)
        writer.Write()
        mesh = pv.read("temp_input.stl")
        mesh = mesh.triangulate()
        mesh = mesh.smooth(n_iter=16)
        visited = np.zeros(len(pts), dtype=bool)
        pts = pts.tolist()
        scals, pts, nms = zip(*sorted(zip(scals, pts, nms)))
        tree = cKDTree(pts)

        ogNumPoints = mesh.n_points
        holesCounter = 0

        for idx in tqdm(range(len(pts))):
            if not visited[idx]:
                center = pts[idx]
                normal = nms[idx]
                scal = scals[idx]
                if scal < self.maxHoleRadius:
                    neighbors = tree.query_ball_point(
                        center, 2 * (self.maxHoleRadius - scal)
                    )
                    neighbors = [
                        n for n in neighbors if not np.array_equal(pts[n], center)
                    ]
                    for neighbor in neighbors:
                        if (self.maxHoleRadius - scal) + (
                            self.maxHoleRadius - scals[neighbor]
                        ) + 1 >= self.distance(center, pts[neighbor]):
                            visited[neighbors] = True
                    cylinder = pv.Cylinder(
                        center=center,
                        radius=self.maxHoleRadius - scal,
                        height=6,
                        direction=normal,
                        resolution=12,
                    )
                    cylinder = cylinder.triangulate()
                    try:
                        difference = mesh.boolean_difference(cylinder)
                        if difference.n_cells > 0:
                            mesh = difference
                            holesCounter += 1
                    except:
                        pass
                    if idx % 10 == 0:
                        mesh = mesh.decimate(0.1)
        mesh.save("outputPolydata.stl")
        reader = vtk.vtkSTLReader()
        reader.SetFileName("outputPolydata.stl")
        reader.Update()
        print(
            f"sampledPoints: {len(pts)} generatedHoles: {holesCounter} initNumVertices: {ogNumPoints} endNumVertices: {mesh.n_points} time: {time.time()-self.initTime}"
        )
        return reader.GetOutput()

    def extractBoundaryPointIDs(self, source):
        # Setting up the ID filter to generate point IDs
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(source)
        idFilter.SetPointIds(True)
        idFilter.SetCellIds(False)
        idFilter.SetPointIdsArrayName("ids")
        idFilter.Update()

        # Extracting boundary edges
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(idFilter.GetOutput())
        edges.BoundaryEdgesOn()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.FeatureEdgesOff()
        edges.Update()

        # Retrieving the point IDs associated with the boundary edges
        array = edges.GetOutput().GetPointData().GetArray("ids")
        n = edges.GetOutput().GetNumberOfPoints()
        boundaryIds = [array.GetValue(i) for i in range(n)]

        return boundaryIds

    def findBoundaryNeighbors(self, polydata, boundaryPointIDs):
        boundarySet = set(boundaryPointIDs)
        neighbors = {pointId: [] for pointId in boundaryPointIDs}
        for pointId in boundaryPointIDs:
            cells = vtk.vtkIdList()
            polydata.GetPointCells(pointId, cells)
            for i in range(cells.GetNumberOfIds()):
                cellId = cells.GetId(i)
                idList = vtk.vtkIdList()
                polydata.GetCellPoints(cellId, idList)
                pointIdsInCell = [
                    idList.GetId(i) for i in range(idList.GetNumberOfIds())
                ]
                count = 0
                neighborIDFR = 0
                for testID in pointIdsInCell:
                    if testID == pointId:
                        pass
                    elif testID in neighbors:
                        count += 1
                        neighborIDFR = testID
                if count == 1:
                    neighbors[pointId].append(neighborIDFR)

        for key in neighbors:
            neighbors[key] = list(set(neighbors[key]))

        return neighbors

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def getMagnitude(self, vec):
        return (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2) ** 0.5

    def getNormal(self, vec1, vec2):
        vec = (
            vec1[1] * vec2[2] - vec1[2] * vec2[1],
            vec1[2] * vec2[0] - vec1[0] * vec2[2],
            vec1[0] * vec2[1] - vec1[1] * vec2[0],
        )
        magnitude = self.getMagnitude(vec)
        return (vec[0] / magnitude, vec[1] / magnitude, vec[2] / magnitude)

    def computeCellNormalsFromPointNormals(self, polydata):
        point_normals = polydata.GetPointData().GetNormals()

        # Create an array to store the computed cell normals
        cell_normals = vtk.vtkDoubleArray()
        cell_normals.SetNumberOfComponents(3)
        cell_normals.SetName("cell_normals")

        # For each cell in the polydata, compute the average normal of its points
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            avg_normal = [0.0, 0.0, 0.0]
            for j in range(cell.GetNumberOfPoints()):
                pointId = cell.GetPointId(j)
                normal = point_normals.GetTuple(pointId)
                avg_normal[0] += normal[0]
                avg_normal[1] += normal[1]
                avg_normal[2] += normal[2]

            # Normalize the average normal
            magnitude = sum([component**2 for component in avg_normal]) ** 0.5
            if magnitude > 0:
                normalized_normal = [component / magnitude for component in avg_normal]
                cell_normals.InsertNextTuple(normalized_normal)
            else:
                cell_normals.InsertNextTuple([0.0, 0.0, 0.0])

        polydata.GetCellData().SetNormals(cell_normals)

        return polydata

    def cleanSocketShell(self, polydata, reverse=True):
        original_normals = polydata.GetPointData().GetNormals()
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(polydata)
        triangle_filter.Update()
        triangulated_data = triangle_filter.GetOutput()
        triangulated_data.GetPointData().SetNormals(original_normals)
        normals_generator = vtk.vtkPolyDataNormals()
        normals_generator.SetInputData(triangulated_data)
        normals_generator.ComputePointNormalsOn()
        normals_generator.ComputeCellNormalsOff()
        normals_generator.Update()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(normals_generator.GetOutput())
        cleaner.SetTolerance(0.001)
        cleaner.PointMergingOn()
        cleaner.Update()
        polydata = cleaner.GetOutput()
        if reverse:
            reverseNormals = vtk.vtkReverseSense()
            reverseNormals.SetInputData(polydata)
            reverseNormals.ReverseCellsOn()
            reverseNormals.ReverseNormalsOn()
            reverseNormals.Update()
            polydata = reverseNormals.GetOutput()

        return polydata

    def removePointsFromPolydata(self, polydata, pointIds_to_remove):
        normals = vtk.vtkFloatArray()
        normals.SetNumberOfComponents(3)
        newPoints = vtk.vtkPoints()

        oldScalars = polydata.GetCellData().GetScalars()
        newScalars = vtk.vtkFloatArray()
        newScalars.SetNumberOfComponents(oldScalars.GetNumberOfComponents())
        newScalars.SetName(oldScalars.GetName())

        adjustedPointIds = {}

        difference = 0
        for i in range(polydata.GetNumberOfPoints()):
            if i not in pointIds_to_remove:
                point = polydata.GetPoint(i)
                direction = polydata.GetPointData().GetNormals().GetTuple(i)
                newPoints.InsertNextPoint(point)
                normals.InsertNextTuple(direction)
            else:
                difference += 1
            adjustedPointIds[i] = i - difference

        cellsToRemove = []
        for pointId in pointIds_to_remove:
            cellIds = vtk.vtkIdList()
            polydata.GetPointCells(pointId, cellIds)
            cells = [cellIds.GetId(i) for i in range(cellIds.GetNumberOfIds())]
            cellsToRemove.extend(cells)
        cellsToRemove = list(set(cellsToRemove))

        newSocketIds = {}
        difference2 = 0
        newCells = vtk.vtkCellArray()

        exp = vtk.vtkPoints()

        for i in range(polydata.GetNumberOfCells()):
            if i not in cellsToRemove:
                cell = polydata.GetCell(i)
                adjustedPointIds = [
                    adjustedPointIds[cell.GetPointId(j)]
                    for j in range(cell.GetNumberOfPoints())
                ]
                newCells.InsertNextCell(len(adjustedPointIds), adjustedPointIds)
                newScalars.InsertNextTuple(oldScalars.GetTuple(i))
            else:
                difference2 += 1

        newLayer = vtk.vtkPolyData()
        newLayer.SetPoints(newPoints)
        newLayer.GetPointData().SetNormals(normals)
        newLayer.SetPolys(newCells)
        newLayer.GetCellData().SetScalars(newScalars)
        return newLayer

    def extrusion(self, polydata):
        boundaryExtractor = BoundaryExtractor(polydata)
        boundaryLoops = boundaryExtractor.produceOrderedLoops()

        for i in range(len(boundaryLoops)):
            boundaryLoop = boundaryLoops[i]
            boundaryPoints = vtk.vtkPoints()
            for pointId in boundaryLoop:
                point = polydata.GetPoint(pointId)
                boundaryPoints.InsertNextPoint(point)
            boundaryPolydata = vtk.vtkPolyData()
            boundaryPolydata.SetPoints(boundaryPoints)
            self.plotPointCloud(
                boundaryPolydata,
                (0, 1 - (1 / len(boundaryLoops)) * i, i * (1 / len(boundaryLoops))),
            )
        totalBoundaryPoints = vtk.vtkPoints()
        for idx in [pt for loop in boundaryLoops for pt in loop]:
            point = polydata.GetPoint(idx)
            totalBoundaryPoints.InsertNextPoint(point)
        pts, nms, scals = self.samplePoints(
            self.computeCellNormalsFromPointNormals(polydata),
            3,
            vtk_to_numpy(totalBoundaryPoints.GetData()),
        )

        cringypoints = vtk.vtkPoints()
        for pt in pts:
            cringypoints.InsertNextPoint(pt)
        cringe = vtk.vtkPolyData()
        cringe.SetPoints(cringypoints)
        self.plotPointCloud(cringe, (0, 1, 1))

        normals = vtk.vtkFloatArray()
        normals.SetNumberOfComponents(3)
        newPoints = vtk.vtkPoints()
        for i in range(polydata.GetNumberOfPoints()):
            old_point = polydata.GetPoint(i)
            direction = polydata.GetPointData().GetNormals().GetTuple(i)
            magnitude = (sum([d * d for d in direction])) ** 0.5
            direction = [-d / magnitude for d in direction]
            new_point = [old_point[j] + self.thickness * direction[j] for j in range(3)]
            newPoints.InsertNextPoint(new_point)
            normals.InsertNextTuple(direction)
        newLayer = vtk.vtkPolyData()
        newLayer.SetPoints(newPoints)
        newLayer.GetPointData().SetNormals(normals)
        newCells = vtk.vtkCellArray()
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            adjusted_pointIds = [
                cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())
            ]
            newCells.InsertNextCell(len(adjusted_pointIds), adjusted_pointIds)
        newLayer.SetPolys(newCells)
        reverseNormals = vtk.vtkReverseSense()
        reverseNormals.SetInputData(newLayer)
        reverseNormals.ReverseCellsOn()
        reverseNormals.ReverseNormalsOff()
        reverseNormals.Update()
        newLayer = reverseNormals.GetOutput()

        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(self.computeCellNormalsFromPointNormals(polydata))
        appender.AddInputData(self.computeCellNormalsFromPointNormals(newLayer))
        appender.Update()
        combinedPolyData = appender.GetOutput()

        newCells = vtk.vtkCellArray()
        normals = vtk.vtkFloatArray()
        normals.SetNumberOfComponents(3)
        for cellId in range(polydata.GetNumberOfCells()):
            newCells.InsertNextCell(combinedPolyData.GetCell(cellId))
            newCells.InsertNextCell(
                combinedPolyData.GetCell(cellId + polydata.GetNumberOfCells())
            )
            normals.InsertNextTuple(
                combinedPolyData.GetCellData().GetNormals().GetTuple(cellId)
            )
            normals.InsertNextTuple(
                combinedPolyData.GetCellData()
                .GetNormals()
                .GetTuple(cellId + polydata.GetNumberOfCells())
            )

        for loop in boundaryLoops:
            for i in range(len(loop)):
                currPointID = loop[i]
                neighborID = loop[(i + 1) % len(loop)]
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, currPointID)
                triangle.GetPointIds().SetId(
                    2, currPointID + polydata.GetNumberOfPoints()
                )
                triangle.GetPointIds().SetId(1, neighborID)
                newCells.InsertNextCell(triangle)
                u = [
                    -combinedPolyData.GetPoint(currPointID)[n]
                    + combinedPolyData.GetPoint(neighborID)[n]
                    for n in range(3)
                ]
                v = [
                    -combinedPolyData.GetPoint(currPointID)[n]
                    + combinedPolyData.GetPoint(
                        currPointID + polydata.GetNumberOfPoints()
                    )[n]
                    for n in range(3)
                ]
                normals.InsertNextTuple([m for m in self.getNormal(u, v)])
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(
                    0, currPointID + polydata.GetNumberOfPoints()
                )
                triangle.GetPointIds().SetId(2, neighborID)
                triangle.GetPointIds().SetId(
                    1, neighborID + polydata.GetNumberOfPoints()
                )
                newCells.InsertNextCell(triangle)
                u = [
                    -combinedPolyData.GetPoint(
                        currPointID + polydata.GetNumberOfPoints()
                    )[n]
                    + combinedPolyData.GetPoint(neighborID)[n]
                    for n in range(3)
                ]
                v = [
                    -combinedPolyData.GetPoint(
                        currPointID + polydata.GetNumberOfPoints()
                    )[n]
                    + combinedPolyData.GetPoint(
                        neighborID + polydata.GetNumberOfPoints()
                    )[n]
                    for n in range(3)
                ]
                normals.InsertNextTuple([-m for m in self.getNormal(u, v)])

        finalData = vtk.vtkPolyData()
        finalData.SetPoints(combinedPolyData.GetPoints())
        finalData.SetPolys(newCells)
        finalData.GetCellData().SetNormals(normals)

        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputData(finalData)
        triangle_filter.Update()
        finalData = triangle_filter.GetOutput()

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(finalData)
        cleaner.SetTolerance(0.001)
        cleaner.PointMergingOn()
        cleaner.Update()
        finalData = cleaner.GetOutput()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName("withoutHoles.stl")
        writer.SetInputData(finalData)
        writer.Write()

        finalData = self.genHoles(pts, nms, scals, finalData)

        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(finalData)
        cleaner.SetTolerance(0.001)
        cleaner.PointMergingOn()
        cleaner.Update()
        finalData = cleaner.GetOutput()

        # self.plotVectors(finalData, 25, 20, 200, 100)
        # print(self.check_constraints(finalData))

        return finalData

    def plotPointCloud(self, polydata, color=(1, 1, 1), size=5):
        glyph_filter = vtk.vtkVertexGlyphFilter()
        glyph_filter.SetInputData(polydata)
        glyph_filter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph_filter.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(size)
        self.renderer.AddActor(actor)
        r, g, b = color
        actor.GetProperty().SetColor(r, g, b)
        self.renderWindow.Render()

    def createBaseConnector(
        self, baseRadius, connectorWidth, connectorHeight, pieceHeight
    ):
        resolution = 30
        connectorWidth /= 2
        connectorHeight += 2  # this is basically the minor axiso f the uper ellipse
        pieceHeight += 3
        bottomPoints = []
        thetaInterval = 2 * np.pi / resolution
        for i in range(resolution):
            bottomPoints.append(
                (
                    baseRadius * np.cos(thetaInterval * i),
                    baseRadius * np.sin(thetaInterval * i),
                    pieceHeight,
                )
            )
        bottomPoints.append((0, 0, pieceHeight))
        topPoints = []
        for i in range(resolution):
            topPoints.append(
                (
                    connectorWidth * np.cos(thetaInterval * i),
                    connectorHeight * np.sin(thetaInterval * i),
                    0,
                )
            )
        topPoints.append((0, 0, 0))
        totalFaces = []
        for i in range(len(bottomPoints) - 1):
            totalFaces.append(
                (3, len(bottomPoints) - 1, i, (i + 1) % (len(bottomPoints) - 1))
            )
        bottomPoints.extend(topPoints)
        offset = len(topPoints)
        for i in range(len(topPoints) - 1):
            totalFaces.append(
                (
                    3,
                    len(topPoints) - 1 + offset,
                    i + offset,
                    (i + 1) % (len(topPoints) - 1) + offset,
                )
            )
        for i in range(resolution):
            totalFaces.append((3, i, (i + 1) % resolution, i + offset))
            totalFaces.append(
                (3, i + offset, (i + 1) % resolution + offset, (i + 1) % resolution)
            )
        mesh = pv.PolyData(bottomPoints, totalFaces).triangulate()
        return mesh.compute_normals(consistent_normals=True, auto_orient_normals=True)

    def createConnectorHalf(
        self, endRadius, length, thickness, resolutionEnd=8, width=6
    ):
        vertices = []
        totalFaces = []
        vertices.append((0, thickness / 2, 0))
        vertices.append((0, thickness / 2, length / 2))
        vertices.append((0, -thickness / 2, length / 2))
        vertices.append((0, -thickness / 2, 0))

        totalFaces.append((3, 0, 1, 2))
        totalFaces.append((3, 2, 3, 0))

        currTheta = np.pi - np.arcsin(thickness / 2 / endRadius)
        thetaInterval = 2 * currTheta / (resolutionEnd + 1)
        currTheta -= thetaInterval
        for _ in range(resolutionEnd):
            vertices.append(
                (
                    0,
                    endRadius * np.sin(currTheta),
                    endRadius * np.cos(currTheta) + length / 2 + endRadius,
                )
            )
            currTheta -= thetaInterval
        vertices.append((0, 0, length / 2 + endRadius))
        offset = 4
        for i in range(resolutionEnd):
            totalFaces.append((3, i + offset, len(vertices) - 1, i + 1 + offset))

        totalFaces.append((3, 1, len(vertices) - 1, 2))
        totalFaces.append((3, 1, len(vertices) - 1, 4))
        totalFaces.append((3, 4 + resolutionEnd - 1, len(vertices) - 1, 2))

        return vertices, totalFaces

    def createFullConnector(self, endRadius, length, thickness, resolutionEnd, width):
        length = length / 2 + 1
        half1Vert, half1Face = self.createConnectorHalf(
            endRadius=endRadius,
            length=length,
            thickness=thickness,
            resolutionEnd=resolutionEnd,
        )
        half2Vert, half2Face = self.createConnectorHalf(
            endRadius=endRadius,
            length=length,
            thickness=thickness,
            resolutionEnd=resolutionEnd,
        )
        for i, vert in enumerate(half2Vert):
            half2Vert[i] = (vert[0] - width / 2, vert[1], vert[2])
        for i, vert in enumerate(half1Vert):
            half1Vert[i] = (vert[0] + width / 2, vert[1], vert[2])
        offset = len(half1Vert)
        for i, face in enumerate(half2Face):
            half2Face[i] = (3, face[1] + offset, face[2] + offset, face[3] + offset)
        totalVerts = []
        totalFaces = []
        totalVerts.extend(half1Vert)
        totalVerts.extend(half2Vert)
        totalFaces.extend(half1Face)
        totalFaces.extend(half2Face)

        totalFaces.append((3, 0, 1, offset))
        totalFaces.append((3, 1, offset + 1, offset))

        totalFaces.append((3, 2, 3, offset + 3))
        totalFaces.append((3, 2, offset + 2, offset + 3))

        totalFaces.append((3, 0, offset, 3))
        totalFaces.append((3, offset + 3, offset, 3))

        for i in range(resolutionEnd - 1):
            totalFaces.append((3, i + 4, i + 5, i + 4 + offset))
            totalFaces.append((3, i + 4 + offset, i + 5 + offset, i + 5))

        totalFaces.append((3, 1, 1 + offset, 4 + offset))
        totalFaces.append((3, 4 + offset, 4, 1))

        totalFaces.append((3, 2, 2 + offset, 3 + offset + resolutionEnd))
        totalFaces.append((3, 3 + offset + resolutionEnd, 3 + resolutionEnd, 2))

        for i, point in enumerate(totalVerts):
            totalVerts[i] = point + np.array([0, 0, -1])

        mesh = (
            pv.PolyData(totalVerts, totalFaces)
            .compute_normals(
                non_manifold_traversal=False,
                consistent_normals=True,
                auto_orient_normals=True,
            )
            .clean()
            .triangulate()
        )
        return mesh

    def createConnectionBetweenFingerAndSocket(
        self,
        baseRadius,
        connectorWidth,
        connectorThickness,
        connectorLength,
        connectorRadius,
    ):
        resolutionEnd = 12
        base = self.createBaseConnector(
            baseRadius=baseRadius,
            connectorWidth=connectorWidth,
            connectorHeight=connectorThickness,
            pieceHeight=connectorLength,
        )
        connector = self.createFullConnector(
            width=connectorWidth * 3,
            endRadius=connectorRadius,
            length=connectorLength,
            thickness=connectorThickness,
            resolutionEnd=resolutionEnd,
        )
        boolean = vtkPolyDataBooleanFilter()
        boolean.SetInputData(0, base)
        boolean.SetInputData(1, connector)
        boolean.SetOperModeToDifference()
        boolean.Update()
        return pv.wrap(boolean.GetOutput())

    def moveAlignMesh(self, mesh, newCenter, newAlignVector, newNormal):
        otherNormal = np.cross(newAlignVector, newNormal)
        v1 = otherNormal / np.linalg.norm(otherNormal)
        v2 = newNormal / np.linalg.norm(newNormal) * -1
        v3 = newAlignVector / np.linalg.norm(newAlignVector)
        if (
            not np.allclose(np.dot(v1, v2), 0)
            or not np.allclose(np.dot(v1, v3), 0)
            or not np.allclose(np.dot(v2, v3), 0)
        ):
            raise ValueError("Basis vectors are not orthogonal")
        transformation_matrix = np.column_stack((v1, v2, v3))
        finalTransformation = np.eye(4)
        finalTransformation[0:3, 0:3] = transformation_matrix
        mesh = mesh.transform(finalTransformation)
        translation_vector = newCenter - np.array([0, 0, 0], dtype=float)
        mesh = mesh.translate(translation_vector)
        return mesh

    def genHandPortion(self, carpals, fingerMeshes, fingerInfo):
        handMesh = pv.wrap(self.actor.GetMapper().GetInput())
        jointRegionalPointIDXs = []

        socket = pv.wrap(self.finalSocket)
        newProportionalPositions = {}

        initRadius = 9.5

        overallTime = time.time()
        for i, carpal in enumerate(carpals):
            start = time.time()

            tempIDList = []
            point, _ = socket.ray_trace(
                carpal["center"],
                [carpal["center"][i] - carpal["normal"][i] for i in range(3)],
                first_point=True,
            )
            selectingSphere = pv.Sphere(radius=initRadius + layers, center=point)
            selectedIds = socket.select_enclosed_points(selectingSphere)[
                "SelectedPoints"
            ].view(bool)

            threshold = 70 * np.pi / 180
            socket = socket.compute_normals()
            for idx in range(len(selectedIds)):
                if selectedIds[idx]:
                    point = socket.points[idx]
                    u = socket.point_normals[idx]
                    v = np.array(carpal["normal"])
                    angle = np.arccos(
                        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                    )
                    if angle < threshold:
                        tempIDList.append(idx)
            print("normaal", np.array(carpal["normal"]))
            pointsToPlot = socket.extract_points(
                tempIDList, adjacent_cells=False, include_cells=True
            ).points
            pls = vtk.vtkPoints()
            for pt in pointsToPlot:
                pls.InsertNextPoint(pt)
            fd = vtk.vtkPolyData()
            fd.SetPoints(pls)
            self.plotPointCloud(fd, color=(0.5, 0.2, 1))
            self.renderWindow.Render()

            jointRegionalPointIDXs.append(tempIDList)

            file1 = open("strengthAnalysis/connectiveGenerationTimes.txt", "a")
            file1.write(f"initial selection {i}: {time.time()-start}\n")
            file1.close()

        def selectPoints(
            carpalOrigin,
            carpalVector,
            radius: int,
            color=(0.3, 0.2, 1),
            size=5,
        ):
            threshold = 5e-1
            start = time.time()
            point, _ = socket.ray_trace(
                carpalOrigin,
                [carpalOrigin[i] - carpalVector[i] for i in range(3)],
                first_point=True,
            )
            selectingSphere = pv.Sphere(radius=radius, center=point)
            selectedIds = socket.select_enclosed_points(selectingSphere)[
                "SelectedPoints"
            ].view(bool)
            extractedPointIdx = []
            threshold = 70 * np.pi / 180
            for idx in range(len(selectedIds)):
                if selectedIds[idx]:
                    point = socket.points[idx]
                    u = socket.point_normals[idx]
                    v = np.array(carpal["normal"])
                    angle = np.arccos(
                        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                    )
                    if angle < threshold:
                        extractedPointIdx.append(idx)
            pointsToPlot = socket.extract_points(
                selectedIds, adjacent_cells=False, include_cells=True
            ).points
            pls = vtk.vtkPoints()
            for pt in pointsToPlot:
                pls.InsertNextPoint(pt)
            fd = vtk.vtkPolyData()
            fd.SetPoints(pls)
            self.plotPointCloud(fd, color=color, size=size)
            self.renderWindow.Render()

            file1 = open("strengthAnalysis/connectiveGenerationTimes.txt", "a")
            file1.write(f"selection: {time.time()-start}\n")
            file1.close()
            return extractedPointIdx

        def moveSelectedPointsToJoints(
            pointIDs, jointIdx, totalJointIds, fingerVector: list, jointOrigin: list
        ):
            distances = []
            diskOfPoints = []
            initDistances = []
            for idx in pointIDs:
                p = np.array(socket.points[idx])
                u = p - np.array(jointOrigin)
                n = np.array(fingerVector) / np.linalg.norm(np.array(fingerVector))
                newPos = p - n * np.dot(u, n)
                newProportionalPositions[idx] = newPos
                for j in range(jointIdx, totalJointIds):
                    if idx in jointRegionalPointIDXs[j]:
                        jointRegionalPointIDXs[j].remove(idx)
                distances.append(np.linalg.norm(newPos - p))
                diskOfPoints.append(newPos)
                initDistances.append(np.linalg.norm(newPos - np.array(jointOrigin)))
            circumferenceOfPoints = [[], [], []]
            for pair in ConvexHull(diskOfPoints, qhull_options="QJ").vertices:
                circumferenceOfPoints[0].append(diskOfPoints[pair])
                circumferenceOfPoints[1].append(distances[pair])
                circumferenceOfPoints[2].append(initDistances[pair])
            pointsToPlot = circumferenceOfPoints[0]
            pls = vtk.vtkPoints()
            for pt in pointsToPlot:
                pls.InsertNextPoint(pt)
            fd = vtk.vtkPolyData()
            fd.SetPoints(pls)
            self.plotPointCloud(fd, color=(1, 1, 0.3))
            self.renderWindow.Render()
            return circumferenceOfPoints

        def proportionallyMovePoints(
            fingerVector: list,
            jointOrigin: list,
            jointIdx,
            circumferenceOfPoints,
        ):
            tree = cKDTree(circumferenceOfPoints[0])
            for idx in jointRegionalPointIDXs[jointIdx]:
                p = np.array(socket.points[idx])
                u = p - np.array(jointOrigin)

                n = np.array(fingerVector) / np.linalg.norm(np.array(fingerVector))
                projectedPos = p - n * np.dot(u, n)
                distance2 = np.linalg.norm(projectedPos - np.array(jointOrigin))

                _, closestPointIdx = tree.query(p)
                intensity = circumferenceOfPoints[1][closestPointIdx]
                distance1 = circumferenceOfPoints[2][closestPointIdx]
                # factor = distance / ((np.linalg.norm(u) / distance) ** 2)
                factor = intensity * (distance1**2) / (distance2**2)
                newPos = p + n * factor
                if idx in newProportionalPositions:
                    newProportionalPositions[idx] = (
                        newProportionalPositions[idx] + newPos
                    ) / 2
                else:
                    newProportionalPositions[idx] = newPos

        conjoiningMeshes = []

        for k, carpal in enumerate(carpals):
            origin = carpal["center"] - carpal["normal"] / np.linalg.norm(
                carpal["normal"]
            ) * (fingerInfo["connectorLength"] + 3)
            listOfIds = selectPoints(
                carpalOrigin=carpal["center"],
                carpalVector=carpal["normal"],
                radius=initRadius - 0.4,
                color=(0.5, 1, 1),
                size=10,
            )
            circumferenceOfPoints = moveSelectedPointsToJoints(
                pointIDs=listOfIds,
                jointIdx=k,
                totalJointIds=len(carpals),
                fingerVector=carpal["normal"],
                jointOrigin=origin,
            )
            proportionallyMovePoints(
                fingerVector=carpal["normal"],
                jointOrigin=origin,
                circumferenceOfPoints=circumferenceOfPoints,
                jointIdx=k,
            )
            connectorPiece = self.createConnectionBetweenFingerAndSocket(
                baseRadius=initRadius,
                connectorWidth=fingerInfo["connectorWidth"],
                connectorThickness=fingerInfo["connectorThickness"],
                connectorLength=fingerInfo["connectorLength"],
                connectorRadius=fingerInfo["connectorRadius"],
            )

            carpalNeighborIndex = k + 1
            if k == len(carpals) - 1:
                carpalNeighborIndex = k - 1
            crossedBiNormal = np.cross(
                carpal["normal"], carpals[carpalNeighborIndex]["normal"]
            )
            connectorPieceLocation = carpal["center"]
            connectorPiece = self.moveAlignMesh(
                connectorPiece,
                connectorPieceLocation,
                -1 * carpal["normal"],
                -1 * crossedBiNormal,
            )
            conjoiningMeshes.append(connectorPiece)

        for pointToMove, newPointPosition in newProportionalPositions.items():
            socket.points[pointToMove] = newPointPosition
        # for mesh in conjoiningMeshes:
        #     socket = socket.merge(mesh)
        smoothSocket = socket.smooth_taubin(n_iter=100, pass_band=0.5)
        cleanedSocket = smoothSocket.clean()
        print(f"final time: {time.time()-overallTime}")
        cleanedSocket.plot()

        self.finalSocket.DeepCopy(cleanedSocket)
        self.finalSocket.Modified()
        self.renderWindow.Render()

        appendFilter = vtkAppendPolyData()
        appendFilter.AddInputData(self.finalSocket)
        for i, finger in enumerate(fingerMeshes):
            appendFilter.AddInputData(finger)
        appendFilter.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName("toBePrinted/totalStruct.stl")
        writer.SetInputData(appendFilter.GetOutput())
        writer.Write()

        writer.SetFileName("imageAnalysisGeneration/socketStruct.stl")
        writer.SetInputData(self.finalSocket)
        writer.Write()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName("imageAnalysisGeneration/connectorSocketToFinger.stl")
        writer.SetInputData(conjoiningMeshes[0])
        writer.Write()


class BoundaryExtractor:
    def __init__(self, source):
        self.source = source
        self.toRemove = []
        self.totalBoundaryIds = []
        self.totalNeighbors = []
        self.genInitTotalBounds()

    def extractBoundaryPointIDs(self):
        idFilter = vtk.vtkIdFilter()
        idFilter.SetInputData(self.source)
        idFilter.SetPointIds(True)
        idFilter.SetCellIds(False)
        idFilter.SetPointIdsArrayName("ids")
        idFilter.Update()
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(idFilter.GetOutput())
        edges.BoundaryEdgesOn()
        edges.ManifoldEdgesOff()
        edges.NonManifoldEdgesOff()
        edges.FeatureEdgesOff()
        edges.Update()
        array = edges.GetOutput().GetPointData().GetArray("ids")
        n = edges.GetOutput().GetNumberOfPoints()
        self.totalBoundaryIds = [array.GetValue(i) for i in range(n)]

    def genInitTotalBounds(self):
        self.extractBoundaryPointIDs()
        neighbors = self.findBoundaryNeighbors()
        toRemove = []
        for i, d in neighbors.items():
            if not d:
                toRemove.append(i)
        for boundId in toRemove:
            self.totalBoundaryIds.remove(boundId)
            del neighbors[boundId]
        self.toRemove = toRemove
        self.totalNeighbors = self.findBoundaryNeighbors()

    def produceOrderedLoops(self):
        loops = []
        bucketOfPoints = self.totalBoundaryIds
        while len(bucketOfPoints) != 0:
            ogPointID = bucketOfPoints[0]
            prevPointID = ogPointID
            currPointID = ogPointID
            start = True
            currLoop = []
            while currPointID != ogPointID or start == True:
                bucketOfPoints.remove(currPointID)
                currLoop.append(currPointID)
                neighborID = self.totalNeighbors[currPointID][0]
                if neighborID == prevPointID:
                    neighborID = self.totalNeighbors[currPointID][1]
                prevPointID = currPointID
                currPointID = neighborID
                start = False
            loops.append(currLoop)
        return loops

    def findBoundaryNeighbors(self):
        boundarySet = set(self.totalBoundaryIds)
        neighbors = {pointId: [] for pointId in self.totalBoundaryIds}
        for pointId in self.totalBoundaryIds:
            cells = vtk.vtkIdList()
            self.source.GetPointCells(pointId, cells)
            for i in range(cells.GetNumberOfIds()):
                cellId = cells.GetId(i)
                idList = vtk.vtkIdList()
                self.source.GetCellPoints(cellId, idList)
                pointIdsInCell = [
                    idList.GetId(i) for i in range(idList.GetNumberOfIds())
                ]
                count = 0
                neighborIDFR = 0
                for testID in pointIdsInCell:
                    if testID == pointId:
                        pass
                    elif testID in neighbors:
                        count += 1
                        neighborIDFR = testID
                if count == 1:
                    neighbors[pointId].append(neighborIDFR)
        for key in neighbors:
            neighbors[key] = list(set(neighbors[key]))
        return neighbors


class VTKErrorObserver:
    def __init__(self):
        self.__ErrorOccurred = False
        self.__ErrorMessage = None
        self.CallDataType = "string0"

    def __call__(self, obj, event, message):
        self.__ErrorOccurred = True
        self.__ErrorMessage = message

    def has_error_occurred(self):
        return self.__ErrorOccurred

    def get_error_message(self):
        return self.__ErrorMessage


class HandManipulator:
    def __init__(self, renderWinIn, ren, renWin):
        self.renderWindowInteractor = renderWinIn
        self.renderer = ren
        self.renderWindow = renWin
        self.boneReference = {}
        self.thumbJoints = {}
        self.indexJoints = {}
        self.middleJoints = {}
        self.thirdJoints = {}
        self.fourthJoints = {}

        self.rootSphere = None
        self.createSkeleton()
        self.pickedActor = None
        self.originalPosPickedActor = [0] * 3

        self.fingerActors = []
        self.fingerConnectorActors = []

        self.proportions = pd.read_csv("csvData/relativeLengths.csv").set_index("Names")

        self.fingerHeight = 12
        self.fingerWidth = 14
        self.connectorThickness = 1.3
        self.connectorRadius = 1.7

    @property
    def jointsList(self):
        return [
            self.indexJoints,
            self.middleJoints,
            self.thirdJoints,
            self.fourthJoints,
        ]

    def getJoints(self):
        joints = []
        for finger in self.jointsList:
            fingerJoints = list(finger.keys())
            palmMostJoint = 4
            for i, joint in enumerate(fingerJoints):
                if joint.toggled:
                    if i <= palmMostJoint:
                        palmMostJoint = i
            if palmMostJoint < 4:
                joints.append(
                    {
                        "center": np.array(fingerJoints[palmMostJoint].center),
                        "normal": np.array(fingerJoints[1].center)
                        - np.array(fingerJoints[0].center),
                    }
                )
        return joints

    def setFingerWidth(self, val):
        self.fingerWidth = val

    def setFingerHeight(self, val):
        self.fingerHeight = val

    def setConnectorThickness(self, val):
        self.connectorThickness = val

    def setConnectorRadius(self, val):
        self.connectorRadius = val

    def genInitBonesPerFinger(self, fileName: str, finger: str):
        bones = []
        joints = {}
        points = pd.read_csv(fileName)
        points = points.set_index("PointName")
        points["Coors"] = points.values.tolist()
        # firstMetacarpal
        bones.append(cylinderBone(points["Coors"].iloc[0], points["Coors"].iloc[1]))
        for i in range(1, points.shape[0] - 1):
            bones.append(
                cylinderBone(points["Coors"].iloc[i], points["Coors"].iloc[i + 1])
            )
            joint = sphereBone(points["Coors"].iloc[i], finger + points.index[i])
            joints.update(
                {
                    joint: [
                        bones[i - 1],
                        bones[i],
                    ]
                }
            )
            bones[i - 1].setEndSphere(joint)
            bones[i].setStartSphere(joint)

        # lastEndJoint
        endJoint = sphereBone(points["Coors"].iloc[-1], finger + points.index[-1])
        joints.update({endJoint: [bones[-1]]})
        bones[-1].setEndSphere(endJoint)

        return bones, joints

    def createSkeleton(self):
        indexBones, self.indexJoints = self.genInitBonesPerFinger(
            "csvData/InitialBoneLocations - Index.csv", "index"
        )
        middleBones, self.middleJoints = self.genInitBonesPerFinger(
            "csvData/InitialBoneLocations - Middle.csv", "middle"
        )
        thirdBones, self.thirdJoints = self.genInitBonesPerFinger(
            "csvData/InitialBoneLocations - Third.csv", "third"
        )
        fourthBones, self.fourthJoints = self.genInitBonesPerFinger(
            "csvData/InitialBoneLocations - Fourth.csv", "fourth"
        )

        point = pd.read_csv(f"csvData/InitialBoneLocations - Index.csv")
        rootSphere = sphereBone(
            [point.loc[0, "X"], point.loc[0, "Y"], point.loc[0, "Z"]], "Root"
        )
        self.rootSphere = rootSphere
        self.boneReference.update(
            {
                rootSphere: [
                    indexBones[0],
                    middleBones[0],
                    thirdBones[0],
                    fourthBones[0],
                ]
            }
        )
        indexBones[0].setStartSphere(rootSphere)
        middleBones[0].setStartSphere(rootSphere)
        thirdBones[0].setStartSphere(rootSphere)
        fourthBones[0].setStartSphere(rootSphere)

        self.boneReference.update(self.indexJoints)
        self.boneReference.update(self.middleJoints)
        self.boneReference.update(self.thirdJoints)
        self.boneReference.update(self.fourthJoints)

        idTagNum = 0
        for sphere, cylinders in self.boneReference.items():
            sphere.setIdTag(idTagNum)
            self.renderer.AddActor(sphere.actor)
            for cylinder in cylinders:
                self.renderer.AddActor(cylinder.actor)
            idTagNum += 1

    def on_click(self, obj, event):
        click_pos = self.renderWindowInteractor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        self.pickedActor = picker.GetActor()
        if "sphere" in getattr(self.pickedActor, "type", "Na"):
            self.originalPosPickedActor = self.pickedActor.GetCenter()
            clickedSphere = list(self.boneReference.keys())[self.pickedActor.idTag]

    def on_drag(self, obj, event):
        if self.pickedActor:
            mouse_pos = self.renderWindowInteractor.GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.Pick(mouse_pos[0], mouse_pos[1], 0, self.renderer)

            camera = self.renderer.GetActiveCamera()
            n = np.array(camera.GetDirectionOfProjection())
            newPoint = (
                np.array(picker.GetPickPosition())
                - np.dot(
                    np.array(picker.GetPickPosition()) - (self.originalPosPickedActor),
                    n,
                )
                * n
            )

            if hasattr(self.pickedActor, "idTag"):
                clickedSphere = list(self.boneReference.keys())[self.pickedActor.idTag]
                self.maintainFingerProportions(clickedSphere, newPoint)
            self.renderWindow.Render()

    def maintainFingerProportions(self, clickedSphere, newPoint):
        if clickedSphere.name == "Root":
            displacement = [
                newPoint[idx] - clickedSphere.center[idx] for idx in range(3)
            ]
            clickedSphere.move(newPoint)
            for sphere in self.boneReference.keys():
                if sphere != clickedSphere:
                    sphere.move(
                        [displacement[idx] + sphere.center[idx] for idx in range(3)]
                    )
                for cylinder in self.boneReference[sphere]:
                    cylinder.move(sphere)
        else:
            fingerDict = {}
            column = ""
            if "index" in clickedSphere.name:
                fingerDict = self.indexJoints
                column = "Index"
            elif "middle" in clickedSphere.name:
                fingerDict = self.middleJoints
                column = "Middle"
            elif "third" in clickedSphere.name:
                fingerDict = self.thirdJoints
                column = "Third"
            elif "fourth" in clickedSphere.name:
                fingerDict = self.fourthJoints
                column = "Fourth"

            sphereIndex = list(fingerDict.keys()).index(clickedSphere)

            directionVector = np.array(
                [newPoint[j] - self.rootSphere.center[j] for j in range(3)]
            )
            directionVector = list(directionVector / np.linalg.norm(directionVector))

            totalNewLength = np.linalg.norm(
                np.array([newPoint[j] - self.rootSphere.center[j] for j in range(3)])
            )
            ogLength = self.proportions[column][: sphereIndex + 1].sum()
            ratio = totalNewLength / ogLength

            for idx in range(len(fingerDict.keys())):
                sphere = list(fingerDict.keys())[idx]
                prevCoor = []
                if idx == 0:
                    prevCoor = self.rootSphere.center
                else:
                    prevCoor = list(fingerDict.keys())[idx - 1].center
                # currLength = np.linalg.norm(
                #     np.array([sphere.center[j] - prevCoor[j] for j in range(3)])
                # )
                ogLengthJoint = self.proportions[column].iloc[idx]
                newLength = ogLengthJoint * ratio
                sphere.move(
                    [directionVector[j] * newLength + prevCoor[j] for j in range(3)]
                )
                for cylinder in fingerDict[sphere]:
                    cylinder.move(sphere)

    def on_release(self, obj, event):
        self.pickedActor = None
        self.originalPosPickedActor = [0] * 3

    def saveFingerPositions(self):
        fingerNames = ["Index", "Middle", "Third", "Fourth"]
        for fingeridx, finger in enumerate(fingerNames):
            points = pd.read_csv(f"csvData/InitialBoneLocations - {finger}.csv")
            points.loc[0, "X"] = self.rootSphere.center[0]
            points.loc[0, "Y"] = self.rootSphere.center[1]
            points.loc[0, "Z"] = self.rootSphere.center[2]
            for idx, sphere in enumerate(self.jointsList[fingeridx].keys()):
                points.loc[idx + 1, "X"] = sphere.center[0]
                points.loc[idx + 1, "Y"] = sphere.center[1]
                points.loc[idx + 1, "Z"] = sphere.center[2]
            points.to_csv(f"csvData/InitialBoneLocations - {finger}.csv", index=False)

    def clearOldFingers(self):
        for finger in self.fingerActors:
            for part in finger:
                self.renderer.RemoveActor(part)
        for actor in self.fingerConnectorActors:
            self.renderer.RemoveActor(actor)
        self.fingerActors = []
        self.fingerConnectorActors = []

    def getFingerMeshes(self):
        fingerMeshes = []
        for finger in self.fingerActors:
            for thing in finger:
                fingerMeshes.append(thing.GetMapper().GetInput())
        return fingerMeshes

    def getFingerInfo(self):
        return {
            "connectorWidth": self.fingerWidth,
            "connectorThickness": self.connectorThickness,
            "connectorLength": 5.8,
            "connectorRadius": self.connectorRadius,
        }

    def getConnectors(self):
        fingerConnectors = []
        for finger in self.fingerConnectorActors:
            for thing in finger:
                fingerConnectors.append(thing.GetMapper().GetInput())
        return fingerConnectors

    def generateFingers(self):
        self.clearOldFingers()
        for jointListIdx in range(len(self.jointsList)):
            start = time.time()
            jointList = self.jointsList[jointListIdx]
            length = np.linalg.norm(
                np.array(list(jointList.keys())[len(jointList.keys()) - 1].center)
                - np.array(list(jointList.keys())[0].center)
            )
            tempMeshList = []
            jointNeighborIndex = jointListIdx + 1
            if jointListIdx == len(self.jointsList) - 1:
                jointNeighborIndex = jointListIdx - 1
            jointNeighborList = self.jointsList[jointNeighborIndex]
            central = np.array(self.rootSphere.center)
            u = np.array(list(jointList.keys())[1].center) - central
            v = np.array(list(jointNeighborList.keys())[1].center) - central
            normal = np.cross(u, v)
            if jointListIdx == len(self.jointsList) - 1:
                normal *= -1
            for jointIdx in range(1, len(self.jointsList[jointListIdx])):
                pointEnd = np.array(list(jointList.keys())[jointIdx].center)
                pointStart = np.array(list(jointList.keys())[jointIdx - 1].center)
                vector = pointEnd - pointStart
                midpoint = (pointStart + pointEnd) / 2
                fingerBodyLength = np.linalg.norm(vector)
                direction = vector / fingerBodyLength

                topRadii = 5
                bottomRadii = 4
                fingerBodyHeight = self.fingerHeight
                fingerBodyWidth = self.fingerWidth
                resolution = 16
                stringHoleRadius = 1
                connectorThickness = self.connectorThickness
                connectorRadius = self.connectorRadius
                connectorLength = 5.8
                connectorDistanceBelowMidline = 1.5

                part = self.generateFingerPartWithConnectorCut(
                    resolution=resolution,
                    topRadii=topRadii,
                    bottomRadii=bottomRadii,
                    fingerBodyHeight=fingerBodyHeight,
                    fingerBodyWidth=fingerBodyWidth,
                    stringHoleRadius=stringHoleRadius,
                    fingerBodyLength=fingerBodyLength,
                    connectorRadius=connectorRadius,
                    connectorLength=connectorLength,
                    connectorThickness=connectorThickness,
                    connectorDistanceBelowMidline=connectorDistanceBelowMidline,
                )

                part = self.moveAlignMesh(part, midpoint, direction, normal)
                vtk_polydata = part.extract_geometry()
                mapper = vtk.vtkPolyDataMapper()

                mapper.SetInputData(vtk_polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                tempMeshList.append(actor)
                file1 = open(
                    "strengthAnalysis/fingerGenerationTimes.txt", "a"
                )  # append mode
                file1.write(f"{length} {time.time()-start}\n")
                file1.close()
            self.fingerActors.append(tempMeshList)

            resolutionEnd = 10
            resolutionBody = 10
            thickness = 1
            endRadius = 1.5
            length = 5.5

            width = fingerBodyWidth

            tempMeshList = []
            for jointIdx in range(1, len(self.jointsList[jointListIdx])):
                pointEnd = np.array(list(jointList.keys())[jointIdx].center)
                pointStart = np.array(list(jointList.keys())[jointIdx - 1].center)
                vector = pointEnd - pointStart
                direction = vector / np.linalg.norm(vector)

                part = self.genConnector(
                    endRadius=endRadius,
                    length=length,
                    thickness=thickness,
                    resolutionBody=resolutionBody,
                    resolutionEnd=resolutionEnd,
                    width=width + 1,
                )
                part = self.moveAlignMesh(
                    part,
                    pointStart,
                    direction,
                    normal,
                )
                vtk_polydata = part.extract_geometry()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(vtk_polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                tempMeshList.append(actor)
            self.fingerConnectorActors.append(tempMeshList)

        self.displayFingerMeshes()

    def generateFingerPartWithConnectorCut(
        self,
        resolution,
        topRadii,
        bottomRadii,
        fingerBodyHeight,
        fingerBodyWidth,
        stringHoleRadius,
        fingerBodyLength,
        connectorRadius,
        connectorLength,
        connectorThickness,
        connectorDistanceBelowMidline,
    ):

        def endFaceTop(
            resolution,
            topRadii,
            bottomRadii,
            fingerBodyHeight,
            fingerBodyWidth,
            holeRadius,
        ):
            newPoints = []
            faces = []
            thetaInterval = np.pi / (resolution)
            newPoints.append((0, 0, 0))
            offset = len(newPoints)
            for i in range(2):
                batch = []
                theta = 0
                while theta < np.pi / 2:
                    radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2 - topRadii:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                radius * np.sin(theta),
                                radius * np.sin(theta),
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                radius * np.sin(theta),
                                radius * np.sin(theta),
                            )
                        )
                        theta += thetaInterval
                    radius = fingerBodyHeight / 2 / np.sin(theta)
                    while radius * np.cos(theta) > 0:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                radius * np.sin(theta),
                                radius * np.sin(theta),
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyHeight / 2 / np.sin(theta)
                if i % 2 == 1:
                    batch.reverse()
                    batch.pop(0)
                newPoints.extend(batch)
            for i in range(resolution + 1):
                theta = 2 * np.pi / resolution * i + (3 * np.pi / 2)
                newPoints.append(
                    (
                        holeRadius * np.cos(theta),
                        holeRadius * np.sin(theta) + fingerBodyHeight / 4,
                        holeRadius * np.sin(theta) + fingerBodyHeight / 4,
                    )
                )
            for i in range(resolution + 1):
                faces.append(
                    (
                        3,
                        i + offset,
                        (i + 1) % (resolution + 1) + offset,
                        (i) % (resolution + 1) + resolution + offset + 1,
                    )
                )
                faces.append(
                    (
                        3,
                        (i + 1) % (resolution + 1) + offset,
                        (i + 1) % (resolution + 1) + resolution + offset + 1,
                        (i) % (resolution + 1) + resolution + offset + 1,
                    )
                )
            return (np.array(newPoints), faces)

        def endFaceBottom(
            resolution,
            topRadii,
            bottomRadii,
            fingerBodyHeight,
            fingerBodyWidth,
            holeRadius,
        ):
            newPoints = []
            faces = []
            thetaInterval = np.pi / (resolution)
            for i in range(2):
                batch = []
                theta = 0
                while theta < np.pi / 2:
                    radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2 - bottomRadii:
                        theta += thetaInterval
                        radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                -radius * np.sin(theta),
                                0,
                            )
                        )
                        theta += thetaInterval
                    radius = fingerBodyHeight / 2 / np.sin(theta)
                    while radius * np.cos(theta) > 0:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                -radius * np.sin(theta),
                                0,
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyHeight / 2 / np.sin(theta)
                if i % 2 == 1:
                    batch.reverse()
                    batch.pop(0)
                newPoints.extend(batch)
            newPoints.pop(0)
            newPoints.pop()
            return np.array(newPoints)

        def createConnector(
            endRadius,
            length,
            thickness,
            connectorDistanceBelowMidline,
            resolutionBody=8,
            resolutionEnd=8,
        ):
            verticesTop = []
            verticesBottom = []
            verticesTop.append((0, thickness / 2, 0))
            verticesTop.append((0, thickness / 2, length / 2))
            currTheta = np.pi - np.arcsin(thickness / 2 / endRadius)
            thetaInterval = 2 * currTheta / (resolutionEnd + 1)
            currTheta -= thetaInterval
            for _ in range(int(resolutionEnd / 2)):
                verticesTop.append(
                    (
                        0,
                        endRadius * np.sin(currTheta),
                        endRadius * np.cos(currTheta) + length / 2 + endRadius,
                    )
                )
                verticesBottom.append(
                    (
                        0,
                        endRadius
                        * np.sin(currTheta - thetaInterval * (int(resolutionEnd / 2))),
                        endRadius
                        * np.cos(currTheta - thetaInterval * (int(resolutionEnd / 2)))
                        + length / 2
                        + endRadius,
                    )
                )
                currTheta -= thetaInterval
            for i, vertex in enumerate(verticesTop):
                verticesTop[i] = (
                    vertex[0],
                    vertex[1] - connectorDistanceBelowMidline,
                    vertex[2],
                )
            verticesBottom.append((0, -thickness / 2, length / 2))
            verticesBottom.append((0, -thickness / 2, 0))

            for i, vertex in enumerate(verticesBottom):
                verticesBottom[i] = (
                    vertex[0],
                    vertex[1] - connectorDistanceBelowMidline,
                    vertex[2],
                )

            return verticesTop, verticesBottom

        totalVerts = []
        totalFaces = []
        frontVerticesTop, facesFrontTop = endFaceTop(
            resolution=resolution,
            topRadii=topRadii,
            bottomRadii=bottomRadii,
            fingerBodyHeight=fingerBodyHeight,
            fingerBodyWidth=fingerBodyWidth,
            holeRadius=stringHoleRadius,
        )
        frontVerticesBottom = endFaceBottom(
            resolution=resolution,
            topRadii=topRadii,
            bottomRadii=bottomRadii,
            fingerBodyHeight=fingerBodyHeight,
            fingerBodyWidth=fingerBodyWidth,
            holeRadius=stringHoleRadius,
        )
        vertsLeftTop, vertsLeftBottom = createConnector(
            endRadius=connectorRadius,
            length=connectorLength,
            thickness=connectorThickness,
            connectorDistanceBelowMidline=connectorDistanceBelowMidline,
        )
        for i, vertex in enumerate(vertsLeftTop):
            vertsLeftTop[i] = (vertex[0] - fingerBodyWidth / 2, vertex[1], vertex[2])
        for i, vertex in enumerate(vertsLeftBottom):
            vertsLeftBottom[i] = (vertex[0] - fingerBodyWidth / 2, vertex[1], vertex[2])
        vertsRightTop, vertsRightBottom = createConnector(
            endRadius=connectorRadius,
            length=connectorLength,
            thickness=connectorThickness,
            connectorDistanceBelowMidline=connectorDistanceBelowMidline,
        )
        for i, vertex in enumerate(vertsRightTop):
            vertsRightTop[i] = (vertex[0] + fingerBodyWidth / 2, vertex[1], vertex[2])
        for i, vertex in enumerate(vertsRightBottom):
            vertsRightBottom[i] = (
                vertex[0] + fingerBodyWidth / 2,
                vertex[1],
                vertex[2],
            )
        pl = pv.Plotter()
        totalVerts.extend(frontVerticesTop)
        totalVerts.extend(frontVerticesBottom)
        totalFaces.extend(facesFrontTop)
        backWall = []
        for vertice in frontVerticesTop:
            backWall.append((vertice[0], vertice[1], fingerBodyLength / 2))
        for vertice in frontVerticesBottom:
            backWall.append((vertice[0], vertice[1], fingerBodyLength / 2))
        totalVerts.extend(backWall)
        offset = len(frontVerticesBottom) + len(frontVerticesTop)

        for idx in range(1, len(frontVerticesTop) - 1):
            if idx != resolution + 1:
                totalFaces.append((3, idx, idx + 1, idx + offset))
                totalFaces.append((3, idx + 1, idx + offset + 1, idx + offset))

        offset = len(frontVerticesTop)
        for idx in range(0, len(frontVerticesBottom) - 1):
            totalFaces.append(
                (
                    3,
                    len(frontVerticesBottom) - 1 + offset,
                    idx + 1 + offset,
                    idx + offset,
                )
            )
        offset1 = len(frontVerticesTop)
        offset2 = 2 * len(frontVerticesTop) + len(frontVerticesBottom)
        for idx in range(0, len(frontVerticesBottom) - 1):
            totalFaces.append((3, idx + offset1, idx + 1 + offset1, idx + offset2))
            totalFaces.append((3, idx + 1 + offset1, idx + offset2 + 1, idx + offset2))

        offset = len(totalVerts)

        totalVerts.extend(vertsLeftTop)
        totalVerts.extend(vertsLeftBottom)
        totalVerts.extend(vertsRightTop)
        totalVerts.extend(vertsRightBottom)

        totalFaces.append((3, 0, offset + len(vertsLeftTop) + len(vertsLeftBottom), 1))
        totalFaces.append(
            (3, offset, offset + len(vertsLeftTop) + len(vertsLeftBottom), 0)
        )
        totalFaces.append((3, resolution + 1, offset, 0))

        totalFaces.append(
            (
                3,
                len(frontVerticesTop),
                offset + 2 * len(vertsLeftTop) + 2 * len(vertsLeftBottom) - 1,
                len(frontVerticesTop) + len(frontVerticesBottom) - 1,
            )
        )
        totalFaces.append(
            (
                3,
                offset + len(vertsLeftTop) + len(vertsLeftBottom) - 1,
                offset + 2 * len(vertsLeftTop) + 2 * len(vertsLeftBottom) - 1,
                len(frontVerticesTop) + len(frontVerticesBottom) - 1,
            )
        )

        for idx in range(len(vertsLeftTop) + len(vertsLeftBottom) - 1):
            totalFaces.append(
                (
                    3,
                    idx + offset,
                    idx + 1 + offset,
                    idx + offset + len(vertsLeftTop) + len(vertsLeftBottom) + 1,
                )
            )
            totalFaces.append(
                (
                    3,
                    idx + offset,
                    idx + offset + len(vertsLeftTop) + len(vertsLeftBottom),
                    idx + offset + len(vertsLeftTop) + len(vertsLeftBottom) + 1,
                )
            )

        # SIDEWALLS
        otherSide = len(vertsLeftTop) + len(vertsLeftBottom)
        for idx in range(1, int(len(vertsLeftTop) / 2) + 1):
            totalFaces.append((3, offset + idx - 1, resolution + 1, offset + idx))
            totalFaces.append(
                (3, offset + idx - 1 + otherSide, 1, offset + idx + otherSide)
            )
        totalFaces.append(
            (
                3,
                resolution + 1,
                offset + int(len(vertsLeftTop) / 2),
                len(frontVerticesTop) + len(frontVerticesBottom) + resolution + 1,
            )
        )
        totalFaces.append(
            (
                3,
                1,
                offset + int(len(vertsLeftTop) / 2) + otherSide,
                len(frontVerticesTop) + len(frontVerticesBottom) + 1,
            )
        )
        for idx in range(int(len(vertsLeftTop) / 2) + 1, len(vertsLeftTop)):
            totalFaces.append(
                (
                    3,
                    offset + idx - 1,
                    len(frontVerticesTop) + len(frontVerticesBottom) + resolution + 1,
                    offset + idx,
                )
            )
            totalFaces.append(
                (
                    3,
                    offset + idx - 1 + otherSide,
                    len(frontVerticesTop) + len(frontVerticesBottom) + 1,
                    offset + idx + otherSide,
                )
            )
        totalFaces.append(
            (
                3,
                len(frontVerticesTop) + len(frontVerticesBottom) + resolution + 1,
                offset + len(vertsLeftTop) - 1,
                2 * len(frontVerticesTop) + 2 * len(frontVerticesBottom) - 1,
            )
        )
        totalFaces.append(
            (
                3,
                len(frontVerticesTop) + len(frontVerticesBottom) + 1,
                offset + len(vertsLeftTop) - 1 + otherSide,
                2 * len(frontVerticesTop) + len(frontVerticesBottom),
            )
        )
        # BOTTOM HALF
        for idx in range(
            len(vertsLeftTop), len(vertsLeftTop) + int(len(vertsLeftBottom) / 2)
        ):
            totalFaces.append(
                (
                    3,
                    offset + idx - 1,
                    2 * len(frontVerticesTop) + 2 * len(frontVerticesBottom) - 1,
                    offset + idx,
                )
            )
            totalFaces.append(
                (
                    3,
                    offset + idx - 1 + otherSide,
                    2 * len(frontVerticesTop) + len(frontVerticesBottom),
                    offset + idx + otherSide,
                )
            )
        totalFaces.append(
            (
                3,
                2 * len(frontVerticesTop) + 2 * len(frontVerticesBottom) - 1,
                offset + len(vertsLeftTop) + int(len(vertsLeftBottom) / 2) - 1,
                len(frontVerticesTop) + len(frontVerticesBottom) - 1,
            )
        )
        totalFaces.append(
            (
                3,
                2 * len(frontVerticesTop) + len(frontVerticesBottom),
                offset
                + len(vertsLeftTop)
                + int(len(vertsLeftBottom) / 2)
                - 1
                + otherSide,
                len(frontVerticesTop),
            )
        )
        for idx in range(
            len(vertsLeftTop) + int(len(vertsLeftBottom) / 2),
            len(vertsLeftTop) + len(vertsLeftBottom),
        ):
            totalFaces.append(
                (
                    3,
                    offset + idx - 1,
                    len(frontVerticesTop) + len(frontVerticesBottom) - 1,
                    offset + idx,
                )
            )
            totalFaces.append(
                (
                    3,
                    offset + idx - 1 + otherSide,
                    len(frontVerticesTop),
                    offset + idx + otherSide,
                )
            )
        for idx, point in enumerate(totalVerts):
            totalVerts[idx] = np.array(point) - np.array([0, 0, fingerBodyLength / 2])
        mesh = pv.PolyData(totalVerts, totalFaces).clean().triangulate()
        half2 = mesh.reflect((0, 0, 1), point=(0, 0, 0))
        merged = mesh.merge(half2)
        cleanedMerge = merged.clean().compute_normals()
        return cleanedMerge

    def generateFingersNoData(self):
        self.clearOldFingers()
        for jointListIdx in range(len(self.jointsList)):
            tempMeshList = []
            jointList = self.jointsList[jointListIdx]
            jointNeighborIndex = jointListIdx + 1
            if jointListIdx == len(self.jointsList) - 1:
                jointNeighborIndex = jointListIdx - 1
            jointNeighborList = self.jointsList[jointNeighborIndex]
            central = np.array(self.rootSphere.center)
            u = np.array(list(jointList.keys())[1].center) - central
            v = np.array(list(jointNeighborList.keys())[1].center) - central
            normal = np.cross(u, v)
            if jointListIdx == len(self.jointsList) - 1:
                normal *= -1
            for jointIdx in range(1, len(self.jointsList[jointListIdx])):
                pointEnd = np.array(list(jointList.keys())[jointIdx].center)
                pointStart = np.array(list(jointList.keys())[jointIdx - 1].center)
                vector = pointEnd - pointStart
                midpoint = (pointStart + pointEnd) / 2
                fingerBodyLength = np.linalg.norm(vector)
                direction = vector / fingerBodyLength

                topRadii = 5
                bottomRadii = 4
                fingerBodyHeight = 12
                fingerBodyWidth = 14
                resolution = 9
                holeRadius = 1

                part = self.generateFingerPart(
                    resolution=resolution,
                    topRadii=topRadii,
                    bottomRadii=bottomRadii,
                    fingerBodyHeight=fingerBodyHeight,
                    fingerBodyWidth=fingerBodyWidth,
                    holeRadius=holeRadius,
                    fingerBodyLength=fingerBodyLength,
                )
                part = self.moveAlignMesh(part, midpoint, direction, normal)
                vtk_polydata = part.extract_geometry()
                mapper = vtk.vtkPolyDataMapper()

                mapper.SetInputData(vtk_polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                tempMeshList.append(actor)
            self.fingerActors.append(tempMeshList)

            resolutionEnd = 10
            resolutionBody = 10
            thickness = 1.3
            endRadius = 1.7
            length = 5.8
            width = fingerBodyWidth

            tempMeshList = []
            for jointIdx in range(1, len(self.jointsList[jointListIdx])):
                pointEnd = np.array(list(jointList.keys())[jointIdx].center)
                pointStart = np.array(list(jointList.keys())[jointIdx - 1].center)
                vector = pointEnd - pointStart
                direction = vector / np.linalg.norm(vector)

                part = self.genConnector(
                    endRadius=endRadius,
                    length=length,
                    thickness=thickness,
                    resolutionBody=resolutionBody,
                    resolutionEnd=resolutionEnd,
                    width=width + 1,
                )
                part = self.moveAlignMesh(
                    part,
                    pointStart,
                    direction,
                    normal,
                )
                vtk_polydata = part.extract_geometry()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(vtk_polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                tempMeshList.append(actor)
            self.fingerConnectorActors.append(tempMeshList)
        # self.cutConnectors()
        self.displayFingerMeshes()

    def displayFingerMeshes(self):
        for finger in self.fingerActors:
            for part in finger:
                self.renderer.AddActor(part)
        for finger in self.fingerConnectorActors:
            for connector in finger:
                self.renderer.AddActor(connector)
        self.renderWindow.Render()

    def generateFingerPart(
        self,
        resolution,
        topRadii,
        bottomRadii,
        fingerBodyHeight,
        fingerBodyWidth,
        holeRadius,
        fingerBodyLength,
    ):
        def endFace(
            resolution,
            topRadii,
            bottomRadii,
            fingerBodyHeight,
            fingerBodyWidth,
            holeRadius,
        ):
            newPoints = []
            faces = []
            thetaInterval = np.pi / (resolution)
            for i in range(2):
                batch = []
                theta = 0
                while theta < np.pi / 2:
                    radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2 - bottomRadii:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                -radius * np.sin(theta),
                                0,
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                -radius * np.sin(theta),
                                0,
                            )
                        )
                        theta += thetaInterval
                    radius = fingerBodyHeight / 2 / np.sin(theta)
                    while radius * np.cos(theta) > 0:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                -radius * np.sin(theta),
                                0,
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyHeight / 2 / np.sin(theta)
                if i % 2 == 1:
                    batch.reverse()
                    batch.pop(0)
                newPoints.extend(batch)
            newPoints.pop(0)
            newPoints.pop()
            newPoints.append((0, 0, 0))
            # ADJUSTED
            for p in range(len(newPoints) - 1):
                faces.append((3, p + 1, len(newPoints) - 1, p))
            for i in range(2):
                batch = []
                theta = 0
                while theta < np.pi / 2:
                    radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2 - topRadii:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                radius * np.sin(theta),
                                radius * np.sin(theta),
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyWidth / 2 / np.cos(theta)
                    while radius * np.sin(theta) < fingerBodyHeight / 2:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                radius * np.sin(theta),
                                radius * np.sin(theta),
                            )
                        )
                        theta += thetaInterval
                    radius = fingerBodyHeight / 2 / np.sin(theta)
                    while radius * np.cos(theta) > 0:
                        batch.append(
                            (
                                radius * np.cos(theta) * ((-1) ** i),
                                radius * np.sin(theta),
                                radius * np.sin(theta),
                            )
                        )
                        theta += thetaInterval
                        radius = fingerBodyHeight / 2 / np.sin(theta)
                if i % 2 == 1:
                    batch.reverse()
                    batch.pop(0)
                newPoints.extend(batch)
            for i in range(resolution + 1):
                theta = 2 * np.pi / resolution * i + (3 * np.pi / 2)
                newPoints.append(
                    (
                        holeRadius * np.cos(theta),
                        holeRadius * np.sin(theta) + fingerBodyHeight / 4,
                        holeRadius * np.sin(theta) + fingerBodyHeight / 4,
                    )
                )
            offset = resolution - 2
            # adjusted
            faces.append((3, resolution + offset, offset, offset - 1))
            faces.append((3, 0, offset, offset + 1))
            # adjusted
            for i in range(resolution + 1):
                faces.append(
                    (
                        3,
                        (i) % (resolution + 1) + resolution + offset + 1,
                        (i + 1) % (resolution + 1) + offset,
                        i + offset,
                    )
                )
                faces.append(
                    (
                        3,
                        (i) % (resolution + 1) + resolution + offset + 1,
                        (i + 1) % (resolution + 1) + resolution + offset + 1,
                        (i + 1) % (resolution + 1) + offset,
                    )
                )
            return (np.array(newPoints), faces)

        faces = []
        totalVertices = []
        frontVertices, facesFront = endFace(
            resolution=resolution,
            topRadii=topRadii,
            bottomRadii=bottomRadii,
            fingerBodyHeight=fingerBodyHeight,
            fingerBodyWidth=fingerBodyWidth,
            holeRadius=holeRadius,
        )
        backVertices, facesBack = endFace(
            resolution=resolution,
            topRadii=topRadii,
            bottomRadii=bottomRadii,
            fingerBodyHeight=fingerBodyHeight,
            fingerBodyWidth=fingerBodyWidth,
            holeRadius=holeRadius,
        )
        adjustedFrontFaceVertices = []
        for vertice in frontVertices:
            adjustedFrontFaceVertices.append(
                (vertice[0], vertice[1], vertice[2] - fingerBodyLength / 2)
            )
        totalVertices.extend(adjustedFrontFaceVertices)
        faces.extend(facesFront)
        adjustedBackFaceVertices = []
        for vertice in backVertices:
            adjustedBackFaceVertices.append(
                (vertice[0], vertice[1], -1 * vertice[2] + fingerBodyLength / 2)
            )
        adjustedBackFaces = []
        # adjusted
        for idx in range(len(facesBack)):
            newFace = (
                3,
                facesBack[idx][3] + len(frontVertices),
                facesBack[idx][2] + len(frontVertices),
                facesBack[idx][1] + len(frontVertices),
            )
            adjustedBackFaces.append(newFace)
        totalVertices.extend(adjustedBackFaceVertices)
        faces.extend(adjustedBackFaces)
        # ADJUSTED
        for idx in range(resolution - 3):
            faces.append([3, idx + len(frontVertices), idx + 1, idx])
            faces.append(
                [3, idx + len(frontVertices), idx + len(frontVertices) + 1, idx + 1]
            )
        offset = resolution - 1
        # ADJUSTED
        for idx in range(resolution - 1):
            faces.append(
                [3, idx + offset, idx + offset + 1, idx + offset + len(frontVertices)]
            )
            faces.append(
                [
                    3,
                    idx + offset + 1,
                    idx + offset + len(frontVertices) + 1,
                    idx + offset + len(frontVertices),
                ]
            )
        # adjusted
        faces.append([3, len(frontVertices) + resolution - 1, len(frontVertices), 0])
        faces.append([3, resolution - 1, len(frontVertices) + resolution - 1, 0])
        faces.append(
            [
                3,
                resolution - 3,
                len(frontVertices) + resolution - 3,
                len(frontVertices) + resolution + resolution - 2,
            ]
        )
        faces.append(
            [
                3,
                resolution - 3,
                len(frontVertices) + resolution + resolution - 2,
                resolution + resolution - 2,
            ]
        )

        offset = resolution + resolution - 1
        for i in range(resolution + 1):
            faces.append(
                [
                    3,
                    len(frontVertices) + offset + (i % (resolution + 1)),
                    offset + ((i + 1) % (resolution + 1)),
                    offset + i,
                ]
            )
            faces.append(
                [
                    3,
                    offset + ((i + 1) % (resolution + 1)),
                    len(frontVertices) + offset + (i % (resolution + 1)),
                    len(frontVertices) + offset + ((i + 1) % (resolution + 1)),
                ]
            )
        mesh = pv.PolyData(totalVertices, faces).triangulate()
        return mesh

    def moveAlignMesh(self, mesh, newCenter, newAlignVector, newNormal):
        otherNormal = np.cross(newAlignVector, newNormal)
        v1 = otherNormal / np.linalg.norm(otherNormal)
        v2 = newNormal / np.linalg.norm(newNormal) * -1
        v3 = newAlignVector / np.linalg.norm(newAlignVector)
        if (
            not np.allclose(np.dot(v1, v2), 0)
            or not np.allclose(np.dot(v1, v3), 0)
            or not np.allclose(np.dot(v2, v3), 0)
        ):
            raise ValueError("Basis vectors are not orthogonal")
        transformation_matrix = np.column_stack((v1, v2, v3))
        finalTransformation = np.eye(4)
        finalTransformation[0:3, 0:3] = transformation_matrix
        mesh = mesh.transform(finalTransformation)
        translation_vector = newCenter - np.array([0, 0, 0], dtype=float)
        mesh = mesh.translate(translation_vector)
        return mesh

    def genConnector(
        self, endRadius, length, thickness, resolutionBody, resolutionEnd, width
    ):
        def createConnector(
            endRadius, length, thickness, resolutionBody, resolutionEnd
        ):
            vertices = []
            faces = []

            for i in range(resolutionBody + 1):
                vertices.append(
                    (0, thickness / 2, length / resolutionBody * i - length / 2)
                )
            for i in range(resolutionBody + 1):
                vertices.append(
                    (0, -thickness / 2, length / resolutionBody * i - length / 2)
                )
            for i in range(resolutionBody):
                faces.append((3, i, i + 1, i + 1 + resolutionBody + 1))
                faces.append((3, i + resolutionBody + 2, i + resolutionBody + 1, i))

            currTheta = np.pi - np.arcsin(thickness / 2 / endRadius)
            thetaInterval = 2 * currTheta / (resolutionEnd + 1)
            currTheta -= thetaInterval
            for _ in range(resolutionEnd):
                vertices.append(
                    (
                        0,
                        endRadius * np.sin(currTheta),
                        endRadius * np.cos(currTheta) + length / 2 + endRadius,
                    )
                )
                currTheta -= thetaInterval

            currTheta = np.arcsin(thickness / 2 / endRadius)
            thetaInterval = 2 * (np.pi - currTheta) / (resolutionEnd + 1)
            currTheta -= thetaInterval * 2
            for _ in range(resolutionEnd):
                vertices.append(
                    (
                        0,
                        endRadius * np.sin(currTheta),
                        endRadius * np.cos(currTheta) - length / 2 - endRadius,
                    )
                )
                currTheta -= thetaInterval

            endCenter1 = (0, 0, (length / 2 + endRadius))
            vertices.append(endCenter1)
            endCenter2 = (0, 0, (-length / 2 - endRadius))
            vertices.append(endCenter2)
            for i in range(resolutionEnd - 1):
                faces.append(
                    (
                        3,
                        i + 2 * resolutionBody + 2,
                        len(vertices) - 2,
                        i + 2 * resolutionBody + 3,
                    )
                )
                faces.append(
                    (
                        3,
                        i + 2 * resolutionBody + 2 + resolutionEnd,
                        len(vertices) - 1,
                        i + 2 * resolutionBody + 3 + resolutionEnd,
                    )
                )

            faces.append((3, len(vertices) - 1, 0, resolutionBody + 1))
            faces.append((3, len(vertices) - 3, 0, len(vertices) - 1))
            faces.append(
                (
                    3,
                    len(vertices) - 1,
                    resolutionBody + 1,
                    len(vertices) - 2 - resolutionEnd,
                )
            )

            faces.append((3, len(vertices) - 2, resolutionBody, 2 * resolutionBody + 1))
            faces.append(
                (
                    3,
                    len(vertices) - 2,
                    2 * resolutionBody + 1,
                    2 * resolutionBody + 1 + resolutionEnd,
                )
            )
            faces.append((3, len(vertices) - 2, resolutionBody, 2 * resolutionBody + 2))
            return vertices, faces

        totalVertices = []
        totalFaces = []
        verts1, faces1 = createConnector(
            endRadius=endRadius,
            length=length,
            thickness=thickness,
            resolutionBody=resolutionBody,
            resolutionEnd=resolutionEnd,
        )
        for vert in verts1:
            totalVertices.append((width / 2, vert[1], vert[2]))
        totalFaces.extend(faces1)
        verts2, faces2 = createConnector(
            endRadius=endRadius,
            length=length,
            thickness=thickness,
            resolutionBody=resolutionBody,
            resolutionEnd=resolutionEnd,
        )
        for vert in verts2:
            totalVertices.append((-width / 2, vert[1], vert[2]))
        for face in faces2:
            totalFaces.append(
                (3, face[1] + len(verts1), face[2] + len(verts1), face[3] + len(verts1))
            )
        for vert in range(resolutionBody):
            totalFaces.append((3, vert, (vert + 1), len(verts1) + (vert + 1)))
            totalFaces.append((3, len(verts1) + (vert + 1), len(verts1) + vert, vert))
            totalFaces.append(
                (
                    3,
                    vert + resolutionBody + 1,
                    (vert + 1) + resolutionBody + 1,
                    len(verts1) + (vert + 1) + resolutionBody + 1,
                )
            )
            totalFaces.append(
                (
                    3,
                    len(verts1) + (vert + 1) + resolutionBody + 1,
                    len(verts1) + vert + resolutionBody + 1,
                    vert + resolutionBody + 1,
                )
            )
        for i in range(resolutionEnd - 1):
            totalFaces.append(
                (
                    3,
                    i + 2 * resolutionBody + 2,
                    i + 2 * resolutionBody + 3,
                    i + 2 * resolutionBody + 2 + len(verts1),
                )
            )
            totalFaces.append(
                (
                    3,
                    i + 2 * resolutionBody + 3 + len(verts1),
                    i + 2 * resolutionBody + 2 + len(verts1),
                    i + 2 * resolutionBody + 3,
                )
            )

            totalFaces.append(
                (
                    3,
                    i + 2 * resolutionBody + 2 + resolutionEnd,
                    i + 2 * resolutionBody + 3 + resolutionEnd,
                    i + 2 * resolutionBody + 2 + len(verts1) + resolutionEnd,
                )
            )
            totalFaces.append(
                (
                    3,
                    i + 2 * resolutionBody + 3 + len(verts1) + resolutionEnd,
                    i + 2 * resolutionBody + 2 + len(verts1) + resolutionEnd,
                    i + 2 * resolutionBody + 3 + resolutionEnd,
                )
            )

        totalFaces.append(
            (
                3,
                2 * resolutionBody + 2,
                len(verts1) + resolutionBody,
                2 * resolutionBody + 2 + len(verts1),
            )
        )
        totalFaces.append(
            (3, 2 * resolutionBody + 2, resolutionBody, len(verts1) + resolutionBody)
        )

        totalFaces.append(
            (
                3,
                2 * resolutionBody + 2 + resolutionEnd - 1 + len(verts1),
                len(verts1) + 2 * resolutionBody + 1,
                2 * resolutionBody + 2 + resolutionEnd - 1,
            )
        )
        totalFaces.append(
            (
                3,
                2 * resolutionBody + 1,
                len(verts1) + 2 * resolutionBody + 1,
                2 * resolutionBody + 2 + resolutionEnd - 1,
            )
        )

        totalFaces.append(
            (3, len(verts1), 0, 2 * resolutionBody + resolutionEnd + resolutionEnd + 1)
        )
        totalFaces.append(
            (
                3,
                2 * resolutionBody + resolutionEnd + resolutionEnd + 1 + len(verts1),
                len(verts1),
                2 * resolutionBody + resolutionEnd + resolutionEnd + 1,
            )
        )

        totalFaces.append(
            (
                3,
                resolutionBody + 1,
                len(verts1) + resolutionBody + 1,
                2 * resolutionBody + 2 + resolutionEnd,
            )
        )
        totalFaces.append(
            (
                3,
                len(verts1) + 2 * resolutionBody + 2 + resolutionEnd,
                len(verts1) + resolutionBody + 1,
                2 * resolutionBody + 2 + resolutionEnd,
            )
        )
        offset = 1.5
        for i, point in enumerate(totalVertices):
            totalVertices[i] = (point[0], point[1] - offset, point[2])
        return pv.PolyData(totalVertices, totalFaces).compute_normals()


app = Qt.QApplication(sys.argv)
window = GUI()
sys.exit(app.exec_())
