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
        app.setStyleSheet("QWidget { font-size: 14px; }")

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

        # Set the main widget of the QMainWindow to use the mainLayout
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
        self.checkboxes = {}
        vbox = Qt.QVBoxLayout()

        label = Qt.QLabel("Index:")
        vbox.addWidget(label)
        for i, sphere in enumerate(self.handManipulator.indexJoints.keys()):
            cb = QtWidgets.QCheckBox(sphere.name)
            vbox.addWidget(cb)
            cb.stateChanged.connect(self.toggleJointInteraction)
            self.checkboxes[cb] = sphere

        label = Qt.QLabel("Middle:")
        vbox.addWidget(label)
        for i, sphere in enumerate(self.handManipulator.middleJoints.keys()):
            cb = QtWidgets.QCheckBox(sphere.name)
            vbox.addWidget(cb)
            cb.stateChanged.connect(self.toggleJointInteraction)
            self.checkboxes[cb] = sphere

        label = Qt.QLabel("Third:")
        vbox.addWidget(label)
        for i, sphere in enumerate(self.handManipulator.thirdJoints.keys()):
            cb = QtWidgets.QCheckBox(sphere.name)
            vbox.addWidget(cb)
            cb.stateChanged.connect(self.toggleJointInteraction)
            self.checkboxes[cb] = sphere

        label = Qt.QLabel("Fourth:")
        vbox.addWidget(label)
        for i, sphere in enumerate(self.handManipulator.fourthJoints.keys()):
            cb = QtWidgets.QCheckBox(sphere.name)
            vbox.addWidget(cb)
            cb.stateChanged.connect(self.toggleJointInteraction)
            self.checkboxes[cb] = sphere

        genFingers = QtWidgets.QPushButton("Generate Fingers")
        vbox.addWidget(genFingers)
        genFingers.clicked.connect(self.genFingers)

        saveFingerPos = QtWidgets.QPushButton("Save Finger Positions")
        vbox.addWidget(saveFingerPos)
        saveFingerPos.clicked.connect(self.saveFingerPositions)

        label = Qt.QLabel("Resize Hand:")
        vbox.addWidget(label)
        handResizer = QtWidgets.QSlider()
        handResizer.setGeometry(QtCore.QRect(190, 100, 160, 16))
        handResizer.setOrientation(QtCore.Qt.Horizontal)
        handResizer.sliderMoved.connect(self.resizer)
        vbox.addWidget(handResizer)

        # HANDMESHSTUFF
        label = Qt.QLabel("Socket Tools:")
        vbox.addWidget(label)
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

        content_widget = Qt.QWidget(self)
        content_widget.setLayout(vbox)
        scroll_area = Qt.QScrollArea(self)
        scroll_area.setWidget(content_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)

        mainLayout.addWidget(scroll_area)

        label = Qt.QLabel("Change Paint Density:")
        vbox.addWidget(label)
        density = QtWidgets.QSlider()
        density.setGeometry(QtCore.QRect(190, 100, 160, 16))
        density.setOrientation(QtCore.Qt.Horizontal)
        density.sliderMoved.connect(self.densityChanger)
        vbox.addWidget(density)

        genSocketButton = QtWidgets.QPushButton("Generate Socket")
        vbox.addWidget(genSocketButton)
        genSocketButton.clicked.connect(self.generateSocket)

        testSocketButton = QtWidgets.QPushButton("test Socket")
        vbox.addWidget(testSocketButton)
        testSocketButton.clicked.connect(self.testSocket)

    def socketToolSelected(self, state):
        self.handMesh.setTool(state)

    def generateSocket(self):
        self.handMesh.generateSocket()

    def genFingers(self):
        self.handManipulator.generateFingers()

    def saveFingerPositions(self):
        self.handManipulator.saveFingerPositions()

    def resizer(self, p):
        self.handMesh.resize(p)

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
        self.handMesh.genHandPortion(
            self.handManipulator.getJoints(), self.handManipulator.getBaseFingerMeshes()
        )


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

    def genHandPortion(self, carpals, baseFingerMeshes):
        handMesh = pv.wrap(self.actor.GetMapper().GetInput())
        usedPointIDs = [0 for _ in range(self.finalSocket.GetNumberOfPoints())]
        socket = pv.wrap(self.finalSocket)
        points = self.finalSocket.GetPoints()

        def selectPoints(
            fingerVector: list,
            fingerJointOrigin: list,
            radius: int,
            color=(0.3, 0.2, 1),
            size=5,
        ):
            point, idx = socket.ray_trace(
                fingerJointOrigin,
                [fingerJointOrigin[i] - fingerVector[i] for i in range(3)],
                first_point=True,
            )
            selectingSphere = pv.Sphere(radius=radius, center=point)
            selectedIds = socket.select_enclosed_points(selectingSphere)[
                "SelectedPoints"
            ].view(bool)
            finalSelectedIDs = []

            for i in range(len(selectedIds)):
                if selectedIds[i]:
                    closestPoint = handMesh.points[
                        handMesh.find_closest_point(socket.points[i])
                    ]
                    distance = np.linalg.norm(
                        np.array(socket.points[i]) - np.array(closestPoint)
                    )
                    threshold = 5e-1
                    if distance >= threshold and usedPointIDs[i] == 0:
                        finalSelectedIDs.append(i)
                        usedPointIDs[i] = 1
            extractedPoints = []
            if finalSelectedIDs:
                extractedPoints = socket.extract_points(
                    finalSelectedIDs, adjacent_cells=False
                ).points
            pointsToPlot = extractedPoints
            pls = vtk.vtkPoints()
            for pt in pointsToPlot:
                pls.InsertNextPoint(pt)
            fd = vtk.vtkPolyData()
            fd.SetPoints(pls)
            self.plotPointCloud(fd, color=color, size=size)
            self.renderWindow.Render()

            return finalSelectedIDs, extractedPoints

        def moveSelectedPointsToJoints(pointIDs, fingerVector: list, jointOrigin: list):
            distances = []
            diskOfPoints = []
            for idx in pointIDs:
                p = np.array(points.GetPoint(idx))
                u = p - np.array(jointOrigin)
                n = np.array(fingerVector) / np.linalg.norm(np.array(fingerVector))
                newPos = p - n * np.dot(u, n)
                points.SetPoint(idx, newPos)
                # distances.append(np.linalg.norm(u))
                distances.append(np.linalg.norm(newPos - p))
                diskOfPoints.append(newPos)
            avgDistance = abs(sum(distances) / len(distances))

            circumferenceOfPoints = [[], []]
            for pair in ConvexHull(diskOfPoints, qhull_options="QJ").vertices:
                circumferenceOfPoints[0].append(diskOfPoints[pair])
                circumferenceOfPoints[1].append(distances[pair])
            pointsToPlot = circumferenceOfPoints[0]
            pls = vtk.vtkPoints()
            for pt in pointsToPlot:
                pls.InsertNextPoint(pt)
            fd = vtk.vtkPolyData()
            fd.SetPoints(pls)
            self.plotPointCloud(fd, color=(1, 1, 0.3))
            self.renderWindow.Render()

            return avgDistance, circumferenceOfPoints

        def proportionallyMovePoints(
            fingerVector: list,
            jointOrigin: list,
            avgDistance,
            circumferenceOfPoints,
            initialRadius=5,
            numLayers=8,
        ):
            tree = cKDTree(circumferenceOfPoints[0])
            for j in range(1, numLayers + 1):
                ids, pts = selectPoints(fingerVector, jointOrigin, initialRadius + j)
                for idx in ids:
                    p = np.array(points.GetPoint(idx))
                    u = p - np.array(jointOrigin)
                    dist, closestPointIdx = tree.query(p)
                    distance = circumferenceOfPoints[1][closestPointIdx]
                    # factor = distance/((np.linalg.norm(u)/distance)**2)
                    n = np.array(fingerVector) / np.linalg.norm(np.array(fingerVector))
                    factor = distance / ((np.linalg.norm(u) / distance) ** 2)
                    # newPos = p + np.array(socket.point_normals[idx])*factor
                    newPos = p + n * factor
                    points.SetPoint(idx, newPos)

        for carpal in carpals:
            usedPointIDs = [0 for _ in range(self.finalSocket.GetNumberOfPoints())]
            listOfIds, _ = selectPoints(
                carpal["normal"], carpal["center"], 5, color=(0.5, 1, 1), size=10
            )
            avgDistance, circumferenceOfPoints = moveSelectedPointsToJoints(
                listOfIds, carpal["normal"], carpal["center"]
            )
            proportionallyMovePoints(
                fingerVector=carpal["normal"],
                jointOrigin=carpal["center"],
                avgDistance=avgDistance,
                circumferenceOfPoints=circumferenceOfPoints,
            )
        points.Modified()
        self.finalSocket.Modified()

        smoothFilter = vtk.vtkLoopSubdivisionFilter()
        smoothFilter.SetNumberOfSubdivisions(3)
        smoothFilter.SetInputData(self.finalSocket)
        self.finalSocket = smoothFilter.GetOutput()
        self.finalSocket.Modified()

        # points = np.array(points)
        # tree = cKDTree(points)
        # # Calculate bounding box
        # minCoords = np.min(points, axis=0)
        # maxCoords = np.max(points, axis=0)
        # origin = minCoords
        # n = 100
        # alpha = max(maxCoords - minCoords)
        # spacing = alpha / n
        # grid = pv.ImageData(
        #     dimensions=(n, n, n), spacing=(spacing, spacing, spacing), origin=origin
        # )
        # x, y, z = grid.points.T
        # distances, _ = tree.query(np.c_[x, y, z])
        # values = distances**2

        # pl = pv.Plotter()
        # actor = pl.add_points(points, render_points_as_spheres=False, point_size=100.0)
        # pl.show()

        # isoValues = [
        #     0.3,
        #     0.5,
        #     0.8,
        #     1.0,
        #     2.0,
        #     3.0,
        #     4.0,
        #     5.0,
        #     10.0,
        #     20.0,
        # ]

        # for isoVal in isoValues:
        #     mesh = grid.contour([isoVal], scalars=values, method="flying_edges")
        #     if mesh.n_points > 0:
        #         mesh.plot(smooth_shading=True, cmap="plasma", show_scalar_bar=False)
        #         print(isoVal)
        #         break
        #     else:
        #         pass
        # appendFilter = vtkAppendPolyData()
        # appendFilter.AddInputData(self.finalSocket)
        # for i, finger in enumerate(baseFingerMeshes):
        #     appendFilter.AddInputData(finger)
        # appendFilter.Update()
        # writer = vtk.vtkSTLWriter()
        # writer.SetFileName("imageAnalysisGeneration/totalStruct.stl")
        # writer.SetInputData(appendFilter.GetOutput())
        # writer.Write()

        # writer.SetFileName("imageAnalysisGeneration/socketStruct.stl")
        # writer.SetInputData(self.finalSocket)
        # writer.Write()

        # appendFilter = vtkAppendPolyData()
        # for i, finger in enumerate(baseFingerMeshes):
        #     appendFilter.AddInputData(finger)
        # appendFilter.Update()
        # writer.SetFileName("imageAnalysisGeneration/fingerStruct.stl")
        # writer.SetInputData(appendFilter.GetOutput())
        # writer.Write()


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

    def getBaseFingerMeshes(self):
        baseFingerMeshes = []
        for finger in self.fingerActors:
            baseFingerMeshes.append(finger[0].GetMapper().GetInput())
        return baseFingerMeshes

    def generateFingers(self):
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

                topRadii = 4
                bottomRadii = 3
                fingerBodyHeight = 8
                fingerBodyWidth = 10
                resolution = 9
                holeRadius = 0.5

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
            thickness = 0.5
            endRadius = 1
            length = 4
            width = fingerBodyWidth

            for jointIdx in range(1, len(self.indexJoints)):
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
                    width=width,
                )
                part = self.moveAlignMesh(part, pointStart, direction, normal)
                vtk_polydata = part.extract_geometry()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(vtk_polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                self.fingerConnectorActors.append(actor)
        self.displayFingerMeshes()

    def displayFingerMeshes(self):
        for finger in self.fingerActors:
            for part in finger:
                self.renderer.AddActor(part)
        for fingerConnector in self.fingerConnectorActors:
            self.renderer.AddActor(fingerConnector)
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
            faces.append((3, len(vertices) - 1, 0, len(vertices) - 3))
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
                2 * resolutionBody + 2 + resolutionEnd - 1,
                len(verts1) + 2 * resolutionBody + 1,
                2 * resolutionBody + 2 + resolutionEnd - 1 + len(verts1),
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
        return pv.PolyData(totalVertices, totalFaces)


app = Qt.QApplication(sys.argv)
window = GUI()
sys.exit(app.exec_())
