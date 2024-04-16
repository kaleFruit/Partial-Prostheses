from lib import (
    vtk,
    PlyData,
    np,
    time,
    o3d,
    pv,
    cKDTree,
    tqdm,
    vtk_to_numpy,
    vtkPolyDataBooleanFilter,
    ConvexHull,
    vtkAppendPolyData,
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
        self.actor = self.genHandView("stlfiles/threequarterscalehand.ply")
        self.is_painting = False

        self.brush_radius = 10.0
        self.enclosed_points = vtk.vtkSelectEnclosedPoints()
        self.socketIds = [0] * self.actor.GetMapper().GetInput().GetNumberOfCells()
        self.holesInCells = [0] * self.actor.GetMapper().GetInput().GetNumberOfCells()
        self.thickness = 2

        self.fromPreviousDesign = True
        self.initTime = 0

        self.finalSocket = None
        self.finalSocketHard = None
        self.finalSocketHardActor = None
        self.clipPlaneOriginY = 55
        self.clipPlaneSource = self.createClipPlane()

    def createClipPlane(self):
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetCenter(0, self.clipPlaneOriginY, 0.0)
        planeSource.SetNormal(0, 1, 0)
        size = 25
        planeSource.SetOrigin(-size, self.clipPlaneOriginY, -size)
        planeSource.SetPoint1(size, self.clipPlaneOriginY, -size)
        planeSource.SetPoint2(-size, self.clipPlaneOriginY, size)
        planeSource.Update()
        plane = planeSource.GetOutput()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(plane)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.colors.GetColor3d("Banana"))
        actor.GetProperty().SetOpacity(0.4)
        self.renderer.AddActor(actor)
        return planeSource

    def getHandMesh(self):
        return self.actor.GetMapper().GetInput()

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

    def changePlaneHeight(self, factor):
        self.clipPlaneOriginY = factor
        self.clipPlaneSource.SetCenter(0, self.clipPlaneOriginY, 0)
        self.clipPlaneSource.Update()
        self.renderWindow.Render()

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

    def getSocketThickness(self):
        return self.thickness

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
            reader.SetFileName("oldDesigns/4thGenSoft.vtk")
            reader.Update()
            self.finalSocket = reader.GetOutput()

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName("oldDesigns/4thGenHard.vtk")
            reader.Update()
            self.finalSocketHard = reader.GetOutput()
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

            duplicateSocketShell = vtk.vtkPolyData()
            duplicateSocketShell.DeepCopy(socketShell)
            duplicateSocketShell.Modified()

            self.finalSocketHard = socketShell
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName("oldDesigns/4thGenHard.vtk")
            writer.SetInputData(self.finalSocketHard)
            writer.Write()

            self.finalSocket = self.extrusion(duplicateSocketShell)
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName("oldDesigns/4thGenSoft.vtk")
            writer.SetInputData(self.finalSocket)
            writer.Write()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName("finalSocket.stl")
        writer.SetInputData(self.finalSocket)
        writer.Write()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.finalSocket)
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

    def smoothBoundary(self, boundaryLoop, pvPolydata):
        nSmooth = 70
        for _ in range(nSmooth):
            for i in range(len(boundaryLoop)):
                currPoint = pvPolydata.points[boundaryLoop[i]]
                lastPoint = pvPolydata.points[boundaryLoop[i - 1]]
                nextPoint = pvPolydata.points[boundaryLoop[(i + 1) % len(boundaryLoop)]]
                pvPolydata.points[boundaryLoop[i]] = (
                    0.5 * currPoint + 0.25 * lastPoint + 0.25 * nextPoint
                )
        return pvPolydata

    def extrusion(self, polydata):
        boundaryExtractor = BoundaryExtractor(polydata)
        boundaryLoops = boundaryExtractor.produceOrderedLoops()
        pvPolydata = pv.wrap(polydata)
        for boundaryLoop in boundaryLoops:
            pvPolydata = self.smoothBoundary(boundaryLoop, pvPolydata)
            # print(", ".join([str(list(x)) for x in things]))
        pvPolydata = pvPolydata.smooth_taubin(n_iter=100)
        polydata.DeepCopy(pvPolydata)
        polydata.Modified()

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

        connector = self.createFullConnector(
            width=connectorWidth + 5,
            endRadius=connectorRadius,
            length=connectorLength,
            thickness=connectorThickness,
            resolutionEnd=resolutionEnd,
        )

        return connector

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

    def removeNonManifoldPartsOfMesh(self, mesh):
        mesh = mesh.triangulate()
        nonManifoldEdges = mesh.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=True,
            feature_edges=False,
            manifold_edges=False,
        )
        manifoldEdges = mesh.extract_feature_edges(
            boundary_edges=False,
            non_manifold_edges=False,
            feature_edges=False,
            manifold_edges=True,
        )

        og = mesh.points
        extracted = nonManifoldEdges.points
        pointMapping = {tuple(point): i for i, point in enumerate(og)}
        remapped1 = set([pointMapping[tuple(point)] for point in extracted])

        extracted2 = manifoldEdges.points
        remapped2 = set([pointMapping[tuple(point)] for point in extracted2])
        exclusiveNonManifoldPointIDs = list(remapped1.difference(remapped2))
        fixedMesh, _ = mesh.remove_points(exclusiveNonManifoldPointIDs)
        return fixedMesh

    def extrudeHardSocket(self, polydata):
        boundaryExtractor = BoundaryExtractor(polydata)
        boundaryLoops = boundaryExtractor.produceOrderedLoops()
        pvPolydata = pv.wrap(polydata)
        for boundaryLoop in boundaryLoops:
            pvPolydata = self.smoothBoundary(boundaryLoop, pvPolydata)
            # print(", ".join([str(list(x)) for x in things]))
        pvPolydata = pvPolydata.smooth_taubin(n_iter=50)
        polydata.DeepCopy(pvPolydata)
        polydata.Modified()
        totalBoundaryPoints = vtk.vtkPoints()
        for idx in [pt for loop in boundaryLoops for pt in loop]:
            point = polydata.GetPoint(idx)
            totalBoundaryPoints.InsertNextPoint(point)

        normals = vtk.vtkFloatArray()
        normals.SetNumberOfComponents(3)
        newPoints = vtk.vtkPoints()
        offsetPoints = vtk.vtkPoints()
        for i in range(polydata.GetNumberOfPoints()):
            old_point = polydata.GetPoint(i)
            direction = polydata.GetPointData().GetNormals().GetTuple(i)
            magnitude = (sum([d * d for d in direction])) ** 0.5
            direction = [-d / magnitude for d in direction]
            new_point = [
                old_point[j] + 2 * self.thickness * direction[j] for j in range(3)
            ]
            newPoints.InsertNextPoint(new_point)
            normals.InsertNextTuple(direction)

            offsetPoints.InsertNextPoint(
                [old_point[j] + self.thickness * direction[j] for j in range(3)]
            )
        polydata.SetPoints(offsetPoints)
        polydata.Modified()

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

        return finalData

    def genHardSocket(self, carpals, fingerMeshes, fingerInfo):
        baseSocket = pv.wrap(self.finalSocketHard)
        planeDirection = np.array([0, 1, 0])

        testPlane = pv.Plane(
            direction=self.clipPlaneSource.GetNormal(),
            center=self.clipPlaneSource.GetCenter(),
            i_size=500,
            j_size=500,
        )

        newSurface, _ = baseSocket.clip(
            normal=-1 * planeDirection,
            origin=self.clipPlaneSource.GetCenter(),
            return_clipped=True,
        )
        newSurface.cell_data.clear()

        self.finalSocketHard = self.extrudeHardSocket(newSurface)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.finalSocketHard)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(self.colors.GetColor3d("Blue"))
        actor.PickableOff()
        actor.GetProperty().SetOpacity(0.5)
        # actor.GetProperty().SetRepresentationToWireframe()
        self.renderer.AddActor(actor)
        self.finalSocketHardActor = actor
        self.renderWindow.Render()

        self.genConnectivePortion(carpals, fingerMeshes, fingerInfo)

    def genConnectivePortion(self, carpals, fingerMeshes, fingerInfo):
        handMesh = (
            pv.wrap(self.actor.GetMapper().GetInput())
            .triangulate()
            .subdivide(2, "butterfly")
        )
        jointRegionalPointIDXs = []

        socket = pv.wrap(self.finalSocketHard).triangulate()
        socket = self.removeNonManifoldPartsOfMesh(socket)

        newProportionalPositions = {}

        initRadius = fingerInfo["connectorWidth"] / 2 + 1
        layers = 11

        rayLength = max(
            [handMesh.bounds[2 * k + 1] - handMesh.bounds[2 * k] for k in range(3)]
        )
        overallTime = time.time()
        for i, carpal in enumerate(carpals):
            start = time.time()
            tempIDList = []
            direction = carpal["center"] - carpal["normal"]
            direction /= np.linalg.norm(direction)
            point, _ = socket.ray_trace(
                carpal["center"],
                -1 * rayLength * direction,
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

        jointRegionalPointIDXs = list(
            set([x for xs in jointRegionalPointIDXs for x in xs])
        )

        def selectPoints(
            carpal,
            radius: int,
            color=(0.3, 0.2, 1),
            size=5,
        ):
            threshold = 5e-1
            start = time.time()
            direction = carpal["center"] - carpal["normal"]
            direction /= np.linalg.norm(direction)
            point, _ = socket.ray_trace(
                carpal["center"],
                -1 * rayLength * direction,
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
                    v = carpal["normal"]
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
                if idx in jointRegionalPointIDXs:
                    jointRegionalPointIDXs.remove(idx)
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

        def findDistanceP(p, fingerVector, jointOrigin):
            u = p - np.array(jointOrigin)
            n = np.array(fingerVector) / np.linalg.norm(np.array(fingerVector))
            projectedPos = p - n * np.dot(u, n)
            distance = np.linalg.norm(projectedPos - np.array(jointOrigin))
            return distance

        def proportionallyMovePoints(carpals, jointIdx, circumferenceOfPoints):
            tree = cKDTree(circumferenceOfPoints[jointIdx][0])
            n1 = (jointIdx - 1) % len(carpals)
            n2 = (jointIdx + 1) % len(carpals)
            neighborTree1 = cKDTree(circumferenceOfPoints[n1][0])
            neighborTree2 = cKDTree(circumferenceOfPoints[n2][0])

            for idx in jointRegionalPointIDXs:
                p = np.array(socket.points[idx])
                n = np.array(carpals[jointIdx]["normal"]) / np.linalg.norm(
                    np.array(carpals[jointIdx]["normal"])
                )
                distanceP = findDistanceP(
                    p, carpals[jointIdx]["normal"], carpals[jointIdx]["center"]
                )
                distanceP1 = findDistanceP(
                    p, carpals[n1]["normal"], carpals[n1]["center"]
                )
                distanceP2 = findDistanceP(
                    p, carpals[n2]["normal"], carpals[n2]["center"]
                )

                _, closestPointIdx = tree.query(p)
                intensity = circumferenceOfPoints[jointIdx][1][closestPointIdx]
                distance = circumferenceOfPoints[jointIdx][2][closestPointIdx]

                _, closestPointIdx = neighborTree1.query(p)
                intensity1 = circumferenceOfPoints[n1][1][closestPointIdx]
                distance11 = circumferenceOfPoints[n1][2][closestPointIdx]

                _, closestPointIdx = neighborTree2.query(p)
                intensity2 = circumferenceOfPoints[n2][1][closestPointIdx]
                distance21 = circumferenceOfPoints[n2][2][closestPointIdx]

                factor = 1
                if distanceP1 > distanceP and distanceP2 > distanceP:
                    if distanceP1 > distanceP2:
                        secondUsedIntensity = intensity2
                        secondUsedDistance = distance21
                        secondUsedPDistance = distanceP2
                    else:
                        secondUsedIntensity = intensity1
                        secondUsedDistance = distance11
                        secondUsedPDistance = distanceP1

                    ratio = 0.5 * distanceP / secondUsedPDistance
                    factor = (1 - ratio) * intensity * (distance**2) / (
                        distanceP**2
                    ) + ratio * secondUsedIntensity * (secondUsedDistance**2) / (
                        secondUsedPDistance**2
                    )
                    newPos = p + n * factor

                    # if idx in newProportionalPositions:
                    #     newProportionalPositions[idx] = (
                    #         0.5 * newProportionalPositions[idx] + 0.5 * newPos
                    #     )
                    # else:
                    newProportionalPositions[idx] = newPos

        conjoiningMeshes = []

        circumferences = []
        for k, carpal in enumerate(carpals):
            listOfIds = selectPoints(
                carpal=carpal,
                radius=initRadius - 0.4,
                color=(0.5, 1, 1),
                size=initRadius,
            )
            circumferenceOfPoints = moveSelectedPointsToJoints(
                pointIDs=listOfIds,
                jointIdx=k,
                totalJointIds=len(carpals),
                fingerVector=carpal["normal"],
                jointOrigin=carpal["center"],
            )
            circumferences.append(circumferenceOfPoints)

        for k, carpal in enumerate(carpals):
            proportionallyMovePoints(
                carpals=carpals,
                circumferenceOfPoints=circumferences,
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
            cylinder = pv.Cylinder(
                center=connectorPieceLocation
                - (fingerInfo["connectorLength"] + 1)
                * carpal["normal"]
                / np.linalg.norm(carpal["normal"]),
                direction=crossedBiNormal,
                radius=1,
                height=initRadius * 3,
            ).triangulate()
            conjoiningMeshes.append(cylinder)

            if k == len(carpals) - 1:
                crossedBiNormal *= -1

            angledVector = carpal["normal"] / np.linalg.norm(
                carpal["normal"]
            ) - crossedBiNormal / np.linalg.norm(crossedBiNormal)
            angledVector = angledVector / np.linalg.norm(angledVector)

            wedge = pv.Cylinder(
                center=connectorPieceLocation
                - (fingerInfo["connectorLength"] + 1)
                * crossedBiNormal
                / np.linalg.norm(crossedBiNormal),
                direction=angledVector,
                radius=fingerInfo["connectorWidth"] / 2 + 1,
                height=6,
            ).triangulate()

            conjoiningMeshes.append(wedge)

        for pointToMove, newPointPosition in newProportionalPositions.items():
            socket.points[pointToMove] = newPointPosition

        socket = socket.compute_normals(
            consistent_normals=True,
            auto_orient_normals=True,
        ).smooth_taubin()
        for mesh in conjoiningMeshes:
            boolean = vtkPolyDataBooleanFilter()
            boolean.SetInputData(0, socket)
            boolean.SetInputData(1, mesh)
            boolean.SetOperModeToDifference()
            boolean.Update()
            socket = pv.wrap(boolean.GetOutput())
        cleanedSocket = socket.clean()

        file1 = open("strengthAnalysis/connectiveGenerationTime2.txt", "a")
        file1.write(f"Time: {time.time()-overallTime} Num Fingers: {len(carpals)}\n")
        file1.close()

        self.finalSocketHard.DeepCopy(cleanedSocket)
        self.finalSocketHard.Modified()
        self.finalSocketHardActor.GetProperty().SetOpacity(1)
        self.finalSocketHardActor.GetProperty().SetColor(self.colors.GetColor3d("Blue"))
        self.renderWindow.Render()

        appendFilter = vtkAppendPolyData()
        appendFilter.AddInputData(self.finalSocket)
        for i, finger in enumerate(fingerMeshes):
            appendFilter.AddInputData(finger)
        appendFilter.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName("imageAnalysisGeneration/hardSocket.stl")
        writer.SetInputData(self.finalSocketHard)
        writer.Write()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName("imageAnalysisGeneration/softSocket.stl")
        writer.SetInputData(self.finalSocket)
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
