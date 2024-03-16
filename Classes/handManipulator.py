from lib import vtk, pd, np, time, pv


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
