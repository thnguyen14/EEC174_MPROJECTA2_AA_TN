import numpy as np

MAX_DISTANCE = 50   

class Tracker:
    def __init__(self, maxLost=30):
        """
        Initialize the tracker.

        Parameters:
        maxLost (int): Maximum number of frames an object can be lost before deregistration.
        """
        self.maxLost = maxLost
        #stores how many frames were lost per object id
        self.lost_frames = {}

        #stores the path of the objects
        #also keeps the first centroid
        self.paths = {}

        self.next_object_id = 0

    def register(self, centroid):
        """
        Register a new object with the next available ID.

        Parameters:
        centroid: The centroid of the new object to register (x, y).

        Returns:
        None
        """
        object_id = self.next_object_id
        self.lost_frames[object_id] = 0
        self.paths[object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, objectID):
        """
        Deregister an object, removing it from tracking.

        Parameters:
        objectID: The ID of the object to deregister.

        Returns:
        None
        """
        del self.lost_frames[objectID]
        del self.paths[objectID]
        

    def update(self, inputCentroids):
        """
        Update the tracked objects with new centroid information from the current frame.

        Parameters:
        inputCentroids: list of centroids detected in the current frame.

        Returns:
        Updated objects with their current centroid positions.
        """

        #The case that no centroids are detected in the frame at all
        if len(inputCentroids) == 0:
            #itterates through each id at a time
            for id in list(self.paths.keys()):
                self.lost_frames[id] += 1
                if self.lost_frames[id] > self.maxLost:
                    self.deregister(id)
            updated_objects = {}
            for id,path in self.paths.items() :
                #-1 returns the last item, in this case, latest
                updated_objects[id] = path[-1]
            return updated_objects

        #All new centroids are detected in the frame
        if len(self.paths) == 0:
            for c in inputCentroids:
                self.register(c)
            updated_objects = {}
            for id,path in self.paths.items() :
                #-1 returns the last item, in this case, latest
                updated_objects[id] = path[-1]
            return updated_objects
        
        #Matching detections 
        obj_ids = list(self.paths.keys())
        last_positions = []
        for id in list(obj_ids):
            #Saves the last known centroid location
            last_positions.append(self.paths[id][-1])
        distances = self._distance(last_positions,inputCentroids)

        #This code stores the distance calculations between each of the
        #currently recorded objects and the new input centroids
        #Row and column will be used to identify the object_id and specific input centroid respectively
        paired_distance_calculations = []
        for row in range(len(obj_ids)):
            for col in range(len(inputCentroids)):
                d = distances[row][col]
                paired_distance_calculations.append((d,row,col))

        #Sorts from smallest item first
        #in this case, since we're using tuples, distance is the sorted item
        paired_distance_calculations.sort()

        matched_rows = set()
        matched_cols = set()

        for distance,row,col in paired_distance_calculations :
            #too far for it to be same person
            if distance > MAX_DISTANCE:
                break

            #row or col has already been matched to another object
            if row in matched_rows or col in matched_cols:
                continue

            #New centroid matches an objects old centroid
            object_id = obj_ids[row]
            next_point = inputCentroids[col]

            self.paths[object_id].append(next_point)
            self.lost_frames[object_id] = 0

            matched_rows.add(row)
            matched_cols.add(col)

        #Objects that were not detected in this frame must have their lost_count incremented
        for row in range(len(obj_ids)):
            if row not in matched_rows:
                object_id = obj_ids[row]
                self.lost_frames[object_id] += 1
                if self.lost_frames[object_id] > self.maxLost:
                    self.deregister(object_id)

        #New Centroid matched no old objects, so it becomes a new object
        for col in range(len(inputCentroids)):
            if col not in matched_cols:
                self.register(inputCentroids[col])

        updated_objects = {}
        for id,path in self.paths.items():
            updated_objects[id] = path[-1]
        return updated_objects

    def get_paths(self):
        """
        Retrieve the paths (list of centroids) of all tracked objects.

        Returns:
        Paths of tracked objects.
        """
        return self.paths

    def _distance(self, a, b):
        """
        Calculate the Euclidean distances between two sets of centroids.

        Parameters:
        a: First set of centroids.
        b: Second set of centroids.

        Returns:
        Matrix of distances between each pair of centroids from set 'a' and 'b'.
        """

        #a and b reshaped into numpy arrays
        #-1 means use as many rows needed, 2 fixed number of columns

        a = np.asarray(a, dtype=float).reshape(-1,2)
        b = np.asarray(b, dtype=float).reshape(-1,2)

        #subtracts all parts of b from all parts of a
        #norm finds distance/pythagorean hypotnuse from a and b
        #axis = 2 leaves a table of distances
        return np.linalg.norm(a[:,None,:] -b[None,:,:], axis = 2)

        
