import numpy as np
from typing import List
from src.point import Point
from src.face import Face
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Container:
    """
    Represents a container defined by a set of faces, which can be used to define a multiply connected domain.
    Each face is a Face object with vertices and a description.
    """
    def __init__(self, faces, description: str = ""):
        self.faces = faces
        self.vertices = self.get_vertices()
        self.edges = self.get_edges()
        self.description = description

    def get_vertices(self):
        verts = set()
        for face in self.faces:
            for v in face.vertices:
                verts.add(v)
        return list(verts)

    def get_edges(self):
        edges = set()
        for face in self.faces:
            n = len(face.vertices)
            for i in range(n):
                a = face.vertices[i]
                b = face.vertices[(i+1)%n]
                edge = (a,b) if a < b else (b,a)
                edges.add(edge)
        return list(edges)

    def contains(self, point: Point, num_rays: int = 10, ray_dir: np.ndarray = None,
                 eps: float = 1e-8, show_rays: bool = False) -> bool:
        """
        Ray-casting test with multiple rays. If show_rays, plots each ray in a unique color
        and prints descriptions of faces each ray intersects.
        Returns True if majority of rays report inside.
        """
        p = point.as_array()
        # Boundary and edge checks
        for face in self.faces:
            pts = [v.as_array() for v in face.vertices]
            normal = face.normal
            if abs(np.dot(p-pts[0], normal))<eps and Container._point_in_polygon(p, pts, normal, eps):
                return True
        for v1, v2 in self.edges:
            a, b = v1.as_array(), v2.as_array()
            if np.linalg.norm(np.cross(b-a, p-a))<eps and \
               np.dot(b-a, p-a)>=-eps and np.dot(a-b, p-b)>=-eps:
                return True

        total = 1 if ray_dir is not None else num_rays
        inside = 0
        ray_data = []  # (ray, t, faces_hit)

        # bounding for ray length
        pts_all = np.array([v.as_array() for v in self.vertices])
        bb = np.linalg.norm(pts_all.max(0) - pts_all.min(0))

        for i in range(total):
            if ray_dir is None:
                r = np.random.normal(size=3)
                ray = r / np.linalg.norm(r)
            else:
                ray = ray_dir / np.linalg.norm(ray_dir)
            t_min = np.inf
            faces_hit = []
            for face in self.faces:
                normal = face.normal
                pts = [v.as_array() for v in face.vertices]
                p0 = pts[0]
                denom = np.dot(ray, normal)
                if abs(denom) < eps: continue
                t = np.dot(p0 - p, normal) / denom
                if t <= eps: continue
                ip = p + t * ray
                if Container._point_in_polygon(ip, pts, normal, eps):
                    desc = face.description
                    faces_hit.append(desc)
                    if t < t_min:
                        t_min = t
            count = len(faces_hit)
            if count % 2 == 1:
                inside += 1
            ray_data.append((ray, t_min if t_min < np.inf else bb * 1.1, faces_hit))

        result = inside > total / 2

        if show_rays:
            ax = self.render(show=False)
            cmap = plt.cm.get_cmap('hsv', total)
            for idx, (ray, t, faces_hit) in enumerate(ray_data):
                end = p + ray * t
                color = cmap(idx)
                ax.plot([p[0], end[0]], [p[1], end[1]], [p[2], end[2]], color=color)
                # print(f"Ray {idx} hits faces: {faces_hit}")
            plt.show()
        # print(f"Inside: {inside}, Total: {total}")
        return result

    @staticmethod
    def _point_in_polygon(pt: np.ndarray, verts: list, normal: np.ndarray, eps: float) -> bool:
        axis = int(np.argmax(np.abs(normal)))
        proj = [np.delete(v, axis) for v in verts]
        p2 = np.delete(pt, axis)
        inside=False
        n=len(proj)
        for i in range(n):
            v1, v2 = proj[i], proj[(i+1)%n]
            if ((v1[1] > p2[1]) != (v2[1] > p2[1])):
                xi = v1[0] + (p2[1] - v1[1]) * (v2[0] - v1[0]) / (v2[1] - v1[1])
                if xi > p2[0] + eps:
                    inside = not inside
        return inside

    def render(self, ax=None, show=True, face_color='cyan', edge_color='black', alpha=0.5):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        poly3d=[[v.as_array() for v in face.vertices] for face in self.faces]
        coll=Poly3DCollection(poly3d,facecolors=face_color,edgecolors=edge_color,alpha=alpha)
        ax.add_collection3d(coll)
        pts_all = np.array([v.as_array() for v in self.vertices])
        mn, mx = pts_all.min(0), pts_all.max(0)
        cen=(mn+mx)/2; rad=(mx-mn).max()/2
        ax.set_xlim(cen[0]-rad,cen[0]+rad)
        ax.set_ylim(cen[1]-rad,cen[1]+rad)
        ax.set_zlim(cen[2]-rad,cen[2]+rad)
        if show: plt.show()
        return ax
    
    def is_active_plane(self, face: Face, point: Point, eps: float = 1e-8) -> bool:
        """
        Check if a face is an active plane for a given point.
        An active plane is one where the point is on the same side as the face normal.
        """
        if face is None or face not in self.faces:
            return False
        
        if self.contains(point) and face.half_space(point) < 0:
            return True
        elif not self.contains(point) and face.half_space(point) > 0:
            return True
        return False
    
    def has_active_footpoint(self, point: Point, face: Face, eps: float = 1e-8) -> bool:
        """
        Check if a point has an active footpoint for a given face.
        A footpoint is active if the point is on the same side as the face normal.
        """
        if face is None or face not in self.faces:
            return False
        
        if not self.is_active_plane(face, point, eps):
            return False
        
        # Draw a normal ray from the point to the face
        normal = -face.normal
        
        foot = point.as_array() + normal
        
        # if the foot is a point on the face, then it is an active footpoint
        if face.half_space(Point(*foot)) < eps:
            return True
        
        return False
    
    def get_active_footpoints(self, point: Point, eps: float = 1e-8) -> list:
        """
        Get all active footpoints for a given point.
        An active footpoint is one where the point is on the same side as the face normal.
        """
        active_faces = [face for face in self.faces if self.is_active_plane(face, point, eps)]
        return [face.project_point(point) for face in active_faces if self.has_active_footpoint(point, face, eps)]
    
    def volume(self):
        volume_summation = 0.0
        for face in self.faces:
            volume_summation += face.area() * np.dot(face.normal, face.vertices[0].as_array())
        return abs(volume_summation) / 3.0
    
    def bounding_box(self):
        pts_all = np.array([v.as_array() for v in self.vertices])
        x_min, x_max = pts_all[:, 0].min(), pts_all[:, 0].max()
        y_min, y_max = pts_all[:, 1].min(), pts_all[:, 1].max()
        z_min, z_max = pts_all[:, 2].min(), pts_all[:, 2].max()
        
        return {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        }
        
    def generate_initial_solution(self, p: int) -> np.ndarray:
        """
        Generate an initial random solution of points within the bounding box of the container.
        :param p: Number of points to generate.
        :return: Array of shape (p, 3) representing the initial solution points.
        """
        bounding_box = self.bounding_box()
        x_min, x_max = bounding_box['x_min'], bounding_box['x_max']
        y_min, y_max = bounding_box['y_min'], bounding_box['y_max']
        z_min, z_max = bounding_box['z_min'], bounding_box['z_max']
        
        points = []
        for _ in range(p):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            z = np.random.uniform(z_min, z_max)
            points.append([x, y, z])
        
        return np.array(points).reshape(-1, 3)
    
    def random_point(self, as_array = False) -> Point:
        """
        Generate a random point within the bounding box of the container.
        :return: A Point object representing the random point.
        """
        bounding_box = self.bounding_box()
        while True:
            x = np.random.uniform(bounding_box['x_min'], bounding_box['x_max'])
            y = np.random.uniform(bounding_box['y_min'], bounding_box['y_max'])
            z = np.random.uniform(bounding_box['z_min'], bounding_box['z_max'])
            point = Point(x, y, z)
            if self.contains(point):
                return point if not as_array else point.as_array()

    def __repr__(self):
        desc = f" {self.description}" if self.description else ""
        return f"Container with {len(self.vertices)} vertices and {len(self.faces)} faces."+desc