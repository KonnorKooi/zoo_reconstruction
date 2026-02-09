"""
Mesh loading and ray-mesh intersection utilities.
"""

import numpy as np
import trimesh
from typing import Optional, Tuple, List
from pathlib import Path

from .data_structures import Ray3D


class MeshIntersector:
    """Handles mesh loading and ray-mesh intersection operations.

    This class encapsulates:
    - Loading and validating PLY meshes
    - Setting up ray intersection acceleration structures
    - Computing ray-mesh intersections
    - Handling edge cases (misses, multiple hits, etc.)

    Attributes:
        mesh: Trimesh object
        ray_intersector: RayMeshIntersector for fast intersection queries
    """

    def __init__(self, mesh_path: str):
        """Load mesh and prepare for ray intersection.

        Args:
            mesh_path: Path to PLY mesh file

        Raises:
            FileNotFoundError: If mesh file doesn't exist
            ValueError: If mesh is invalid or degenerate
        """
        mesh_path = Path(mesh_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        # Load mesh
        print(f"Loading mesh from {mesh_path.name}...")
        self.mesh = trimesh.load(str(mesh_path))

        # Validate mesh
        self._validate_mesh()

        # Create ray intersector (builds acceleration structure)
        print("Building ray intersection acceleration structure...")
        self.ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

        print(f"✓ Mesh loaded: {len(self.mesh.vertices):,} vertices, "
              f"{len(self.mesh.faces):,} faces")
        print(f"  Bounds: X=[{self.mesh.bounds[0,0]:.2f}, {self.mesh.bounds[1,0]:.2f}], "
              f"Y=[{self.mesh.bounds[0,1]:.2f}, {self.mesh.bounds[1,1]:.2f}], "
              f"Z=[{self.mesh.bounds[0,2]:.2f}, {self.mesh.bounds[1,2]:.2f}]")
        print(f"  Watertight: {self.mesh.is_watertight}")

    def _validate_mesh(self):
        """Validate mesh properties."""
        # Check for NaN vertices
        if np.any(np.isnan(self.mesh.vertices)):
            raise ValueError("Mesh contains NaN vertices")

        # Check for degenerate faces
        if hasattr(self.mesh, 'remove_degenerate_faces'):
            original_faces = len(self.mesh.faces)
            self.mesh.remove_degenerate_faces()
            if len(self.mesh.faces) < original_faces:
                print(f"  Warning: Removed {original_faces - len(self.mesh.faces)} "
                      "degenerate faces")

        # Basic validity check (if available)
        if hasattr(self.mesh, 'is_valid') and not self.mesh.is_valid:
            print("  Warning: Mesh may have validity issues (continuing anyway)")

    def intersect_ray(
        self,
        ray: Ray3D,
        return_all_hits: bool = False
    ) -> Optional[np.ndarray]:
        """Intersect a single ray with the mesh.

        Args:
            ray: Ray3D object to intersect
            return_all_hits: If True, return all intersection points.
                           If False, return only the closest intersection.

        Returns:
            - If hit: 3D intersection point (closest if multiple hits)
            - If miss: None

        Note:
            The ray direction MUST be normalized (handled by Ray3D constructor).
        """
        # Prepare ray for trimesh (needs shape (1, 3))
        ray_origins = ray.origin.reshape(1, 3)
        ray_directions = ray.direction.reshape(1, 3)

        # Compute intersections
        locations, index_ray, index_tri = self.ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=return_all_hits
        )

        # Handle no intersection
        if len(locations) == 0:
            return None

        # If only one hit or want closest, return first (closest)
        if not return_all_hits or len(locations) == 1:
            return locations[0]

        # Return all hits sorted by distance
        distances = np.linalg.norm(locations - ray.origin, axis=1)
        sorted_indices = np.argsort(distances)
        return locations[sorted_indices]

    def intersect_rays_batch(
        self,
        rays: List[Ray3D]
    ) -> List[Optional[np.ndarray]]:
        """Intersect multiple rays with mesh (batched for performance).

        Args:
            rays: List of Ray3D objects

        Returns:
            List of intersection points (None for misses)

        Note:
            This is much faster than calling intersect_ray in a loop.
        """
        if not rays:
            return []

        # Prepare batch arrays
        ray_origins = np.array([ray.origin for ray in rays])
        ray_directions = np.array([ray.direction for ray in rays])

        # Compute intersections (first hit only)
        locations, index_ray, index_tri = self.ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        # Build result list (None for rays that missed)
        results = [None] * len(rays)
        for i, ray_idx in enumerate(index_ray):
            results[ray_idx] = locations[i]

        return results

    def get_intersection_info(
        self,
        ray: Ray3D
    ) -> Optional[dict]:
        """Get detailed intersection information.

        Args:
            ray: Ray3D object

        Returns:
            Dictionary with intersection details:
                - point: 3D intersection point
                - distance: Distance from ray origin to intersection
                - triangle_index: Index of intersected triangle
                - normal: Surface normal at intersection
            Returns None if ray misses mesh.
        """
        # Prepare ray
        ray_origins = ray.origin.reshape(1, 3)
        ray_directions = ray.direction.reshape(1, 3)

        # Get intersection with triangle indices
        locations, index_ray, index_tri = self.ray_intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            multiple_hits=False
        )

        if len(locations) == 0:
            return None

        # Get intersection point and triangle
        intersection_point = locations[0]
        triangle_idx = index_tri[0]

        # Compute distance
        distance = np.linalg.norm(intersection_point - ray.origin)

        # Get triangle normal
        normal = self.mesh.face_normals[triangle_idx]

        return {
            'point': intersection_point,
            'distance': distance,
            'triangle_index': triangle_idx,
            'normal': normal,
        }

    def check_ray_mesh_geometry(
        self,
        ray: Ray3D,
        max_distance: float = 1000.0
    ) -> dict:
        """Check geometric relationship between ray and mesh.

        Useful for debugging why a ray might be missing the mesh.

        Args:
            ray: Ray3D object
            max_distance: Maximum distance to check along ray

        Returns:
            Dictionary with geometric information:
                - intersects: Whether ray hits mesh
                - ray_origin_in_bounds: Whether ray starts inside mesh bounds
                - min_distance_to_bounds: Minimum distance from ray to mesh bounds
                - closest_point_on_mesh: Closest mesh vertex to ray
        """
        intersection = self.intersect_ray(ray)

        # Check if ray origin is within mesh bounding box
        bounds = self.mesh.bounds
        origin_in_bounds = np.all(
            (ray.origin >= bounds[0]) & (ray.origin <= bounds[1])
        )

        # Compute minimum distance from ray origin to mesh bounds
        # (Distance to closest corner of bounding box)
        corners = []
        for x in bounds[:, 0]:
            for y in bounds[:, 1]:
                for z in bounds[:, 2]:
                    corners.append([x, y, z])
        corners = np.array(corners)
        distances_to_corners = np.linalg.norm(corners - ray.origin, axis=1)
        min_dist_to_bounds = np.min(distances_to_corners)

        # Find closest mesh vertex to ray origin
        vertex_distances = np.linalg.norm(self.mesh.vertices - ray.origin, axis=1)
        closest_vertex_idx = np.argmin(vertex_distances)
        closest_vertex = self.mesh.vertices[closest_vertex_idx]
        closest_vertex_dist = vertex_distances[closest_vertex_idx]

        return {
            'intersects': intersection is not None,
            'intersection_point': intersection,
            'ray_origin_in_bounds': origin_in_bounds,
            'min_distance_to_bounds': min_dist_to_bounds,
            'closest_mesh_vertex': closest_vertex,
            'closest_vertex_distance': closest_vertex_dist,
        }

    def get_mesh_statistics(self) -> dict:
        """Get mesh statistics for debugging.

        Returns:
            Dictionary with mesh properties
        """
        return {
            'num_vertices': len(self.mesh.vertices),
            'num_faces': len(self.mesh.faces),
            'num_edges': len(self.mesh.edges),
            'is_watertight': self.mesh.is_watertight,
            'is_valid': self.mesh.is_valid if hasattr(self.mesh, 'is_valid') else None,
            'bounds_min': self.mesh.bounds[0],
            'bounds_max': self.mesh.bounds[1],
            'extents': self.mesh.extents,
            'center': self.mesh.centroid,
            'volume': self.mesh.volume if self.mesh.is_watertight else None,
            'area': self.mesh.area if hasattr(self.mesh, 'area') else None,
        }


def create_sphere_at_point(
    point: np.ndarray,
    radius: float = 0.05,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> trimesh.Trimesh:
    """Create a sphere mesh at a given 3D point.

    Useful for visualizing intersection points in 3D.

    Args:
        point: 3D coordinates (x, y, z)
        radius: Sphere radius in world units
        color: RGB color (0-255)

    Returns:
        Trimesh sphere object
    """
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    sphere.apply_translation(point)
    sphere.visual.vertex_colors = np.array([color[0], color[1], color[2], 255])
    return sphere


def visualize_ray_mesh_intersection(
    mesh: trimesh.Trimesh,
    ray: Ray3D,
    intersection_point: Optional[np.ndarray] = None,
    ray_length: float = 10.0
) -> trimesh.Scene:
    """Create a 3D scene showing ray and mesh intersection.

    Args:
        mesh: Trimesh object
        ray: Ray3D object
        intersection_point: Optional intersection point to highlight
        ray_length: Length of ray to visualize

    Returns:
        Trimesh scene ready for visualization
    """
    scene = trimesh.Scene()

    # Add mesh
    scene.add_geometry(mesh)

    # Create ray as a line
    ray_end = ray.origin + ray.direction * ray_length
    ray_line = trimesh.load_path([ray.origin, ray_end])
    ray_line.colors = np.array([[0, 255, 0, 255]])  # Green
    scene.add_geometry(ray_line)

    # Add sphere at ray origin (camera center)
    origin_sphere = create_sphere_at_point(ray.origin, radius=0.02, color=(0, 0, 255))
    scene.add_geometry(origin_sphere)

    # Add sphere at intersection if provided
    if intersection_point is not None:
        intersection_sphere = create_sphere_at_point(
            intersection_point, radius=0.05, color=(255, 0, 0)
        )
        scene.add_geometry(intersection_sphere)

    return scene
