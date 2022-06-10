import numpy as np


class Point:
    def __init__(self, x, y) -> None:
        self.coordinates = np.array([x, y])

    def __sub__(self, other):
        """Calculate euclidean distance between two Point objects."""
        return np.linalg.norm(self.coordinates - other.coordinates)

    def __repr__(self) -> str:
        return f"Point(x={self.coordinates[0]}, y={self.coordinates[1]})"


if __name__ == "__main__":
    p1 = Point(23.2, 33.1)
    print(p1)
    p2 = Point(10, 10)
    print(p2)
    p3 = Point(10, 10)
    print(p1 - p2)
    print(p2 - p1)
    print(p2 - p3)
