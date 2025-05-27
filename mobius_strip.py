import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        """
        Initialize a Mobius strip model.

        Parameters:
        - R: Radius (distance from center to strip centerline)
        - w: Width of the strip
        - n: Resolution (number of points in mesh)
        """
        self.R = R
        self.w = w
        self.n = n

        # Create the u and v parameters
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)

        # Calculate the 3D coordinates
        self.X, self.Y, self.Z = self._compute_points()

    def _compute_points(self):
        """
        Calculate the (x, y, z) points of the Mobius strip surface
        using the provided parametric equations.
        """
        U, V = self.U, self.V
        X = (self.R + V * np.cos(U / 2)) * np.cos(U)
        Y = (self.R + V * np.cos(U / 2)) * np.sin(U)
        Z = V * np.sin(U / 2)
        return X, Y, Z

    def plot(self):
        """
        Plot the Mobius strip using matplotlib's 3D plotting.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(self.X, self.Y, self.Z,
                        cmap='viridis',
                        edgecolor='none',
                        alpha=0.85)

        ax.set_title('Mobius Strip')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def surface_area(self):
        """
        Approximate the surface area using numerical integration.
        This uses the magnitude of the cross product of the tangent vectors.
        """
        # Calculate tangent vectors along u and v
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]

        Xu = np.gradient(self.X, axis=1) / du
        Yu = np.gradient(self.Y, axis=1) / du
        Zu = np.gradient(self.Z, axis=1) / du

        Xv = np.gradient(self.X, axis=0) / dv
        Yv = np.gradient(self.Y, axis=0) / dv
        Zv = np.gradient(self.Z, axis=0) / dv

        # Cross product of tangent vectors gives normal vector
        Nx = Yu * Zv - Zu * Yv
        Ny = Zu * Xv - Xu * Zv
        Nz = Xu * Yv - Yu * Xv

        # Area element (dA) is the magnitude of the normal vector
        dA = np.sqrt(Nx**2 + Ny**2 + Nz**2)

        # Integrate over v then u
        area_v = simps(dA, self.v, axis=0)
        total_area = simps(area_v, self.u)
        return total_area

    def edge_length(self):
        """
        Approximate the total edge length of the strip
        by summing distances along the boundaries (v = ±w/2).
        """
        total_length = 0

        # Check the edges at v = -w/2 and v = w/2
        for v_idx in [0, -1]:
            x_edge = self.X[v_idx, :]
            y_edge = self.Y[v_idx, :]
            z_edge = self.Z[v_idx, :]

            # Calculate distance between consecutive edge points
            dx = np.diff(x_edge)
            dy = np.diff(y_edge)
            dz = np.diff(z_edge)
            ds = np.sqrt(dx**2 + dy**2 + dz**2)

            total_length += np.sum(ds)

        return total_length

if __name__ == "__main__":
    # Create an instance of the Mobius strip
    strip = MobiusStrip(R=1.0, w=0.3, n=200)

    # Compute and print the geometric properties
    area = strip.surface_area()
    length = strip.edge_length()
    print(f"Surface Area: {area:.4f}")
    print(f"Edge Length: {length:.4f}")

    # Visualize the Mobius strip
    strip.plot()
