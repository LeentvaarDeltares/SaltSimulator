import numpy as np
import pandas as pd
from pygimli.meshtools import readGmsh
import pygimli as pg
import glob
import os
import matplotlib.gridspec as gridspec

from scipy.spatial import distance_matrix
from scipy.interpolate import griddata
from itertools import combinations
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, cKDTree
from pygimli.physics import ert
from ertoolbox import inversion
from ertoolbox import ert_postprocessing

from pygimli.physics import ert


def inversieintern(data, mesh, cmin, cmax):
    mgr = ert.ERTManager(data, verbose=True)
    inversie = mgr.invert(mesh=mesh)
    mgr.showResult(cMin=cmin, cMax=cmax)
    mesh = mgr.paraDomain
    return mgr, inversie


def calculate_distances(points):
    """
    Calculate distances between consecutive points.

    Parameters:
    points (array-like): A 2D array or list of points where each point is represented as [x, y].

    Returns:
    np.ndarray: An array of distances between consecutive points.
    """
    points = np.array(points)  # Ensure the input is a NumPy array
    # Calculate the distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    return distances


folder = "p:/11209233-mad09a2023ijsselmeer3d/C_Work/02_FM/03_postprocessing/09_input_tomografische_metingen/voorhaven_kwz/sal_tem"
counter = 0
for csv in glob.glob(f"{folder}/*.csv"):
    counter += 1
    print(os.path.basename(csv)[0:-4])
    data = pd.read_csv(csv)

    data.columns = ["x", "z", "salinity", "temperature"]
    pressure = 1025 * 9.81 * -data["z"] / 10000

    data["resistivity"] = 1 / ert_postprocessing.salinity_to_conductivity(
        data["salinity"], data["temperature"], pressure
    )

    # Define grid parameters
    x_min, x_max = data["x"].min(), data["x"].max()
    z_min, z_max = data["z"].min(), data["z"].max()
    grid_size = 50  # Number of grid points along each axis

    # Create a regular grid
    x_grid = np.linspace(x_min, x_max, grid_size)
    z_grid = np.linspace(z_min, z_max, grid_size)
    X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
    Y_grid = np.zeros(np.shape(X_grid))

    # Stack grid data
    xyz = np.column_stack([X_grid.flatten(), Z_grid.flatten()])

    # Prepare data for interpolation
    points = data[["x", "z"]].values
    values = data["resistivity"].values

    # Interpolate
    grid_values = griddata(points, values, (X_grid, Z_grid), method="cubic")
    grid_values[grid_values < 0] = 0.001
    grid_values_og = grid_values.copy()

    # Interpolate onto the structured grid
    grid_z = griddata(points, values, (X_grid, Z_grid), method="cubic")

    ##---
    padding_points = 50  # Number of points for padding
    new_grid_size = grid_size + padding_points

    # Extend x and z grid with 50 points padding on both sides (left, right, below)
    x_grid_padded = np.linspace(
        x_min - (x_grid[1] - x_grid[0]) * padding_points,
        x_max + (x_grid[1] - x_grid[0]) * padding_points,
        new_grid_size,
    )
    z_grid_padded = np.linspace(
        z_min - (z_grid[1] - z_grid[0]) * padding_points, z_max, new_grid_size
    )
    X_grid_padded, Z_grid_padded = np.meshgrid(x_grid_padded, z_grid_padded)
    Y_grid_padded = np.zeros(np.shape(X_grid_padded))
    grid_values_padded = griddata(
        points, values, (X_grid_padded, Z_grid_padded), method="cubic"
    )
    # Interpolate onto the extended grid
    grid_values_padded = griddata(
        points, values, (X_grid_padded, Z_grid_padded), method="cubic"
    )
    grid_values_padded[grid_values_padded < 0] = 0.00001

    grid_values_padded = np.nan_to_num(grid_values_padded, nan=1000)

    xyz_padded = np.column_stack([X_grid_padded.flatten(), Z_grid_padded.flatten()])

    # Create a DataFrame
    df = pd.DataFrame(
        {"x": X_grid.flatten(), "z": Z_grid.flatten(), "value": grid_values.flatten()}
    )
    from scipy import interpolate

    # Only use valid data points
    df = df.dropna()
    # #take 1 node above the bottom
    min_z_indices = df.groupby("x")["z"].idxmin()
    df = df.drop(min_z_indices)
    df.reset_index(drop=True, inplace=True)

    min_z_indices = df.groupby("x")["z"].idxmin()
    min_z_df = df.loc[min_z_indices]
    min_z_df.reset_index(drop=True, inplace=True)

    x_points = min_z_df["x"]
    z_points = min_z_df["z"]
    # Interpolation
    distance = 1  # cm

    # Calculate the distances between the original points
    distances = np.sqrt(np.diff(x_points) ** 2 + np.diff(z_points) ** 2)

    # Calculate cumulative distances
    cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

    # New distance points from the start to the end
    new_distances = np.arange(0, cumulative_distances[-1], distance)

    # Interpolating
    new_x = np.interp(new_distances, cumulative_distances, x_points)
    new_z = np.interp(new_distances, cumulative_distances, z_points)

    new_x = np.append(new_x, new_x[-1])
    new_x = np.append(new_x[0], new_x)
    new_z = np.append(new_z, 0)
    new_z = np.append(0, new_z)

    # Resulting interpolated points
    interpolated_points = np.column_stack((new_x, new_z))
    interpolated_points_mesh = np.vstack(
        [interpolated_points[0], interpolated_points, interpolated_points[0]]
    )
    interpolated_points_mesh[0][1] = 0
    interpolated_points_mesh[-1][1] = 0

    distances = calculate_distances(interpolated_points)

    # Create 2D Pygimli  mesh
    mesh = pg.Mesh(2)

    cs = []
    cr = []

    # Make nodes for all datapoints
    for p in xyz_padded:
        c = mesh.createNode((p[0], p[1]))
        cs.append(c)

    # Triangulate points into mesh with triangles
    tri = Delaunay(xyz_padded)
    triangles = tri.simplices

    # Calculate centroids of the triangles
    centroids = np.array(
        [
            np.mean(xyz_padded[tri.simplices[i]], axis=0)
            for i in range(len(tri.simplices))
        ]
    )

    # Find closest grid point to the triangle
    tree = cKDTree(xyz_padded)
    distances, indices = tree.query(centroids)

    # Add value to the grid cell
    triangle_values = grid_values_padded.flatten()[indices]

    # Add triangles to mesh object
    tlist = []
    for t in range(0, len(triangles)):
        mesh.createTriangle(
            cs[triangles[t][0]], cs[triangles[t][1]], cs[triangles[t][2]], marker=t
        )  # marker is value
        tlist = np.append(tlist, t)

    mesh.createNeighborInfos()

    listc = list(range(1, len(triangle_values) + 1))
    plotdata = np.array([[a, b] for a, b in zip(listc, triangle_values)])
    filtered_data = plotdata[~np.isnan(plotdata[:, 1])]

    # Define electrode positions
    EZ = new_z
    EX = new_x

    start_index_EX = len(EX) // 2 - 32  # 32 elements before the middle
    end_index_EX = len(EX) // 2 + 32  # 32 elements after the middle
    middle_64_EX = EX[start_index_EX:end_index_EX]
    start_index_EZ = len(EZ) // 2 - 32  # 32 elements before the middle
    end_index_EZ = len(EZ) // 2 + 32  # 32 elements after the middle
    middle_64_EZ = EZ[start_index_EZ:end_index_EZ]
    electrodes = [[x, y] for x, y in zip(middle_64_EX, middle_64_EZ)]

    # Create measurement scheme
    # schemes = ['wa', 'wb', 'pp', 'pd', 'dd', 'slm', 'hw', 'gr']
    scheme_dd = ert.createData(elecs=electrodes, schemeName="wa")

    # Make simulation data
    simdata = ert.simulate(
        mesh=mesh,
        scheme=scheme_dd,
        res=triangle_values,
        noiseLevel=1,
        noiseAbs=1e-6,
        seed=1337,
    )

    simdata.remove(simdata["rhoa"] < 0)

    # Create 2D Pygimli  mesh ONLY INVERSION DOMAIN
    meshi = pg.Mesh(2)

    cs = []
    cr = []

    # Make nodes for all datapoints
    for p in xyz:
        c = meshi.createNode((p[0], p[1]))
        cs.append(c)

    # Triangulate points into mesh with triangles
    tri = Delaunay(xyz)
    triangles = tri.simplices

    # Calculate centroids of the triangles
    centroids = np.array(
        [np.mean(xyz[tri.simplices[i]], axis=0) for i in range(len(tri.simplices))]
    )

    # Find closest grid point to the triangle
    tree = cKDTree(xyz)
    distances, indices = tree.query(centroids)

    # # Add value to the grid cell

    triangle_values = grid_values.flatten()[indices]

    # Add triangles to mesh object
    tlist = []
    for t in range(0, len(triangles)):
        meshi.createTriangle(
            cs[triangles[t][0]], cs[triangles[t][1]], cs[triangles[t][2]], marker=t
        )  # marker is value
        tlist = np.append(tlist, t)

    # Add boundaries
    line = pg.meshtools.createPolygon(interpolated_points_mesh, marker=1)

    # meshi = pg.meshtools.mergeMeshes([meshi,line])
    meshi.createNeighborInfos()
    # mesh = pg.meshtools.appendTriangleBoundary(mesh)
    listc = list(range(1, len(triangle_values) + 1))
    plotdata = np.array([[a, b] for a, b in zip(listc, triangle_values)])

    maxclim = max(simdata["rhoa"]) * 1.1
    mrg, inversion_dd_1 = inversieintern(simdata, meshi, cmin=0, cmax=maxclim)

    # # mgr.showResultAndFit()
    # Extract coordinates (x, z) of cell centers and resistivity values
    electrodes = np.array(electrodes)
    mesh = mrg.paraDomain
    coverage_values = mrg.coverage()
    cell_centers = np.array([cell.center() for cell in mesh.cells()])
    x, z = cell_centers[:, 0], cell_centers[:, 1]

    # Define grid parameters
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    grid_size = 50  # Number of grid points along each axis
    x_grid = np.linspace(x_min, x_max, grid_size)
    z_grid = np.linspace(z_min, z_max, grid_size)
    X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
    Y_grid = np.zeros(np.shape(X_grid))
    xyz = np.column_stack([X_grid.flatten(), Z_grid.flatten()])
    points = cell_centers[:, 0:2]
    values = inversion_dd_1

    # # Interpolate
    maxclim = 1.2
    coverage_grid = griddata(
        points, coverage_values, (X_grid, Z_grid), method="nearest"
    )
    grid_values = griddata(points, values, (X_grid, Z_grid), method="cubic")
    grid_values[grid_values >= maxclim] = maxclim
    grid_values[grid_values < 0] = 0

    # PLOT

    # Create a figure
    fig = plt.figure(figsize=(10, 10))  # Adjust size as needed

    # Define grid layout
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])  # 3 rows, 2 columns

    # Add subplots
    ax1 = fig.add_subplot(gs[0, :])  # First row (spans both columns)
    ax2 = fig.add_subplot(gs[1, :])  # Second row (spans both columns)
    ax3 = fig.add_subplot(gs[2, 0])  # Third row, first column
    ax4 = fig.add_subplot(gs[2, 1])  # Third row, second column

    # Plot contour
    # plt.figure(figsize=(20, 5))
    ax1.fill_between(
        np.append(new_x, 100),
        np.append(new_z, 0),
        y2=min(new_z) - 1,
        color="white",
        zorder=3,
    )
    contour = ax1.contourf(
        X_grid, Z_grid, grid_values, cmap="Spectral", levels=50, vmin=0, vmax=maxclim
    )  # place levels for smoother colorbar
    ax1.plot(new_x, new_z, color="black", linewidth=3, zorder=4)
    ax1.scatter(electrodes[:, 0], electrodes[:, 1], color="red", zorder=5)
    fig.colorbar(contour, ax=ax1, label="Resistivity (Ohm·m)")

    # Plot original model
    # Define grid parameters
    x_min, x_max = data["x"].min(), data["x"].max()
    z_min, z_max = data["z"].min(), data["z"].max()
    grid_size = 50  # Number of grid points along each axis

    # Create a regular grid
    x_grid = np.linspace(x_min, x_max, grid_size)
    z_grid = np.linspace(z_min, z_max, grid_size)
    X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
    Y_grid = np.zeros(np.shape(X_grid))

    # Plot original points

    ogmodel = ax2.contourf(
        X_grid,
        Z_grid,
        grid_values_og,
        levels=100,
        cmap="Spectral",
        vmin=0,
        vmax=maxclim,
    )
    ax2.scatter(
        data["x"],
        data["z"],
        c=data["resistivity"],
        cmap="Spectral",
        edgecolor="k",
        vmin=0,
        vmax=maxclim,
    )
    ax2.set_title("Original Data")
    fig.colorbar(ogmodel, ax=ax2, label="Resistivity (Ohm·m)")

    # plot coverage
    # plt.figure(figsize=(20, 5))
    ax3.fill_between(
        np.append(new_x, 100),
        np.append(new_z, 0),
        y2=min(new_z) - 1,
        color="white",
        zorder=3,
    )
    coverage = ax3.contourf(
        X_grid, Z_grid, coverage_grid, cmap="binary", levels=50, vmin=0, vmax=maxclim
    )  # place levels for smoother colorbar
    ax3.plot(new_x, new_z, color="black", linewidth=3, zorder=4)
    ax3.scatter(electrodes[:, 0], electrodes[:, 1], color="red", zorder=5)
    fig.colorbar(coverage, ax=ax3)
    # plt.colorbar(coverage, label="Coverage")

    # Plot contour with coverage
    # plt.figure(figsize=(20, 5))
    coverage = ax4.contourf(
        X_grid,
        Z_grid,
        coverage_grid,
        levels=[min(coverage_values), 0.2],
        colors="white",
        zorder=3,
    )
    ax4.fill_between(
        np.append(new_x, 100),
        np.append(new_z, 0),
        y2=min(new_z) - 1,
        color="white",
        zorder=3,
    )
    contour2 = ax4.contourf(
        X_grid, Z_grid, grid_values, cmap="Spectral", levels=50, vmin=0, vmax=maxclim
    )  # place levels for smoother colorbar
    ax4.plot(new_x, new_z, color="black", linewidth=3, zorder=4)
    ax4.scatter(electrodes[:, 0], electrodes[:, 1], color="red", zorder=5)
    fig.colorbar(contour2, ax=ax4)

    x_min, x_max = ax1.get_xlim()
    y_min, y_max = ax1.get_ylim()

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

    fig.tight_layout()
    plt.savefig(f"results/{os.path.basename(csv)[0:-4]}_results_wenner_plc1000.jpg")
