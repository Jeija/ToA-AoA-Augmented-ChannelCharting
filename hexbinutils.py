import matplotlib.transforms as mtransforms
import matplotlib.cbook as cbook
import numpy as np
import math

def hexbin_colors(x, y, C, gridsize=100, xscale='linear',
            yscale='linear', extent=None, reduce_C_function=np.mean,
            mincnt=None):
    x, y, C = cbook.delete_masked_points(x, y, C)

    # Set the size of the hexagon grid
    if np.iterable(gridsize):
        nx, ny = gridsize
    else:
        nx = gridsize
        ny = int(nx / math.sqrt(3))
    # Count the number of data in each hexagon
    x = np.array(x, float)
    y = np.array(y, float)

    if xscale == 'log':
        if np.any(x <= 0.0):
            raise ValueError("x contains non-positive values, so can not"
                                " be log-scaled")
        x = np.log10(x)
    if yscale == 'log':
        if np.any(y <= 0.0):
            raise ValueError("y contains non-positive values, so can not"
                                " be log-scaled")
        y = np.log10(y)
    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin, xmax = (np.min(x), np.max(x)) if len(x) else (0, 1)
        ymin, ymax = (np.min(y), np.max(y)) if len(y) else (0, 1)

        # to avoid issues with singular data, expand the min/max pairs
        xmin, xmax = mtransforms.nonsingular(xmin, xmax, expander=0.1)
        ymin, ymax = mtransforms.nonsingular(ymin, ymax, expander=0.1)

    # In the x-direction, the hexagons exactly cover the region from
    # xmin to xmax. Need some padding to avoid roundoff errors.
    padding = 1.e-9 * (xmax - xmin)
    xmin -= padding
    xmax += padding
    sx = (xmax - xmin) / nx
    sy = (ymax - ymin) / ny

    x = (x - xmin) / sx
    y = (y - ymin) / sy
    ix1 = np.round(x).astype(int)
    iy1 = np.round(y).astype(int)
    ix2 = np.floor(x).astype(int)
    iy2 = np.floor(y).astype(int)

    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny
    n = nx1 * ny1 + nx2 * ny2

    d1 = (x - ix1) ** 2 + 3.0 * (y - iy1) ** 2
    d2 = (x - ix2 - 0.5) ** 2 + 3.0 * (y - iy2 - 0.5) ** 2
    bdist = (d1 < d2)
    if C is None:
        lattice1 = np.zeros((nx1, ny1))
        lattice2 = np.zeros((nx2, ny2))
        c1 = (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1) & bdist
        c2 = (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2) & ~bdist
        np.add.at(lattice1, (ix1[c1], iy1[c1]), 1)
        np.add.at(lattice2, (ix2[c2], iy2[c2]), 1)
        if mincnt is not None:
            lattice1[lattice1 < mincnt] = np.nan
            lattice2[lattice2 < mincnt] = np.nan
        accum = np.concatenate([lattice1.ravel(), lattice2.ravel()])
        good_idxs = ~np.isnan(accum)

    else:
        if mincnt is None:
            mincnt = 0

        # create accumulation arrays
        lattice1 = np.empty((nx1, ny1), dtype=object)
        for i in range(nx1):
            for j in range(ny1):
                lattice1[i, j] = []
        lattice2 = np.empty((nx2, ny2), dtype=object)
        for i in range(nx2):
            for j in range(ny2):
                lattice2[i, j] = []

        for i in range(len(x)):
            if bdist[i]:
                if 0 <= ix1[i] < nx1 and 0 <= iy1[i] < ny1:
                    lattice1[ix1[i], iy1[i]].append(C[i])
            else:
                if 0 <= ix2[i] < nx2 and 0 <= iy2[i] < ny2:
                    lattice2[ix2[i], iy2[i]].append(C[i])

        for i in range(nx1):
            for j in range(ny1):
                vals = lattice1[i, j]
                if len(vals) > mincnt:
                    lattice1[i, j] = reduce_C_function(vals)
                else:
                    lattice1[i, j] = np.nan
        for i in range(nx2):
            for j in range(ny2):
                vals = lattice2[i, j]
                if len(vals) > mincnt:
                    lattice2[i, j] = reduce_C_function(vals)
                else:
                    lattice2[i, j] = np.nan

        accum = np.concatenate([lattice1.astype(float).ravel(),
                                lattice2.astype(float).ravel()])
        good_idxs = ~np.isnan(accum)

    offsets = np.zeros((n, 2), float)
    offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
    offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
    offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
    offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
    offsets[:, 0] *= sx
    offsets[:, 1] *= sy
    offsets[:, 0] += xmin
    offsets[:, 1] += ymin
    # remove accumulation bins with no data
    offsets = offsets[good_idxs, :]
    accum = accum[good_idxs]

    return accum