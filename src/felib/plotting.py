import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from numpy.typing import NDArray

from .element import IsoparametricElement
from .element.geom import Pn


def rplot1(p: NDArray, r: NDArray) -> None:
    """Make plots of reactions on left and right edges of a uniform square"""
    _, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Left reaction
    ilo = [n for n, x in enumerate(p) if isclose(x[0], -1.0)]
    ylo = p[ilo, 1]
    rlo = r[ilo]
    ix = np.argsort(ylo)
    ylo = ylo[ix]
    rlo = rlo[ix]

    axs[0].plot(ylo, rlo, "o", label="LHS")
    axs[0].set_title("LHS reaction")
    axs[0].set_xlabel("y")
    axs[0].set_ylabel("Heat flux/reaction")
    axs[0].grid(True)

    # Right reaction
    ihi = [n for n, x in enumerate(p) if isclose(x[0], 1.0)]
    yhi = p[ihi, 1]
    rhi = r[ihi]
    ix = np.argsort(yhi)
    yhi = yhi[ix]
    rhi = rhi[ix]

    axs[1].plot(yhi, rhi, "o", label="RHS")
    axs[1].set_title("RHS reaction")
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("Heat flux/reaction")
    axs[1].grid(True)

    print(np.sum(rlo))
    print(np.sum(rhi))

    plt.tight_layout()
    plt.show()


def isclose(a, b, rtol: float = 0.0001, atol: float = 1e-8) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


def iso_plot_quad(
    element: IsoparametricElement,
    p: NDArray,
    connect: NDArray,
    z: NDArray,
    title: str = "FEA Solution",
    nplot: int = 12,
) -> None:
    """
    Generic isoparametric contour plot.

    Args:
        element : IsoparametricElement instance
        p       : global nodal coordinates (nnode_total, ndim)
        connect : element connectivity (nel, element.nnode)
        z       : global nodal scalar field
        title   : plot title
        nplot   : resolution per element in reference space
    """

    fig, ax = plt.subplots(figsize=(7, 5))

    # Reference sampling grid
    xi = np.linspace(-1.0, 1.0, nplot)
    eta = np.linspace(-1.0, 1.0, nplot)
    XI, ETA = np.meshgrid(xi, eta)

    # Flatten for vectorized shape evaluation
    sample_pts = np.column_stack([XI.ravel(), ETA.ravel()])

    for elem in connect:
        pe = p[elem]  # (nnode, ndim)
        ze = z[elem]  # (nnode,)

        X = np.zeros(sample_pts.shape[0])
        Y = np.zeros(sample_pts.shape[0])
        Z = np.zeros(sample_pts.shape[0])

        for i, xi_pt in enumerate(sample_pts):
            # geometry mapping
            xy = element.interpolate(pe, xi_pt)
            X[i], Y[i] = xy[:2]

            # field interpolation
            N = element.shape(xi_pt)
            Z[i] = np.dot(N, ze)

        # reshape to grid
        X = X.reshape(XI.shape)
        Y = Y.reshape(YI := ETA.shape)  # keeps same shape
        Z = Z.reshape(XI.shape)

        ax.pcolormesh(X, Y, Z, shading="gouraud", cmap="turbo")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def iso_plot_tri3(
    element: Pn,
    p: NDArray,
    connect: NDArray,
    z: NDArray,
    title: str = "FEA Solution",
    nplot: int = 12,
) -> None:

    fig, ax = plt.subplots(figsize=(7, 5))

    for elem in connect:
        pe = p[elem]
        ze = z[elem]

        # reference sampling inside triangle
        xi = np.linspace(0, 1, nplot)
        _pts = []
        for i in range(nplot):
            for j in range(nplot - i):
                _pts.append([xi[i], xi[j]])

        pts = np.array(_pts)

        # shape functions evaluated at sample points
        N = np.array([element.shape(pt) for pt in pts])

        # geometry mapping
        X = N @ pe[:, 0]
        Y = N @ pe[:, 1]

        # field interpolation
        Z = N @ ze

        # triangulate sampled triangle
        triang = tri.Triangulation(X, Y)

        ax.tricontourf(triang, Z, levels=50, cmap="turbo")

        # element edges
        ax.plot(
            np.append(pe[:, 0], pe[0, 0]),
            np.append(pe[:, 1], pe[0, 1]),
            color="k",
            linewidth=0.2,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def tplot(p: NDArray, t: NDArray, z: NDArray, title: str = "FEA Solution") -> None:
    from .element.geom import P3

    return iso_plot_tri3(P3(), p, t, z, title=title)


def tplot3d(
    p: NDArray, t: NDArray, z: NDArray, label: str | None = None, title: str = "FE Solution"
) -> None:
    """Make temperature contour plot"""
    triang = tri.Triangulation(p[:, 0], p[:, 1], t)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_trisurf(  # type: ignore
        triang, z, cmap="turbo", linewidth=0.2, antialiased=True
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if label:
        ax.set_zlabel(label)  # type: ignore
    fig.colorbar(surf, ax=ax, shrink=0.6, label=label)

    plt.title(title)
    plt.show()

    plt.clf()
    plt.cla()
    plt.close("all")
