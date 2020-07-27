import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np


def showMeshPlot(nodes, elements, values):
    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, values, ax=None, **kwargs):
        if not ax: ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        pc.set_array(values)
        ax.add_collection(pc)
        ax.autoscale()
        return pc

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    pc = quatplot(y, z, np.asarray(elements), values, ax=ax,
                  edgecolor="crimson", cmap="rainbow")
    fig.colorbar(pc, ax=ax)
    ax.plot(y, z, marker="o", ls="", color="crimson")

    ax.set(title='This is the plot for: quad', xlabel='Y Axis', ylabel='Z Axis')

    plt.show()


nodes = np.array([[0, 0.5], [0.5, 0], [0.5, 0.5], [0.5, 1], [1, 0.5]])
elements = np.array([[0, 1, 2], [0, 2, 3], [2, 4, 3], [1, 4, 2]])
stresses = np.array([1, 2, 3, 4])

showMeshPlot(nodes, elements, stresses)
