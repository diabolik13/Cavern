import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.collections
import warnings
import xlsxwriter

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D


def impose_dirichlet(k, f, d_bnd):
    """
    Impose Dirichlet boundary conditions.
    """

    k[d_bnd, :] = 0
    k[:, d_bnd] = 0
    k[d_bnd, d_bnd] = 1
    f[d_bnd] = 0

    return k, f


def deviatoric_stress(stress):
    """
    Calculate deviatoric stress.
    """

    stressx = stress[0]
    stressy = stress[1]
    stressxy = stress[2]
    dstressx = stressx - 0.5 * (stressx + stressy)
    dstressy = stressy - 0.5 * (stressx + stressy)
    dstressxy = stressxy
    dstress = np.array([dstressx, dstressy, dstressxy])

    return dstress


def von_mises_stress(stress):
    """
    Calculate von Mises stress.
    """
    dstress = deviatoric_stress(stress)
    stressx = stress[0]
    stressy = stress[1]
    stressxy = stress[2]
    svm = np.sqrt(np.square(stressx) - stressx * stressy + np.square(stressy) + 3 * np.square(stressxy))
    # svm = np.sqrt(3 / 2 * np.sum((np.transpose(dstress) * np.transpose(dstress)), axis=1))

    return svm


def solve_disp(k, f):
    """
    Gaussian elimination to solve for displacements.
    """
    return np.linalg.solve(k, f)


def showMeshPlot(nodes, elements, values):
    """
    Plot the input parameter 'values' elementwise.
    """
    x = nodes[0, :]
    y = nodes[1, :]

    def quatplot(y, z, quatrangles, values, ax=None, **kwargs):
        quatrangles = np.transpose(quatrangles)
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

    pc = quatplot(x, y, np.asarray(elements), values, ax=ax,
                  edgecolor="crimson", cmap="rainbow", linewidth=0.1)
    # fig.colorbar(pc, ax=ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # ax.plot(y, z, marker=".", ls="", color="crimson")
    ax.plot(x, y, ls="", color="crimson")
    # plt.colorbar(pc, cax=cax)
    # cbar = plt.colorbar(pc, cax=cax, format='%.0e', ticks=np.linspace(np.min(values), np.max(values), 3))
    cbar = plt.colorbar(pc, cax=cax, format='%.1f', ticks=np.linspace(np.min(values), np.max(values), 3))
    cbar.set_label('Impurity content, [-]', fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    # ax.set(title='Mechanical properties heterogeneity')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # for item in [fig, ax]:
    #     item.patch.set_visible(False)

    plt.show()


def save_plot_A(nt, mesh, output, folder, node):
    """
    Saves results for one particular point.
    """

    from matplotlib.ticker import ScalarFormatter
    from pathlib import Path

    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

    p = mesh.meshdata().points
    t = mesh.meshdata().cells

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = mesh.nnodes()

    data = {
        0: {
            0: {
                "title": 'displacement_x',
                "units": '[m]',
                "value": output['displacement'][:nnodes]
            },
            1: {
                "title": 'displacement_y',
                "units": '[m]',
                "value": output['displacement'][nnodes:]
            }
        },
        1: {
            0: {
                "title": 'strain_x',
                "units": '[-]',
                "value": output['strain'][:nnodes]
            },
            1: {
                "title": 'strain_y',
                "units": '[-]',
                "value": output['strain'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'strain_shear',
                "units": '[-]',
                "value": output['strain'][2 * nnodes:3 * nnodes]
            }
        },
        2: {
            0: {
                "title": 'stress_x',
                "units": '[Pa]',
                "value": output['stress'][:nnodes]
            },
            1: {
                "title": 'stress_y',
                "units": '[Pa]',
                "value": output['stress'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'stress_shear',
                "units": '[Pa]',
                "value": output['stress'][2 * nnodes:3 * nnodes]
            }
        },
        3: {
            0: {
                "title": 'von_mises_stress',
                "units": '[Pa]',
                "value": output['Von Mises stress'][:nnodes]
            }
        },
        4: {
            0: {
                "title": 'creep_forces_x',
                "units": '[N]',
                "value": output['creep forces'][:nnodes]
            },
            1: {
                "title": 'creep_forces_y',
                "units": '[N]',
                "value": output['creep forces'][nnodes:]
            }
        }
    }

    if nt > 1:
        iter = len(data)
    elif nt == 1:
        iter = len(data) - 1

    Path('./output/' + folder).mkdir(parents=True, exist_ok=True)
    folder = './output/' + folder + '/'
    for k in range(iter):
        var = data[k]
        for j in range(len(var)):
            label = var[j]['title']
            units = var[j]['units']
            z = var[j]['value']

            # fig, ax = plt.subplots(constrained_layout=True)
            # fig.tight_layout()
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.cla()
            plt.cla()
            # ax.set_aspect('equal', 'box')
            # ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))

            ax.plot(output['elapsed time'] / 86400, z[node], 'rx-')
            # yfmt = ScalarFormatterForceFormat()
            # yfmt.set_powerlimits((0, 0))
            # ax.yaxis.set_major_formatter(yfmt)
            # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2e}'))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(useMathText=True)
            ax.grid(True)
            ax.set_xlabel('elapsed time, [days]', fontsize=20)
            ax.set_ylabel(label + ', ' + units, fontsize=20)
            # if label == 'strain_x':
            #     plt.ylim(-2.2e-5, -1e-5)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlim(0, np.max(output['elapsed time'] / 86400))

            plt.savefig(folder + label + '_vs_disp.eps', format='eps')
            # plt.show()
            plt.close(fig)

    print('Done writing results to the output files.')


def plot_parameter(mesh, f, folder=None, u=0, amp=0, l=10):
    """
    Plot parameter 'f' evaluated at nodal points.
    """
    p = mesh.coordinates()
    t = mesh.cells()
    f = f.reshape((mesh.nnodes(),))
    x, y = p
    if not (u == 0):
        x = x + u[::2].reshape((len(p[0]),)) * amp
        y = y + u[1::2].reshape((len(p[0]),)) * amp

    # Plot the triangulation.
    triang = mtri.Triangulation(x, y, t.transpose())

    # Set up the figure
    fig, axs = plt.subplots(nrows=1, ncols=1)
    lw = 0.2

    im = axs.tricontourf(triang, f, l, cmap='plasma')
    axs.triplot(triang, lw=lw)
    # axs.set_title('Plot')
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if folder == (not None):
        plt.savefig('./output/' + folder + "/plot.png")
    axs.axes.xaxis.set_ticks([])
    axs.axes.yaxis.set_ticks([])
    axs.set_xlabel('x [m]', fontsize=16)
    axs.set_ylabel('y [m]', fontsize=16)
    cbar = plt.colorbar(im, cax=cax, format='%.2f', ticks=np.linspace(np.min(f), np.max(f), 3))
    cbar.set_label('Impurity content', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.show()


def check_matrix(k):
    """Checks stiffness matrix.

    Returns:
        nz_k: number of nonzero values.
        det: determinant of k.
    """

    nz_k = np.count_nonzero(k)
    det = np.linalg.det(k)
    txt = "Number of nonzero values in k = {}, det = {}".format(nz_k, det)
    plt.spy(k)
    plt.title(txt)
    plt.show()


def hist(a):
    """
    Show histogram of 'a'.
    """
    plt.hist(a, bins='auto')
    plt.title("Histogram with 'auto' bins")
    plt.show()


def write_xls(filename, output):
    """
    Write output results to *.xlsx spreadsheet.
    """
    workbook = xlsxwriter.Workbook('./output/' + filename.split(".")[0] + '/data.xlsx')
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    order = sorted(output.keys())
    for key in order:
        row += 1
        worksheet.write(row, col, key)
        for item in output[key]:
            row += 1
            i = 0
            if type(item) == np.ndarray:
                for value in item:
                    worksheet.write(row, col + i, value)
                    i += 1
            else:
                worksheet.write(row, col + i, item)
                i += 1

    workbook.close()


def polyarea(coord):
    """
    Calculate are of fe with given corner points coordinates 'coord'.
    """
    x, y = coord
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def el_tenzor(e, nu, eltno=None, imp_1=None, imp_2=None):
    """
    Evaluate elasticity tensor for a given element.
    """
    if eltno == None:
        d = e / (1 - nu ** 2) * np.array([[1, nu, 0],
                                          [nu, 1, 0],
                                          [0, 0, (1 - nu) / 2]])
    else:
        if imp_1 is not None:
            E = (imp_1[eltno] * e[0] + imp_2[eltno] * e[1] + (1 - imp_1[eltno] - imp_2[eltno]) * e[2])
            NU = (imp_1[eltno] * nu[0] + imp_2[eltno] * nu[1] + (1 - imp_1[eltno] - imp_2[eltno]) * nu[2])
            d = E / (1 - NU ** 2) * np.array([[1, NU, 0],
                                              [NU, 1, 0],
                                              [0, 0, (1 - NU) / 2]])
        else:
            # d = e[eltno] / (1 - nu[eltno] ** 2) * np.array([[1, nu[eltno], 0],
            #                                                 [nu[eltno], 1, 0],
            #                                                 [0, 0, (1 - nu[eltno]) / 2]])
            d = e / (1 - nu ** 2) * np.array([[1, nu, 0],
                                              [nu, 1, 0],
                                              [0, 0, (1 - nu) / 2]])
    return d
