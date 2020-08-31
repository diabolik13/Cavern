import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.tri as mtri
import meshio
import inspect

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import (AnchoredOffsetbox, DrawingArea, HPacker,
                                  TextArea)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def write_results(nt, mesh, output, folder, ext1, amp=False, l=10):
    """Saves results in one file in *.xdmf format for ParaView."""
    from pathlib import Path

    x, y = mesh.coordinates()
    t = mesh.cells()
    nnodes = mesh.nnodes()

    if nt > 1:
        time = np.round((output['elapsed time'] / 86400), 2)
    elif nt == 1:
        time = np.array([0])

    data = {
        0: {
            0: {
                "title": 'u_x',
                "units": '[m]',
                "value": output['displacement'][:nnodes]
            },
            1: {
                "title": 'u_y',
                "units": '[m]',
                "value": output['displacement'][nnodes:]
            }
        },
        1: {
            0: {
                "title": 'e_x',
                "units": '[-]',
                "value": output['strain'][:nnodes]
            },
            1: {
                "title": 'e_y',
                "units": '[-]',
                "value": output['strain'][nnodes:2 * nnodes]
            },
            2: {
                "title": 'e_sh',
                "units": '[-]',
                "value": output['strain'][2 * nnodes:3 * nnodes]
            }
        },
        2: {
            0: {
                "title": 's_x',
                "units": '[Pa]',
                "value": output['stress'][:nnodes]
            },
            1: {
                "title": 's_y',
                "units": '[Pa]',
                "value": output['stress'][nnodes:2 * nnodes]
            },
            2: {
                "title": 's_sh',
                "units": '[Pa]',
                "value": output['stress'][2 * nnodes:3 * nnodes]
            }
        },
        3: {
            0: {
                "title": 's_vm',
                "units": '[Pa]',
                "value": output['Von Mises stress']
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

    if ext1 == '.xdmf':
        Path('./output/' + folder).mkdir(parents=True, exist_ok=True)
        with meshio.xdmf.TimeSeriesWriter('./output/' + folder + '/output_data.xdmf') as writer:
            writer.write_points_cells(mesh.meshdata().points, mesh.meshdata().cells)
            for i in range(nt):
                # if i > 0:  # remove the very first frame with linear elastic response only
                writer.write_data(time[i],
                                  point_data={
                                      data[0][0]['title'] + ', ' + data[0][0]['units']: data[0][0]['value'][:, i],
                                      data[0][1]['title'] + ', ' + data[0][1]['units']: data[0][1]['value'][:, i],
                                      data[1][0]['title'] + ', ' + data[1][0]['units']: data[1][0]['value'][:, i],
                                      data[1][1]['title'] + ', ' + data[1][1]['units']: data[1][1]['value'][:, i],
                                      data[1][2]['title'] + ', ' + data[1][2]['units']: data[1][2]['value'][:, i],
                                      data[2][0]['title'] + ', ' + data[2][0]['units']: data[2][0]['value'][:, i],
                                      data[2][1]['title'] + ', ' + data[2][1]['units']: data[2][1]['value'][:, i],
                                      data[2][2]['title'] + ', ' + data[2][2]['units']: data[2][2]['value'][:, i],
                                      data[3][0]['title'] + ', ' + data[3][0]['units']: data[3][0]['value'][:, i],
                                      data[4][0]['title'] + ', ' + data[4][0]['units']: data[4][0]['value'][:, i],
                                      data[4][1]['title'] + ', ' + data[4][1]['units']: data[4][1]['value'][:, i],
                                  })

    if ext1 == '.gif' or ext1 == '.png':

        def ex(i):
            return np.log(i + 10) * 30

        ext = ext1
        if nt < 2:
            del data[4]
        Path('./output/' + folder).mkdir(parents=True, exist_ok=True)
        for k in data.keys():
            var = data[k]
            for j in range(len(var)):
                label = var[j]['title']
                units = var[j]['units']
                z = var[j]['value']

                # fig, ax = plt.subplots()
                fig = plt.figure(figsize=(11, 9))
                ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
                # fig.tight_layout()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)

                def animate(i):
                    ax.cla()
                    plt.cla()
                    # fig.tight_layout()
                    ax.set_aspect('equal', 'box')
                    ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                    if amp == True:
                        xc = x + ex(i) * data[0][0]['value'][:, i]
                        # yc = y + ex(i) * data[0][1]['value'][:, i]
                        yc = y
                    else:
                        xc, yc = x, y
                    triang = mtri.Triangulation(xc, yc, t.transpose())
                    c = ax.tricontourf(triang, z[:, i], l, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                                       levels=np.linspace(np.min(z), np.max(z), l))
                    ax.locator_params(axis='both', nbins=3)
                    ax.tick_params(axis='both', which='major', labelsize=30)
                    ax.tick_params(axis='both', which='minor', labelsize=16)
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    ax.triplot(triang, color='white', lw=0.1)
                    # ax.set_title(
                    #     label + ', elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                    #     fontsize=30)
                    ax.set_title(
                        'elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                        fontsize=30)
                    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
                    cbar.set_label(label + ' magnitude ' + units, fontsize=30)
                    cbar.ax.tick_params(labelsize=30)
                    # cbar.ax.ticklabel_format(useMathText=True)
                    ax.set_xlabel('x', fontsize=30)
                    ax.set_ylabel('z', fontsize=30)
                    ax.ticklabel_format(useMathText=True)

                # if nt < 20:
                #     anim = FuncAnimation(
                #         fig, animate, interval=100, frames=nt)
                #     anim.save('./output/' + folder + '/' + label + ext, writer='imagemagick')
                # else:
                #     anim = FuncAnimation(
                #         fig, animate, interval=100, frames=20)
                #     anim.save('./output/' + folder + '/' + label + ext, writer='imagemagick')

                anim = FuncAnimation(
                    fig, animate, interval=100, frames=nt)
                anim.save('./output/' + folder + '/' + label + ext, writer='imagemagick')

    if ext1 == '.eps':

        def ex(i):
            return np.log(i + 10) * 30

        ext = ext1
        if nt < 2:
            del data[4]
        Path('./output/' + folder).mkdir(parents=True, exist_ok=True)
        for k in data.keys():
            var = data[k]
            for j in range(len(var)):
                for i in range(nt):
                    label = var[j]['title']
                    units = var[j]['units']
                    z = var[j]['value']

                    # fig, ax = plt.subplots()
                    fig = plt.figure(figsize=(11, 9))
                    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
                    # fig.tight_layout()
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)

                    # def animate(i):
                    #     ax.cla()
                    #     plt.cla()
                    #     # fig.tight_layout()
                    #     ax.set_aspect('equal', 'box')
                    #     ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                    #     if amp == True:
                    #         xc = x + ex(i) * data[0][0]['value'][:, i]
                    #         # yc = y + ex(i) * data[0][1]['value'][:, i]
                    #         yc = y
                    #     else:
                    #         xc, yc = x, y
                    #     triang = mtri.Triangulation(xc, yc, t.transpose())
                    #     c = ax.tricontourf(triang, z[:, i], l, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                    #                        levels=np.linspace(np.min(z), np.max(z), l))
                    #     ax.locator_params(axis='both', nbins=3)
                    #     ax.tick_params(axis='both', which='major', labelsize=30)
                    #     ax.tick_params(axis='both', which='minor', labelsize=16)
                    #     ax.axes.xaxis.set_ticks([])
                    #     ax.axes.yaxis.set_ticks([])
                    #     ax.triplot(triang, color='white', lw=0.1)
                    #     # ax.set_title(
                    #     #     label + ', elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                    #     #     fontsize=30)
                    #     ax.set_title(
                    #         'elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                    #         fontsize=30)
                    #     cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
                    #     cbar.set_label(label + ' magnitude ' + units, fontsize=30)
                    #     cbar.ax.tick_params(labelsize=30)
                    #     # cbar.ax.ticklabel_format(useMathText=True)
                    #     ax.set_xlabel('x', fontsize=30)
                    #     ax.set_ylabel('z', fontsize=30)
                    #     ax.ticklabel_format(useMathText=True)
                    #
                    # anim = FuncAnimation(
                    #     fig, animate, interval=100, frames=nt)
                    # anim.save('./output/' + folder + '/' + label + ext, writer='imagemagick')

                    ax.cla()
                    plt.cla()
                    # fig.tight_layout()
                    ax.set_aspect('equal', 'box')
                    ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
                    if amp == True:
                        xc = x + ex(0) * data[0][0]['value'][:, i]
                        # yc = y + ex(i) * data[0][1]['value'][:, i]
                        yc = y
                    else:
                        xc, yc = x, y
                    triang = mtri.Triangulation(xc, yc, t.transpose())
                    c = ax.tricontourf(triang, z[:, i], l, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                                       levels=np.linspace(np.min(z), np.max(z), l))
                    ax.locator_params(axis='both', nbins=3)
                    ax.tick_params(axis='both', which='major', labelsize=30)
                    ax.tick_params(axis='both', which='minor', labelsize=16)
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                    # ax.triplot(triang, color='white', lw=0.1)
                    # ax.set_title(
                    #     label + ', elapsed time ' + "{:10.2f}".format((output['elapsed time'][i] / 86400)) + ' days.\n',
                    #     fontsize=30)
                    # ax.set_title(
                    #     'elapsed time ' + "{:10.2f}".format((output['elapsed time'][0] / 86400)) + ' days.\n',
                    #     fontsize=30)
                    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
                    cbar.set_label(label + units, fontsize=30)
                    cbar.ax.tick_params(labelsize=30)
                    # cbar.ax.ticklabel_format(useMathText=True)
                    ax.set_xlabel('x', fontsize=30)
                    ax.set_ylabel('z', fontsize=30)
                    ax.ticklabel_format(useMathText=True)
                    plt.savefig('./output/' + folder + '/' + label + str(i) + ext, format='eps')

    print("Done writing results to output files in the ./output/" + folder + " folder.")


def save_plot(input, output, node):
    """Saves results for one particular point."""

    from matplotlib.ticker import ScalarFormatter

    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

    nt = input['number of time steps']
    p = input['points']
    t = input['elements']

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])

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

    for k in range(iter):
        var = data[k]
        for j in range(len(var)):
            folder = './output/A/'
            label = var[j]['title']
            units = var[j]['units']
            z = var[j]['value']

            # fig, ax = plt.subplots(constrained_layout=True)
            # fig.tight_layout()
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_axes([0.2, 0.15, 0.7, 0.7])
            # ax.cla()
            # plt.cla()
            # ax.set_aspect('equal', 'box')
            ax.set(xlim=(np.min(output['elapsed time']) / 86400,
                         np.max(output['elapsed time']) / 86400))
            # ylim=(np.min(z[node]), np.max(z[node])))

            ax.plot(output['elapsed time'] / 86400, z[node], 'ro-')
            # yfmt = ScalarFormatterForceFormat()
            # yfmt.set_powerlimits((0, 0))
            # ax.yaxis.set_major_formatter(yfmt)
            # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2e}'))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(useMathText=True)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.tick_params(axis='both', which='minor', labelsize=12)
            ax.locator_params(axis='both', nbins=5)
            ax.grid(True)
            ax.set_xlabel('elapsed time, [days]', fontsize=16)
            ax.set_ylabel(label + ', ' + units, fontsize=16)
            plt.savefig(folder + label + '_A.png')
            plt.close('all')
            # plt.show()

    print('Done writing results to the output files.')


def save_plot2(input, output, node):
    """Saves results for one particular point."""

    from matplotlib.ticker import ScalarFormatter

    class ScalarFormatterForceFormat(ScalarFormatter):
        def _set_format(self):  # Override function that finds format to use.
            self.format = "%1.1f"  # Give format here

    nt = input['number of time steps']
    p = input['points']
    t = input['elements']

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    nnodes = len(p[0])

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

    for k in range(iter):
        var = data[k]
        for j in range(len(var)):
            folder = './output/'
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

            ax.plot(output['elapsed time'] / 86400, z[node], 'ro-')
            # yfmt = ScalarFormatterForceFormat()
            # yfmt.set_powerlimits((0, 0))
            # ax.yaxis.set_major_formatter(yfmt)
            # ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2e}'))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.ticklabel_format(useMathText=True)
            ax.grid(True)
            ax.set_xlabel('elapsed time, [days]', fontsize=20)
            ax.set_ylabel(label + ', ' + units, fontsize=20)
            plt.savefig(folder + label + '_vs_disp.png')
            # plt.show()

    print('Done writing results to the output files.')
