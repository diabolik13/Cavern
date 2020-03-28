import time

from elasticity import *
from animate_plot import *


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# fig, ax = plt.subplots()
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# ax.set_aspect('equal', 'box')
# ax.set(xlim=(min(x), max(x)), ylim=(min(y), max(y)))
# triang = mtri.Triangulation(xc, yc, t.transpose())
# c = ax.tricontourf(triang, z[:, i], 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
#                    levels=np.linspace(np.min(z), np.max(z), l))
# ax.locator_params(axis='both', nbins=3)
# ax.triplot(triang, color='white', lw=0.1)
# cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
# cbar.set_label('label', fontsize=16)
# cbar.ax.ticklabel_format(useMathText=True)
# ax.set_xlabel('x [m]', fontsize=16)
# ax.set_ylabel('y [m]', fontsize=16)
# ax.ticklabel_format(useMathText=True)


sys.stdout = Logger("log.txt")
time_start = time.time()
mesh_filename = 'new_cave.msh'  # supported formats: *.mat and *.msh
input_param = load_input(mesh_filename)
output = calculate_creep(input_param)
elapsed_1 = time.time() - time_start
time_implicit = time.time()
# output_NR = calculate_creep_NR(input_param)
elapsed_2 = time.time() - time_implicit
# diff_d = np.max(abs(output['displacement'] - output_NR['displacement']))
# diff_e = np.max(abs(output['strain'] - output_NR['strain']))
# diff_s = np.max(abs(output['stress'] - output_NR['stress']))

print("\nExplicit simulation is done in {} seconds.\n"
      "Implicit simulation is done in {} seconds.\n"
      "Total simulation time is {} days.\n"
      "\nMaximum displacement is {} m, creep displacement is {} m."
      .format(float("{0:.2f}".format(elapsed_1)),
              float("{0:.2f}".format(elapsed_2)),
              float("{0:.2f}".format((output['elapsed time'][-1] / 86400))),
              float("{0:.3f}".format(np.max(abs(output['displacement'])))),
              float("{0:.1e}".format(np.max(abs(output['displacement'][:, -1] - output['displacement'][:, 0]))))))
# print("\n Difference in explicit and implicit results (maximum absolute value): "
#       "\n Displacement: {0:.1e} m,".format(diff_d),
#       "\n Strain: {0:.1e},".format(diff_e),
#       "\n Stress: {0:.1e} Pa.\n ".format(diff_s))

# write_results(input_param, output, 15, '.png', exaggerate=True)
# write_results_xdmf(input_param, output)
# save_plot(input_param, output, 228)
print('Done in ' + str(time.time() - time_start) + ' seconds.')
