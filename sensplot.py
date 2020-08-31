# Program to extract a particular row value
import xlrd
import matplotlib.pyplot as plt

loc = ("data_base_case.xlsx")

wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(2)
sheet3 = wb.sheet_by_index(3)
sheet4 = wb.sheet_by_index(4)
sheet5 = wb.sheet_by_index(5)

# sheet 2
timeline = sheet.row_values(3)[1:-1]
d1 = sheet.row_values(0)[1:-1]
d2 = sheet.row_values(1)[1:-1]

d3 = sheet.row_values(5)[1:-1]
d4 = sheet.row_values(8)[1:-1]

d5 = sheet.row_values(16)[1:-1]
d6 = sheet.row_values(19)[1:-1]

# sheet 3
d7 = sheet3.row_values(5)[1:-1]
d8 = sheet3.row_values(8)[1:-1]

# sheet 4
d9 = sheet4.row_values(5)[1:-1]
d10 = sheet4.row_values(8)[1:-1]

# sheet 5
d11 = sheet5.row_values(5)[1:-1]
d12 = sheet5.row_values(8)[1:-1]

fig, ax = plt.subplots()
plt.plot(timeline, d1, label='Point A')
plt.plot(timeline, d2, label='Point B')
plt.xlabel('Time, days', fontsize=16)
plt.ylabel('Displacement, m', fontsize=16)
# plt.title('Interesting Graph\nCheck it out')
# plt.set_aspect('equal', 'box')
plt.tight_layout()
plt.legend()
ax.set_aspect('auto')
# plt.show()
plt.savefig('plot1.eps', format='eps')
# plt.cla()

fig, ax = plt.subplots()
plt.plot(timeline, d1, label='depth 600 m')
plt.plot(timeline, d3, label='depth 300 m')
plt.plot(timeline, d4, label='depth 900 m')
plt.xlabel('Time, days', fontsize=16)
plt.ylabel('Displacement, m', fontsize=16)
# plt.title('Interesting Graph\nCheck it out')
# plt.set_aspect('equal', 'box')
plt.tight_layout()
plt.legend()
ax.set_aspect('auto')
# plt.show()
plt.savefig('plot2.eps', format='eps')
# plt.cla()

fig, ax = plt.subplots()
plt.plot(timeline, d1, label='Point A1')
plt.plot(timeline, d2, label='Point B1')
plt.plot(timeline, d5, label='Point A2')
plt.plot(timeline, d6, label='Point B2')
plt.xlabel('Time, days', fontsize=16)
plt.ylabel('Displacement, m', fontsize=16)
# plt.title('Interesting Graph\nCheck it out')
# plt.set_aspect('equal', 'box')
plt.tight_layout()
plt.legend()
ax.set_aspect('auto')
# plt.show()
plt.savefig('plot3.eps', format='eps')
# plt.cla()

fig, ax = plt.subplots()
plt.plot(timeline, d1, label='K = 24.3 GPa')
plt.plot(timeline, d7, label='K = K+0.5 K')
plt.plot(timeline, d8, label='K = K-0.5 K')
plt.xlabel('Time, days', fontsize=16)
plt.ylabel('Displacement, m', fontsize=16)
# plt.title('Interesting Graph\nCheck it out')
# plt.set_aspect('equal', 'box')
plt.tight_layout()
plt.legend()
ax.set_aspect('auto')
# plt.show()
plt.savefig('plot4.eps', format='eps')
# plt.cla()

fig, ax = plt.subplots()
plt.plot(timeline, d1, label='T = 33.8 C')
plt.plot(timeline, d9, label='T = 24.4 C')
plt.plot(timeline, d10, label='T = 43.2 C')
plt.xlabel('Time, days', fontsize=16)
plt.ylabel('Displacement, m', fontsize=16)
# plt.title('Interesting Graph\nCheck it out')
# plt.set_aspect('equal', 'box')
plt.tight_layout()
plt.legend()
ax.set_aspect('auto')
# plt.show()
plt.savefig('plot5.eps', format='eps')
# plt.cla()

fig, ax = plt.subplots()
plt.plot(timeline, d1, label='G = 11.2 GPa')
plt.plot(timeline, d11, label='G = G+0.5G')
plt.plot(timeline, d12, label='G = G-0.5G')
plt.xlabel('Time, days', fontsize=16)
plt.ylabel('Displacement, m', fontsize=16)
# plt.title('Interesting Graph\nCheck it out')
# plt.set_aspect('equal', 'box')
plt.tight_layout()
plt.legend()
ax.set_aspect('auto')
# plt.show()
plt.savefig('plot6.eps', format='eps')
# plt.cla()


