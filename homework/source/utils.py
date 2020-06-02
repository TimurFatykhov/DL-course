import numpy as np
import matplotlib.pyplot as plt

def get_y(x, std):
    std = np.sqrt(std)
    y = np.exp(-0.5*(x/std)**2) / np.sqrt(2*np.pi) / std
    return y

x = np.concatenate([np.linspace(-10, -2, 100), 
                    np.linspace(-2, 2, 400),
                    np.linspace(2, 10, 100)])
stds = np.arange(0.25, 10.25, 0.25)
ys = dict([[std, get_y(x, std)] for std in stds])

# show line-plot
fig = plt.figure(figsize=(5,3))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-10, 10)
ax.set_ylim(-0.1, 1.1)
line, = ax.plot(x, ys[1.0])

# show bar-plot
# widths = x[1:] - x[0:-1]
# widths = np.concatenate([widths, [0]])
# bar = ax.bar(x, ys[1.0], widths, align='edge')

def normal_pdf_slider(отклонение = 1.0):
    line.set_ydata(ys[отклонение])
    fig.canvas.draw_idle()