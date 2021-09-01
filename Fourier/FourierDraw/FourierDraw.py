import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *
from PIL import Image
import sys
import argparse

def detect_pattern(image_name, level=[200]):
    fig, ax = plt.subplots(1, 1)
    # read image to array, then get image border with contour
    im = array(Image.open(image_name).convert('L'))
    contour_plot = ax.contour(im, levels=level, colors='black', origin='image')
    plt.close()
    # get contour path
    contour_path = contour_plot.collections[0].get_paths()[0] # It has to be a polygon.
    x, y = contour_path.vertices[:, 0], contour_path.vertices[:, 1]
    # center the image
    x = x - min(x)
    y = y - min(y)
    x = x - max(x) / 2
    y = y - max(y) / 2
    return x, y

# Discrete Fourier Transform Series
def DTFS(z, k): ### k is the term number
    N = len(z)
    ak = 0+0j
    for n in range(N):
        ak += z[n] * (np.cos(-2*np.pi/N*n*k) + 1j*np.sin(-2*np.pi/N*n*k))
    ak = ak/N
    return ak

def DFT(x, y, m):
    z = x + 1j * y
    cn = []
    for k in range(-m, m+1):
        cn.append(DTFS(z,k))
    T = np.linspace(0, 2*np.pi, 100)
    px = []
    py = []
    for t in T:
        p = 0+0j
        for j in range(2*m+1):
            p += cn[j] * ( np.cos((j-m)*t) + 1j*np.sin((j-m)*t) )
        px.append(p.real)
        py.append(p.imag)

    n = np.linspace(-m,m,2*m+1)
    cn = np.array(cn) ### term
    r = abs(cn) ### amplitude
    p = np.angle(cn) ### first phase
    w = 2 * np.pi * n / len(z) ### omiga
    return px, py, n, r, w, p


def FS(ft,k): ### Fourier Series
    dt = 1/len(ft)
    ak = 0+0j
    for i in range(len(ft)):
        ak+=ft[i]*(np.cos(-2*np.pi*k*dt*i) + 1j*np.sin(-2*np.pi*k*dt*i))*dt
    return ak


def draw_epicycles(x, y, m, fig, ax, __type__ = "DFT"):
    ax.plot(x, y, linewidth=2, color='black')
    if __type__ == "DFT":
        px, py, n, r, w, p = DFT(x, y, m)
        # ax.plot(px, py, linewidth=2, color='red')

        outline, = ax.plot([], [], linewidth=1, color='red') 

        circles = []
        dots = []
        for i in range(len(n)):
            circle, = ax.plot([], [], linewidth=1, color='grey')
            if i == len(n)-1:
                dot, = ax.plot([], [], 'o', color='red')
            else:
                dot, = ax.plot([], [], 'o', color='grey')
            circles.append(circle)
            dots.append(dot)
        theta = np.linspace(0, 2*np.pi, 100)

        xx = []
        yy = []
        def Anim(t):
            center = [0,0]
            for i in range(len(n)):
                circles[i].set_data(center[0]+r[i]*np.cos(theta), center[1]+r[i]*np.sin(theta))
                center = [center[0] + r[i]*np.cos(p[i]+w[i]*t), center[1]+r[i]*np.sin(p[i]+w[i]*t)]
                dots[i].set_data([center[0],center[1]])

            xx.append(center[0])
            yy.append(center[1])
            outline.set_data(xx, yy)

        return animation.FuncAnimation(fig, Anim, frames=len(x), interval=10, repeat=False)

    elif __type__ == "FS":
        
        n = np.linspace(-m,m,2*m+1)
        Cx = []
        Cy = []
        for k in n:
            Cx.append(FS(x, k))
            Cy.append(FS(y, k))

        Cx = np.array(Cx)
        rx = abs(Cx)
        px = np.angle(Cx)
        wx = 2 * np.pi * n

        Cy = np.array(Cy)
        ry = abs(Cy)
        py = np.angle(Cy)
        wy = 2 * np.pi * n

        circles_x = []
        dots_x = []
        circles_y = []
        dots_y = []

        ref =[]

        dot, = ax.plot([], [], 'o', color='red')

        for i in range(len(n)):
            circle_x, = ax.plot([], [], linewidth=1, color='cadetblue')
            circle_y, = ax.plot([], [], linewidth=1, color='dimgray')
            if i == len(n)-1:
                dot_x, = ax.plot([], [], 'o', color='blue')
                dot_y, = ax.plot([], [], 'o', color='blue')
            else:
                dot_x, = ax.plot([], [], 'o', color='cadetblue')
                dot_y, = ax.plot([], [], 'o', color='dimgray')

            circles_x.append(circle_x)
            dots_x.append(dot_x)
            circles_y.append(circle_y)
            dots_y.append(dot_y)
            ref.append([[],[]])

        line_x, = ax.plot([], [], linewidth=1, color='blue', linestyle='--')
        line_y, = ax.plot([], [], linewidth=1, color='blue', linestyle='--')
        outline, = ax.plot([], [], linewidth=1, color='red') 

        theta = np.linspace(0, 2*np.pi, 100)
        n_frames = 100
        
        xx = []
        yy = []
        def Anim(t):
            dt = t / n_frames
            center_x = [0,0]
            center_y = [0,0]
            ### X 
            for i in range(len(n)):
                circles_x[i].set_data( center_x[0]+rx[i]*np.cos(theta), center_x[1]+rx[i]*np.sin(theta))
                center_x = [center_x[0]+rx[i]*np.cos(px[i]+wx[i]*dt), center_x[1] + rx[i]*np.sin(px[i]+wx[i]*dt)]
                dots_x[i].set_data([center_x[0], center_x[1]])

            ### Y
                circles_y[i].set_data( center_y[1]+ry[i]*np.sin(theta), center_y[0]+ry[i]*np.cos(theta))
                center_y = [center_y[0]+ry[i]*np.cos(py[i]+wy[i]*dt), center_y[1] + ry[i]*np.sin(py[i]+wy[i]*dt)]
                dots_y[i].set_data([center_y[1],center_y[0]])
                
                line_x.set_data([center_x[0], center_x[0]] , [0, center_y[0]])  ### (center_x[0], 0) to (center_x[0], center_y[0])
                line_y.set_data([0, center_x[0]] , [center_y[0], center_y[0]])  ### (0, center_y[0]) to (center_x[0], center_y[0])


            dot.set_data(center_x[0], center_y[0])
            xx.append(center_x[0])
            yy.append(center_y[0])
            outline.set_data(xx, yy)

        anim = animation.FuncAnimation(fig, Anim, frames=100, interval=n_frames, repeat=False)
        return anim



def sparse_outline(Xs, Ys):
    count = 0
    sparse_x = []
    sparse_y = []
    for X in Xs:
        if count % 5 == 0 : # we do not need too much points
            sparse_x.append(X)
        count += 1    
    count = 0
    for Y in Ys:
        if count % 5 == 0 : # we do not need too much points
            sparse_y.append(Y)
        count += 1
    x = np.array(sparse_x)
    y = np.array(sparse_y)
    return x, y


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fourier Draw Machine")
    parser.add_argument("-n", type=int, default=25, help="number of Fourier terms")
    parser.add_argument("-filename", "-f",  type=str, required=True, help="image path")
    parser.add_argument("-method", "-m", type=str, default="DFT", help="DFT/FS")
    parser.add_argument("-save", "-s", action='store_true', help="whether to save the animation as gif")
    parser.add_argument("-savePath", "-sp", default='Gif/result.gif', help="whether to save the animation as gif")
    args = parser.parse_args()

    fig, ax = plt.subplots(1,1)
    plt.axis('equal')
    x_table, y_table = detect_pattern(args.filename)
    x, y = sparse_outline(x_table.tolist(), y_table.tolist())
    anim = draw_epicycles(x, y, args.n, fig, ax, __type__ = args.method)


    
    plt.show()

    if args.save:
        print("Please wait...")
        anim.save('Gif/result.gif', writer='imagemagick') # you need to install imagemagick to save gif
        print(f"The gif has saved as {args.savePath}")
        sys.exit()
    