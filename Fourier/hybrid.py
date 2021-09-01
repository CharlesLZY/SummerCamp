import cv2
import numpy as np
import scipy.signal
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import argparse

window_name = "test"
cv2.namedWindow(window_name)


# discrete Fourier transform
def DFT(img):
    return fftshift(fft2(img))


# take the maginitude of the result of DFT and convert it to log scale
def scale_spectrum(dft):
    mag = np.abs(dft)
    return np.log10(mag + 1)


# inverse discrete Fourier transform
def IDFT(dft):
    return ifft2(ifftshift(dft))

# generate Gaussian kernel
def Gaussian_kernel(size, sigma):
    gaussian_1d = scipy.signal.gaussian(size, sigma)
    return np.outer(gaussian_1d , gaussian_1d)

# helper display function
def show_DFT(dft):
    plt.imshow(scale_spectrum(dft), cmap='gray')

# low pass filter    
def low_pass(dft, sigma=15):
    size = dft.shape[0]
    gaussian = Gaussian_kernel(size, sigma)
    weight = gaussian
    return dft * weight

# high pass filter 
def high_pass(dft, sigma=15):
    size = dft.shape[0]
    gaussian = Gaussian_kernel(size, sigma)
    weight = (1 - gaussian)
    return dft * weight


def hybrid_image(img1, img2):
    # convert images to frequency domain
    dft1 = DFT(img1)
    dft2 = DFT(img2)
    # get the low frequency component and the high frequency component of two images
    low1 = low_pass(dft1)
    high2 = high_pass(dft2)
    # convert back to time domain
    img1_low = IDFT(low1)
    img2_high = IDFT(high2)
    # mix components by adding them together
    hybrid_img = img1_low + img2_high
    # keep only the real part of the result
    return np.real(hybrid_img)


def padding(img):
    h, w, c = img.shape
    temp = np.zeros((max(h,w), max(h,w), c), dtype=np.uint8) + 255
    if h>= w:
        margin = (h-w)
        temp[:,margin:margin+w,:] = img
    else:
        margin = (w-h)
        temp[margin:margin+h,:,:] = img

    return temp  

class Operation:

    def __init__(self, window_name, img, hybrid):
        self.mouse_pressed = False
        self.mouse_loc_button_down = [0,0]
        self.mouse_loc_button_up = [0,0]
        self.select = False
        self.window_name = window_name
        self.img = img.copy()
        self.img_show = self.img.copy()
        self.hybrid = hybrid

        self.mouse_loc = [0,0]
        self.loc = [0,0]
        self.zoom = 1
        self.window_H, self.window_W = None, None
        self.zoom_H, self.zoom_W = hybrid.shape[:2]

    def refresh(self):
        if not self.select:
            cv2.imshow(self.window_name, self.img_show)
        else:
            self.zoom_H, self.zoom_W = int(self.zoom*self.hybrid.shape[0]), int(self.zoom*self.hybrid.shape[1])
            zoom_src = cv2.resize(self.hybrid, (self.zoom_W, self.zoom_H))
            src_show_loc = [0,0]

            if self.loc[0] < 0:
                src_show_loc[0] = -self.loc[0]
            if self.loc[1] < 0:
                src_show_loc[1] = -self.loc[1]

            if self.loc[0] > self.window_H or self.loc[1] > self.window_W:  ### right down corner
                self.img_show = self.img.copy()
            elif self.loc[0] + self.zoom_H < 0 or self.loc[1] + self.zoom_W < 0:  ### left up corner
                self.img_show = self.img.copy()
            else:
                src_show_H, src_show_W = self.zoom_H, self.zoom_W
                if self.loc[0] < 0:
                    src_show_H = self.zoom_H + self.loc[0]
                if self.loc[1] < 0 :
                    src_show_W = self.zoom_W + self.loc[1]
                if self.zoom_H + self.loc[0] > self.window_H:
                    src_show_H = min(self.window_H - self.loc[0], self.window_H)
                if self.zoom_W + self.loc[1] > self.window_W:
                    src_show_W = min(self.window_W - self.loc[1], self.window_W)
                self.img_show = self.img.copy()

                self.img_show[max(self.loc[0], 0):max(self.loc[0], 0)+src_show_H , max(self.loc[1],0):max(self.loc[1],0)+src_show_W ,:] = zoom_src[src_show_loc[0]:src_show_loc[0]+src_show_H, src_show_loc[1]:src_show_loc[1]+src_show_W,:]


            cv2.imshow(self.window_name, (0.5*self.img_show+0.5*self.img).astype(np.uint8) )


def mouseCallback(event, x, y, flags, param):
    op = param
    if not op.select:
        if event == cv2.EVENT_LBUTTONDOWN  and not op.mouse_pressed:
            op.mouse_pressed = True
            op.mouse_loc_button_down = [y, x]

        elif event == cv2.EVENT_LBUTTONUP and op.mouse_pressed:
            op.mouse_pressed = False
            op.mouse_loc_button_up = [y, x]
            if not op.select:
                lu = op.mouse_loc_button_down ### left up corner
                rd = op.mouse_loc_button_up ### right down corner
                selected_region = op.img.copy()[lu[0]:rd[0], lu[1]:rd[1], :]
                selected_region = padding(selected_region)
                cv2.imshow(op.window_name, selected_region)
                op.img_show = selected_region.copy()
                op.img = selected_region.copy()
                op.window_H, op.window_W = selected_region.shape[0], selected_region.shape[1]
            op.select = True

        elif event == cv2.EVENT_MOUSEMOVE and op.mouse_pressed:
            op.img_show = op.img.copy()
            cv2.rectangle(op.img_show, (op.mouse_loc_button_down[1], op.mouse_loc_button_down[0]), (x,y), (255,255,255), 1)
            
    else:

        if event == cv2.EVENT_LBUTTONDOWN and not op.mouse_pressed:
            op.mouse_pressed = True
            op.mouse_loc = [y, x]


        elif event == cv2.EVENT_LBUTTONUP and op.mouse_pressed:
            op.mouse_pressed = False

        elif event == cv2.EVENT_MOUSEMOVE and op.mouse_pressed:
            op.loc[0] += y - op.mouse_loc[0]
            op.loc[1] += x - op.mouse_loc[1]
            op.mouse_loc = [y, x]

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 :
                if op.zoom < 3:
                    op.zoom += 0.05
            else:
                if op.zoom > 0.3:
                    op.zoom -= 0.05

    op.refresh()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Draft mask and align the source image.")
    parser.add_argument("-src", help="Source Image", required=True)
    parser.add_argument("-dst", help="Destination Image", required=True)
    args = parser.parse_args()

    dst = cv2.imread(args.dst)
    src = cv2.imread(args.src)

    op = Operation(window_name, dst, src)
    cv2.setMouseCallback(window_name, mouseCallback, op)

    while(True):
        op.refresh()
        if cv2.waitKey() == 27: ### ESC
            if op.select:
                # img1 = cv2.cvtColor(op.img, cv2.COLOR_BGR2GRAY)
                # img2 = cv2.cvtColor(op.img_show, cv2.COLOR_BGR2GRAY)

                # hybrid = hybrid_image(img1, img2)
                # cv2.imwrite("result.jpg", hybrid)

                img1 = op.img
                img2 = op.img_show
                b = hybrid_image(img1[:,:,0], img2[:,:,0])
                g = hybrid_image(img1[:,:,1], img2[:,:,1])
                r = hybrid_image(img1[:,:,2], img2[:,:,2])

                cv2.imwrite("result.jpg", cv2.merge([b,g,r]))



            break
            
        elif cv2.waitKey() == 119: ### w
            if preprocess.zoom < 5:
                preprocess.zoom += 0.05
        elif cv2.waitKey() == 115: ### s
            if preprocess.zoom > 0.2:
                preprocess.zoom -= 0.05
    cv2.destroyAllWindows()