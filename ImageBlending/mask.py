import cv2
import numpy as np 

import argparse

class Preprocess:

    def __init__(self, dst, src, window_name="Preprocessing..."):
        self.dst = dst
        self.src = src
        self.img_show = dst.copy()
        self.mask = np.zeros((dst.shape[0], dst.shape[1]),dtype=np.uint8)
        self.mask_with_img = dst.copy()
        self.draft = True ### draft mask first
        self.loc = [0,0]
        self.zoom = 1
        self.window_H, self.window_W = dst.shape[:2]
        self.zoom_H, self.zoom_W = src.shape[:2]
        self.mouse_pressed = False
        self.mouce_loc = [0,0]
        self.window_name = window_name

    def refresh(self):
        if self.draft:
            cv2.imshow(self.window_name, self.mask_with_img)

        else:
            self.zoom_H, self.zoom_W = int(self.zoom*self.src.shape[0]), int(self.zoom*self.src.shape[1])
            zoom_src = cv2.resize(self.src, (self.zoom_W, self.zoom_H))
            src_show_loc = [0,0]

            if self.loc[0] < 0:
                src_show_loc[0] = -self.loc[0]
            if self.loc[1] < 0:
                src_show_loc[1] = -self.loc[1]

            if self.loc[0] > self.window_H or self.loc[1] > self.window_W:  ### right down corner
                self.img_show = self.dst.copy()
            elif self.loc[0] + self.zoom_H < 0 or self.loc[1] + self.zoom_W < 0:  ### left up corner
                self.img_show = self.dst.copy()
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
                self.img_show = self.dst.copy()

                self.img_show[max(self.loc[0], 0):max(self.loc[0], 0)+src_show_H , max(self.loc[1],0):max(self.loc[1],0)+src_show_W ,:] = zoom_src[src_show_loc[0]:src_show_loc[0]+src_show_H, src_show_loc[1]:src_show_loc[1]+src_show_W,:]

            fg = cv2.bitwise_and(self.img_show,self.img_show, mask = self.mask) ### foreground
            temp = cv2.add(self.mask_with_img, fg)


            cv2.imshow(self.window_name, temp)




def mouseCallback(event, x, y, flags, param):
    preprocess = param
    if preprocess.draft:
        if event == cv2.EVENT_LBUTTONDOWN:
            preprocess.mouse_pressed = True
        elif event == cv2.EVENT_MOUSEMOVE and preprocess.mouse_pressed:
            if x < 0 or y < 0 or y >= preprocess.window_H or x >= preprocess.window_W:
                pass
            else:
                cv2.circle(preprocess.mask, (x,y), 15, (255,255,255), -1)
                cv2.circle(preprocess.mask_with_img, (x,y), 15, (0,0,0), -1)
        elif event == cv2.EVENT_LBUTTONUP:
            preprocess.mouse_pressed = False

    else:
        
        if event == cv2.EVENT_LBUTTONDOWN:
            preprocess.mouse_pressed = True
            preprocess.mouse_loc = [y, x]
        elif event == cv2.EVENT_MOUSEMOVE and preprocess.mouse_pressed:
            preprocess.loc[0] += y - preprocess.mouse_loc[0]
            preprocess.loc[1] += x - preprocess.mouse_loc[1]
            preprocess.mouse_loc = [y, x]
        elif event == cv2.EVENT_LBUTTONUP:
            preprocess.mouse_pressed = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0 :
                if preprocess.zoom < 3:
                    preprocess.zoom += 0.05
            else:
                if preprocess.zoom > 0.3:
                    preprocess.zoom -= 0.05

    preprocess.refresh()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Draft mask and align the source image.")
    parser.add_argument("-src", help="Source Image", required=True)
    parser.add_argument("-dst", help="Destination Image", required=True)
    args = parser.parse_args()

    dst = cv2.imread(args.dst)
    src = cv2.imread(args.src)
    window_name = "Preprocessing..."

    preprocess = Preprocess(dst, src, window_name=window_name)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback("Preprocessing...", mouseCallback, preprocess)


    while(True):
        preprocess.refresh()
        # print(cv2.waitKey())
        if cv2.waitKey() == 27: ### ESC
            cv2.imwrite("mask.jpg", preprocess.mask)
            cv2.imwrite("src.jpg", preprocess.img_show)
            break
        elif cv2.waitKey() == 32: ### space
            if preprocess.draft:
                preprocess.draft = False

        elif cv2.waitKey() == 8 or cv2.waitKey() == 127: ### backspace or del
            preprocess.mask = np.zeros((dst.shape[0], dst.shape[1]),dtype=np.uint8)
            preprocess.mask_with_img = dst.copy()
            preprocess.draft = True
        elif cv2.waitKey() == 18: ### alt
            pass
        elif cv2.waitKey() == 119: ### w
            if preprocess.zoom < 5:
                preprocess.zoom += 0.05
        elif cv2.waitKey() == 115: ### s
            if preprocess.zoom > 0.2:
                preprocess.zoom -= 0.05

    cv2.destroyAllWindows()





