import cv2 as cv
import numpy as np

import argparse

import poisson

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Poisson Blending")
    parser.add_argument("-src", help="Source Image", required=True)
    parser.add_argument("-dst", help="Destination Image", required=True)
    parser.add_argument("-mask", help="Mask", required=True)
    parser.add_argument("-m", help="Mixing Gradient Flag", action="store_true")
    args = parser.parse_args()



    src  = cv.imread(args.src)
    dst  = cv.imread(args.dst)
    mask = cv.imread(args.mask)
    mixing = args.m

    index_map, coordinate_map = poisson.map_Omega(src, dst, mask)
    A,b = poisson.construct_DPE(src, dst, mask, index_map, coordinate_map, mixing)
    xr, xg, xb = poisson.solve_DPE(A,b)

    # generate the image
    for i in range(len(xr)):
        y,x = coordinate_map[i]
        dst[y,x,0] = np.clip(xr[i],0,255) # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html
        dst[y,x,1] = np.clip(xg[i],0,255)
        dst[y,x,2] = np.clip(xb[i],0,255)

    cv.imwrite("result.jpg", dst)

    