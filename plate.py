import numpy as np
import cv2
import imutils
import os

def get_probable_rect(img, contours, max_area = 0.25, min_area = 0.05):
    height, width = img.shape
    area = height * width
    curr_probable = []
    deviation = []
    for contour in contours:
        c_height = (np.abs(contour[0, 0, 1]-contour[1, 0, 1]) + np.abs(contour[2, 0, 1]-contour[3, 0, 1])) / 2
        c_width = (np.abs(contour[0, 0, 0]-contour[2, 0, 0]) + np.abs(contour[1, 0, 0]-contour[3, 0, 0])) / 2
        if (min_area * area) < c_height * c_width < (max_area * area):
            curr_probable.append(contour)
            y_deviation = (np.abs(contour[0, 0, 1]-contour[2, 0, 1]) + np.abs(contour[1, 0, 1]-contour[3, 0, 1])) / 2
            x_deviation = (np.abs(contour[0, 0, 0]-contour[1, 0, 0]) + np.abs(contour[2, 0, 0]-contour[3, 0, 0])) / 2
            deviation.append(y_deviation / c_height + x_deviation / c_width)
    return [c for _, c in sorted(zip(deviation, curr_probable))]


def main():
    for filename in os.listdir("plates"):
        file = os.path.join('plates', filename)

        curr_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # read and convert to grayscale
        cv2.imshow(f"gray{filename}", curr_img)
        cv2.waitKey(0)
        curr_img = cv2.bilateralFilter(curr_img, 15, 19, 19) # blur to remove background noise
        cv2.imshow(f"blur{filename}", curr_img)
        cv2.waitKey(0)
        edges = cv2.Canny(curr_img, 30, 200) # edge recognition
        cv2.imshow(f"edges{filename}", edges)
        cv2.waitKey(0)
        edges = cv2.dilate(edges, None, iterations=2) # close up some slightly open curves
        cv2.imshow(f"dilated{filename}", edges)
        cv2.waitKey(0)
        edges = cv2.erode(edges, None, iterations=1) # thin out edges
        cv2.imshow(f"eroded{filename}", edges)
        cv2.waitKey(0)
        contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = []

        for contour in contours:
            # reduce number of corners
            reduction = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
            if len(reduction) == 4:
                screenCnt.append(reduction)

        if contour is not None:
            # sort contours that "seem" like rectangles, heuristics
            prob_cnt = get_probable_rect(curr_img, screenCnt, max_area=0.9, min_area=0.001)
            if len(prob_cnt) != 0:
                print(len(prob_cnt))
                orig_img = cv2.imread(file, cv2.IMREAD_COLOR)
                for cnt in prob_cnt[:1]:
                    cv2.drawContours(orig_img, [cnt], -1, (0, 255, 0), 2)
                cv2.imshow(f"label{filename}", orig_img)
                cv2.waitKey(0)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()