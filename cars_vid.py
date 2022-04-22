import cv2
import numpy as np

def get_region(img, h, w, low_y, high_y):
    # isolates desired region in image
    region = np.array([[(0, int(h*high_y)), (0, int(h*low_y)), (w, int(h*low_y)), (w, int(h*high_y))]])
    window = cv2.fillPoly(np.zeros_like(img), region, 255)
    return cv2.bitwise_and(img, window)

def test(f1, f2):
    cv2.imshow("orig", f1)
    cv2.waitKey(0)

    # differencing to only highlight changes
    gray0 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray0, gray1)
    cv2.imshow("diff", diff)
    cv2.waitKey(0)

    # thresholding
    ret, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow("threshold", thresh)
    cv2.waitKey(0)

    # dilation to fill in gaps
    filled = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=1)
    cv2.imshow("filled", filled)
    cv2.waitKey(0)

    # windowing to only look at cars in a certain section
    windowed = get_region(filled, filled.shape[0], filled.shape[1], 0.5, 0.9)
    cv2.imshow("windowed", windowed)
    cv2.waitKey(0)

    # find contours
    probable_car = []
    contours, hierarchy = cv2.findContours(windowed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour, level in zip(contours, hierarchy[0]):
        if cv2.contourArea(contour) > 1500: # size check
            if level[3] == -1: # only add if it is outermost contour
                probable_car.append(contour)

    # draw bounding boxes
    for car in probable_car:
        rect = cv2.boundingRect(car)
        x, y, w, h = rect
        cv2.rectangle(f2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("detection", f2)
    cv2.waitKey(0)

def generate(f1, f2, height, width):
    # differencing to only highlight changes
    gray0 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray0, gray1)

    # thresholding
    ret, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

    # dilation to fill in gaps
    filled = cv2.dilate(thresh, np.ones((4, 4), np.uint8), iterations=2)

    # windowing to only look at cars in a certain section
    windowed = get_region(filled, height, width, 0.5, 0.9)

    # find contours
    probable_car = []
    contours, hierarchy = cv2.findContours(windowed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour, level in zip(contours, hierarchy[0]):
        if cv2.contourArea(contour) > 1000:  # size check
            if level[3] == -1:  # only add if it is outermost contour
                probable_car.append(contour)

    # draw bounding boxes
    out_frame = f1.copy()
    for car in probable_car:
        rect = cv2.boundingRect(car)
        x, y, w, h = rect
        cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.line(out_frame, (0, int(0.5*height)), (width, int(0.5*height)), (0, 111, 255), 1)
    cv2.line(out_frame, (0, int(0.9*height)), (width, int(0.9*height)), (0, 111, 255), 1)
    return out_frame

def main():
    capture = cv2.VideoCapture('traffic_vids/vid1.mp4')

    # extract frames
    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    h, w, _ = frames[0].shape

    out_frames = []
    for i in range(len(frames)-1):
        out_frames.append(generate(frames[i], frames[i+1], h, w))
        print(i)

    output_vid = cv2.VideoWriter('traffic_vids/vid1_out.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 25, (w, h))
    for frame in out_frames:
        output_vid.write(frame)
    output_vid.release()

if __name__ == "__main__":
    main()