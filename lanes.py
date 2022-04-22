import numpy as np
import cv2
import os

def get_region(img, h, w, low_y, left, right, mid_y, mid_x):
    # isolates desired region in image
    region = np.array([[(int(left*w), int(low_y*h)), (int(mid_x*w), int(mid_y*h)), (int(right*w), int(low_y*h))]])
    window = cv2.fillPoly(np.zeros_like(img), region, 255)
    return cv2.bitwise_and(img, window) # only the filled region (window) stays non-zero

def get_points(avg_line, h):
    # get start and end points of average lines
    gradient, intercept = avg_line
    y1 = h
    y2 = int(y1 * (2/5))
    x1 = int((y1 - intercept) / gradient)
    x2 = int((y2 - intercept) / gradient)
    return np.array([x1, y1, x2, y2])

def magnitude(*line):
    # get length of line
    x1, y1, x2, y2 = line
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def get_lane_markings(lines, h):
    left, left_weights = [], []
    # weights so that longer lines contribute more to the average line
    # useful for curved sections, or else algorithm only works for straight roads
    right, right_weights = [], []
    for line in lines:
        # add gradient and y-intercept of lines to left or right list based on gradient
        x1, y1, x2, y2 = line.reshape(4)
        gradient, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if gradient < 0:
            left.append((gradient, intercept))
            left_weights.append(magnitude(x1, y1, x2, y2)**2)
        else:
            right.append((gradient, intercept))
            right_weights.append(magnitude(x1, y1, x2, y2)**2)
    # only try to calculate the line with something detected
    if len(right_weights) == 0:
        right_avg = np.array((1, -100))
    else:
        right_avg = np.average(right, axis=0, weights=right_weights)
    if len(left_weights) == 0:
        left_avg = np.array((-1, -10000))
    else:
        left_avg = np.average(left, axis=0, weights=left_weights)
    left_line = get_points(left_avg, h)
    right_line = get_points(right_avg, h)
    # if lines cross, terminate lines at intersection
    if left_line[2] > right_line[2]: # lines cross
        intersection_x = (left_avg[1]-right_avg[1])/(right_avg[0] - left_avg[0])
        intersection_y = left_avg[1] + left_avg[0] * intersection_x
        left_line[2], right_line[2] = intersection_x, intersection_x
        left_line[3], right_line[3] = intersection_y, intersection_y
    return np.array([left_line, right_line])

def main():
    for filename in os.listdir('lanes'):
        file = os.path.join('lanes', filename)

        curr_img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        h, w = curr_img.shape
        print(h, w)
        cv2.imshow(f"gray{filename}", curr_img)
        cv2.waitKey(0)
        curr_img = cv2.bilateralFilter(curr_img, 10, 15, 15)
        cv2.imshow(f"blur{filename}", curr_img)
        cv2.waitKey(0)
        edges = cv2.Canny(curr_img, 20, 150)
        cv2.imshow(f"edges{filename}", edges)
        cv2.waitKey(0)
        cropped = get_region(edges, h, w, 0.85, 0.1, 0.9, 0.6, 0.5)
        cv2.imshow(f"cropped{filename}", cropped)
        cv2.waitKey(0)
        lines = cv2.HoughLinesP(cropped,
                                theta=np.pi/360, rho=2, threshold=60, maxLineGap=200, minLineLength=45)
        id_lines = get_lane_markings(lines, h)
        orig_img = cv2.imread(file, cv2.IMREAD_COLOR)
        orig_img_mask = cv2.line(orig_img.copy(),
                                 (id_lines[0, 0], id_lines[0, 1]), (id_lines[0, 2], id_lines[0, 3]), (0, 0, 255), 5)
        orig_img_mask = cv2.line(orig_img_mask,
                                 (id_lines[1, 0], id_lines[1, 1]), (id_lines[1, 2], id_lines[1, 3]), (0, 255, 0), 5)
        orig_img = cv2.addWeighted(orig_img, 1, orig_img_mask, 0.7, 1)
#         curr_img = cv2.addWeighted(curr_img, 1, get_lines(curr_img, id_lines), 1, 1)
        cv2.imshow(f"final{filename}", orig_img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()