import cv2
import math
import numpy as np


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def weighted_img(img, initial_img, α=0.8, β=1.0, γ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, γ)


def average_lines(
    lines,
    theta_threshold,
    left_theta_exp,
    right_theta_exp,
    x_threshold,
    left_x_exp,
    right_x_exp,
    region_top,
    region_bottom,
):

    left_theta_sum = 0
    right_theta_sum = 0
    left_theta_count = 0
    right_theta_count = 0
    left_x_sum = 0
    right_x_sum = 0
    left_x_count = 0
    right_x_count = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            theta = math.atan2(y1 - y2, x1 - x2)
            bottom_x = line_x(theta, x1, y1, region_bottom)
            length = line_length(x1, y1, x2, y2)
            if (
                (not y1 - y2 == 0)
                and (
                    (
                        math.fabs(bottom_x - left_x_exp) < x_threshold
                        or math.fabs(bottom_x - right_x_exp) < x_threshold
                    )
                )
                and (
                    math.fabs(theta - left_theta_exp) < theta_threshold
                    or math.fabs(theta - right_theta_exp) < theta_threshold
                )
            ):
                if theta > 0:
                    left_theta_sum += theta * length
                    left_theta_count += length
                    left_x_sum += bottom_x * length
                    left_x_count += length
                else:
                    right_theta_sum += theta * length
                    right_theta_count += length
                    right_x_sum += bottom_x * length
                    right_x_count += length
    left_theta = (
        left_theta_exp if left_theta_count == 0 else left_theta_sum / left_theta_count
    )
    right_theta = (
        right_theta_exp
        if right_theta_count == 0
        else right_theta_sum / right_theta_count
    )
    left_x = left_x_exp if left_x_count == 0 else left_x_sum / left_x_count
    right_x = right_x_exp if right_x_count == 0 else right_x_sum / right_x_count

    left_x2 = line_x(left_theta, left_x, region_bottom, region_top)
    right_x2 = line_x(right_theta, right_x, region_bottom, region_top)
    left_line = [int(left_x), region_bottom, int(left_x2), region_top]
    right_line = [int(right_x), region_bottom, int(right_x2), region_top]
    return [[left_line, right_line]]


def line_length(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def line_x(theta, x, y, y1):

    tangent = math.tan(theta)
    x1 = x - (y - y1) / tangent
    return x1


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    return lines


def steering_lines(averaged_lines, width, height):
    steering_line = [
        (int((averaged_lines[0][0][0] + averaged_lines[0][1][0]) / 2), height),
        (
            int((averaged_lines[0][0][0] + averaged_lines[0][1][0]) / 2),
            int(height * 0.7),
        ),
    ]
    x1 = int(steering_line[0][0])
    y1 = int(steering_line[0][1])
    x2 = int(steering_line[1][0])
    y2 = int(steering_line[1][1])
    return x1, y1, x2, y2


def steering_advice(frame, xOfCarDir, x1, y1, x2, y2):
    text = ""
    lenght = line_length(x1, y1, x2, y2)
    if (x1 - xOfCarDir) > 0:
        if lenght > 0 and lenght < 30:
            text = "Straight"
        elif lenght > 30 and lenght < 60:
            text = "Soft Right"
        elif lenght > 60:
            text = "Right"

    else:
        if lenght > 0 and lenght < 30:
            text = "Straight"
        elif lenght > 30 and lenght < 60:
            text = "Soft Left"
        elif lenght > 60:
            text = "Left"

    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return frame


def define_dashboard(frame, x1, width, height):
    cv2.line(
        frame,
        (int(width / 2), int(height - 30)),
        (int(width / 2), int(height - 68)),
        (0, 140, 255),
        2,
    )
    cv2.circle(frame, (x1, int(height - 50)), 5, (0, 255, 0), -1)
    cv2.line(
        frame,
        (x1, int(height - 50)),
        (int(width / 2), int(height - 50)),
        (255, 255, 255),
        1,
    )
    return frame
