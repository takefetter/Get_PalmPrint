import cv2, imutils as im, argparse
import numpy as np
import math


def process_image(img):
    CORRECTION_NEEDED = False
    # Define lower and upper bounds of skin areas in YCrCb colour space.
    lower = np.array([0, 139, 60], np.uint8)
    upper = np.array([255, 180, 127], np.uint8)
    # convert img into 300*x large
    r = 300.0 / img.shape[1]
    dim = (300, int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    original = img.copy()

    # Extract skin areas from the image and apply thresholding
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    mask = cv2.inRange(ycrcb, lower, upper)
    skin = cv2.bitwise_and(ycrcb, ycrcb, mask=mask)
    _, black_and_white = cv2.threshold(mask, 127, 255, 0)

    # Find contours from the thresholded image
    _, contours, _ = cv2.findContours(black_and_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the maximum contour. It is usually the hand.
    length = len(contours)
    maxArea = -1
    final_Contour = np.zeros(img.shape, np.uint8)
    # print(final_Contour)
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        largest_contour = contours[ci]

    # print(largest_contour)
    # Draw it on the image, in case you need to see the ellipse.
    cv2.drawContours(final_Contour, [largest_contour], 0, (0, 255, 0), 2)

    # Get the angle of inclination
    ellipse = _, _, angle = cv2.fitEllipse(largest_contour)

    # original = cv2.bitwise_and(original, original, mask=black_and_white)

    # Vertical adjustment correction
    '''
    This variable is used when the result of hand segmentation is upside down. Will change it to 0 or 180 to correct the actual angle.
    The issue arises because the angle is returned only between 0 and 180, rather than 360.
    '''
    vertical_adjustment_correction = 0
    if CORRECTION_NEEDED: vertical_adjustment_correction = 180

    # Rotate the image to get hand upright
    if angle >= 90:
        black_and_white = im.rotate_bound(black_and_white, vertical_adjustment_correction + 180 - angle)
        original = im.rotate_bound(original, vertical_adjustment_correction + 180 - angle)
        final_Contour = im.rotate_bound(original, vertical_adjustment_correction + 180 - angle)
    else:
        black_and_white = im.rotate_bound(black_and_white, vertical_adjustment_correction - angle)
        original = im.rotate_bound(original, vertical_adjustment_correction - angle)
        final_Contour = im.rotate_bound(final_Contour, vertical_adjustment_correction - angle)

    original = cv2.bitwise_and(original, original, mask=black_and_white)
    # cv2.imshow('Extracted Hand', final_Contour)
    #cv2.imshow('Original image', original)

    # 求手掌中心
    # 参考至http://answers.opencv.org/question/180668/how-to-find-the-center-of-one-palm-in-the-picture/
    # 因为已经是黑白的，所以省略这一句
    # cv2.threshold(black_and_white, black_and_white, 200, 255, cv2.THRESH_BINARY)

    distance = cv2.distanceTransform(black_and_white, cv2.DIST_L2, 5, cv2.CV_32F)
    # Calculates the distance to the closest zero pixel for each pixel of the source image.
    maxdist = 0
    # rows,cols = img.shape
    for i in range(distance.shape[0]):
        for j in range(distance.shape[1]):
            # 先扩展一下访问像素的 .at 的用法：
            # cv::mat的成员函数： .at(int y， int x)
            # 可以用来存取图像中对应坐标为（x，y）的元素坐标。
            # 但是在使用它时要注意，在编译期必须要已知图像的数据类型.
            # 这是因为cv::mat可以存放任意数据类型的元素。因此at方法的实现是用模板函数来实现的。
            dist = distance[i][j]
            if maxdist < dist:
                x = j
                y = i
                maxdist = dist
    # 得到手掌中心并画出最大内切圆
    final_img = original.copy()
    cv2.circle(original, (x, y), maxdist, (255, 100, 255), 1, 8, 0)
    half_slide = maxdist * math.cos(math.pi / 4)
    (left, right, top, bottom) = ((x - half_slide), (x + half_slide), (y - half_slide), (y + half_slide))
    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))
    cv2.rectangle(original, p1, p2, (77, 255, 9), 1, 1)
    final_img = final_img[int(top):int(bottom),int(left):int(right)]
    cv2.imshow('palm image', original)
    return final_img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='Supply image to the program')
    args = vars(ap.parse_args())

    CORRECTION_NEEDED, IMAGE_FILE = False, './hand.jpg'

    if args.get('image'): IMAGE_FILE = args.get('image')

    # Read image
    img = cv2.imread(IMAGE_FILE)
    final_image = process_image(img)
    cv2.imshow('output', final_image)
    cv2.imwrite('hand1.jpg',final_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
