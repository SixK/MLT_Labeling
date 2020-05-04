#!/bin/python
import argparse
import glob
import json
import os
import re

import cv2
import numpy as np

from collections import deque
from copy import copy, deepcopy

# DATA_FOLDER = "/media/D/DataSet/mlt_selected/"
# OUTPUT = "data/dataset/mlt/"
# DATA_FOLDER = "/home/guest/IA/IA/text-detection-ctpn/data/labelising"
# DATA_FOLDER = "/home/guest/IA/IA/text-detection-ctpn/data/dataset/mlt"
# DATA_FOLDER = "/tmp/autolabeled"
# OUTPUT = "/tmp/mlt/"
DATA_FOLDER = "data"
OUTPUT = "res"

img_path="test1.jpg"

prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1
closing_polygon = False

LINE_THICKNESS = 1

WINDOW_NAME    = 'MLT Labeling'

mouse_x, mouse_y = 0, 0
 
points=list()

img_objects = []
class_index=0


DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


# Class to deal with bbox resizing
class dragBBox:
    '''
        LT -- MT -- RT
        |            |
        LM          RM
        |            |
        LB -- MB -- RB
    '''

    # Size of resizing anchors (depends on LINE_THICKNESS)
    sRA = LINE_THICKNESS * 2

    # Object being dragged
    selected_object = None

    # Flag indicating which resizing-anchor is dragged
    anchor_being_dragged = None
    dragged_points = None

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''
    @staticmethod
    def check_point_inside_resizing_anchors(eX, eY, obj):
        _class_name, points = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        x_left, y_top=points[0]
        x_right, y_bottom=points[2]
        
        if pointInRect(eX, eY,
                        x_left - dragBBox.sRA,
                        y_top - dragBBox.sRA,
                        x_right + dragBBox.sRA,
                        y_bottom + dragBBox.sRA):

            anchor_dict = dragBBox.get_anchors_rectangles(x_left, y_top, x_right, y_bottom)
            for anchor_key in anchor_dict:
                rX_left, rY_top, rX_right, rY_bottom = anchor_dict[anchor_key]
                if pointInRect(eX, eY, rX_left, rY_top, rX_right, rY_bottom):
                    dragBBox.anchor_being_dragged = anchor_key
                    dragBBox.save_dragged_points(points)
                    print("inside anchor and dragging", anchor_key)
                    break

    '''
    \brief This method is used to select an object if one presses a resizing anchor
    '''
    @staticmethod
    def handler_left_mouse_down(eX, eY, obj):
        dragBBox.check_point_inside_resizing_anchors(eX, eY, obj)
        if dragBBox.anchor_being_dragged is not None:
            print("dragging:", dragBBox.anchor_being_dragged, obj)
            dragBBox.selected_object = obj

    @staticmethod
    def save_dragged_points(points) :
        dragBBox.dragged_points=copy(points)

    @staticmethod
    def get_anchor_center_pos(anchor) :
        ax1, ay1, ax2, ay2 = anchor
        return (ax1+(ax1-ax2)/2, ay1+(ay1-ay2)/2)

    @staticmethod
    def get_anchors_rectangles(xmin, ymin, xmax, ymax):
        anchor_list = {}

        mid_x = (xmin + xmax) / 2
        mid_y = (ymin + ymax) / 2

        L_ = [xmin - dragBBox.sRA, xmin + dragBBox.sRA]
        M_ = [mid_x - dragBBox.sRA, mid_x + dragBBox.sRA]
        R_ = [xmax - dragBBox.sRA, xmax + dragBBox.sRA]
        _T = [ymin - dragBBox.sRA, ymin + dragBBox.sRA]
        _M = [mid_y - dragBBox.sRA, mid_y + dragBBox.sRA]
        _B = [ymax - dragBBox.sRA, ymax + dragBBox.sRA]

        anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
        anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
        anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
        anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
        anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
        anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
        anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
        anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

        return anchor_list


    @staticmethod
    def handler_mouse_move(eX, eY):
        print("handle_mouse dragging:", dragBBox.selected_object)
        if dragBBox.selected_object is not None:
            print("handle_mouse dragging:", dragBBox.selected_object)
            if dragBBox.anchor_being_dragged in ("MT", "MB", "LM", "RM") :
                points=copy(dragBBox.dragged_points)
                print('old Points:', points)
                ind=dragBBox.selected_object[0]
            else :
                ind, points = dragBBox.selected_object

            x_left, y_top=points[0]
            x_right, y_bottom=points[2]

            if dragBBox.anchor_being_dragged in ("LT", "RT", "RB","LB") :
                points[("LT", "RT", "RB","LB").index(dragBBox.anchor_being_dragged)] = (eX, eY)
            else :
                anchor_list=dragBBox.get_anchors_rectangles(x_left, y_top, x_right, y_bottom)
                center=dragBBox.get_anchor_center_pos(anchor_list[dragBBox.anchor_being_dragged])
                print("anchor: ",center)
                decay=(eX-center[0], eY-center[1])

                if dragBBox.anchor_being_dragged == "LM" :
                    points[0]=(int(points[0][0] + decay[0]), points[0][1])
                    points[3]=(int(points[3][0] + decay[0]), points[3][1])
                    print("new points:", points[0], points[3])
                elif dragBBox.anchor_being_dragged == "RM" :
                    points[1]=(int(points[1][0] + decay[0]), points[1][1])
                    points[2]=(int(points[2][0] + decay[0]), points[2][1])
                    print("new points:", points[1], points[2])
                elif dragBBox.anchor_being_dragged == "MT" :
                    points[0]=(points[0][0], int(points[0][1] + decay[1]))
                    points[1]=(points[1][0], int(points[1][1] + decay[1]))
                    print("new points:", points[0], points[1])
                elif dragBBox.anchor_being_dragged == "MB" :
                    points[2]=(points[2][0], int(points[2][1] + decay[1]))
                    points[3]=(points[3][0], int(points[3][1] + decay[1]))
                    print("new points:", points[2], points[3])

            action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
            print("Change made !!! : ", action)
            dragBBox.selected_object[1] = points

    '''
    \brief This method will reset this class
     '''
    @staticmethod
    def handler_left_mouse_up():
        if dragBBox.selected_object is not None:
            dragBBox.selected_object = None
            dragBBox.anchor_being_dragged = None
            dragBBox.dragged_points = None


def draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color):
    anchor_dict = dragBBox.get_anchors_rectangles(xmin, ymin, xmax, ymax)
    for anchor_key in anchor_dict:
        x1, y1, x2, y2 = anchor_dict[anchor_key]
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return tmp_img

def draw_selected_anchors(img, objects, selected_bbox) :
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox :
            ind, points = obj
            draw_bbox_anchors(img, points[0][0], points[0][1], points[2][0], points[2][1], (0, 255, 0))


def is_mouse_inside_delete_button():
    for idx, obj in enumerate(img_objects):
        if idx == selected_bbox:
            _ind, points = obj
            x1, y1 = points[0]
            x2, y2 = points[2]

            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if pointInRect(mouse_x, mouse_y, x1_c, y1_c, x2_c, y2_c):
                return True
    return False

def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global is_bbox_selected, prev_was_double_click, mouse_x, mouse_y, points, closing_polygon, class_index, selected_bbox

    set_class = True
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        prev_was_double_click = True
        #print('Double click')
        points=list()
        closing_polygon = False

        # if clicked inside a bounding box we set that bbox
        set_selected_bbox(set_class)
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        set_class = False
        set_selected_bbox(set_class)
        if is_bbox_selected and len(points) == 0:
            obj_to_edit = img_objects[selected_bbox]          
            img_objects.remove(obj_to_edit)
            is_bbox_selected = False
        else :
            points=list()
            closing_polygon = False
    elif event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MBUTTONDOWN :
        if prev_was_double_click:
            #print('Finish double click')
            prev_was_double_click = False
        else:
            #print('Normal left click')
            threshold = 5

            # Check if mouse inside on of resizing anchors of the selected bbox
            if is_bbox_selected:
                dragBBox.handler_left_mouse_down(x, y, img_objects[selected_bbox])


            if dragBBox.anchor_being_dragged is None:
                if len(points) == 0:
                    closing_polygon = False
                    if is_bbox_selected:
                        if is_mouse_inside_delete_button():
                            set_selected_bbox(set_class)
                            obj_to_edit = img_objects[selected_bbox]
                            edit_bbox(obj_to_edit, 'delete')
                        is_bbox_selected = False
                    else:
                        # first click (start drawing a bounding box or delete an item)
                        print("Point_1:", x, y)
                        points.append((x, y))
                else:
                    # threshold is minimal size for bounding box to avoid errors
                    if abs(x - points[-1][0]) > threshold or abs(y - points[-1][1]) > threshold:
                        # second click                        
                        points.append((x, y))
                        print("Point",len(points), x, y)
                        
                        if len(points) == 4 :
                            closing_polygon = True
                            img_objects.append([class_index, points])

                            print('poly list :', img_objects)


    elif event == cv2.EVENT_LBUTTONUP:
        if dragBBox.anchor_being_dragged is not None:
            dragBBox.handler_left_mouse_up()

# Check if a point belongs to a rectangle
def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom

def set_selected_bbox(set_class):
    global is_bbox_selected, selected_bbox
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(img_objects):
        ind, points = obj
        x1 = points[0][0] - dragBBox.sRA
        y1 = points[0][1] - dragBBox.sRA
        x2 = points[2][0] + dragBBox.sRA
        y2 = points[2][1] + dragBBox.sRA
        if pointInRect(mouse_x, mouse_y, x1, y1, x2, y2):
            is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                selected_bbox = idx
                print("bbox selected:", idx)

def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height

def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img

def draw_info_bb_selected(tmp_img):
    for idx, obj in enumerate(img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img
    
def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)

def draw_polylines_from_list(img, objects) :
    for o in objects :
        draw_polylines(img, 0, 0, o[1], True, (88, 227, 60))

def draw_polylines(img, x, y, points, closing, color) :
    cv2.polylines(img, np.array([points]), closing, color)
    if len(points) > 0 and closing == False:
        cv2.line(img, points[-1], (x,y) , (100, 227, 100))

def loadPolygonsFromFiles(filename):
    global img_objects
    try :
        img_objects=[]
        with open(filename, 'r') as infile:
            lines=infile.read().splitlines()
        
        for l in lines :
            data=l.split(',')
            points=list()
            points.append((int(data[0]), int(data[1])))
            points.append((int(data[2]), int(data[3])))
            points.append((int(data[4]), int(data[5])))
            points.append((int(data[6]), int(data[7])))
            ind=data[8]

            img_objects.append([ind, points])

        closing_polygon=True
    except FileNotFoundError:
        display_text("Warning - File {} not found".format(filename), 5000)


def savePolygonsToFile(filename) :
    global img_objects
    with open(filename, 'w') as outfile:
        for obj in img_objects :
            ind, points  = obj
            coordlist = [str(item) for t in points for item in t] 
            outfile.write(','.join(coordlist)+','+str(ind)+'\n')
        display_text("gt file saved: {}".format(filename), 5000)

def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)

def resize_image(img):
    img_size = img.shape

    print("imagesize: ", img_size)
        
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return re_im

def getInputImgAndLabelPaths(im_fn) :
    _, fn = os.path.split(im_fn)
    bfn, ext = os.path.splitext(fn)
    #if ext.lower() not in ['.jpg', '.png']:
    #    continue

    gt_path = os.path.join(DATA_FOLDER, "label", 'gt_' + bfn + '.txt')
    img_path = os.path.join(DATA_FOLDER, "image", im_fn)
    return img_path, gt_path
    
def getOutputImgAndLabelPaths(im_fn) :
    _, fn = os.path.split(im_fn)
    bfn, ext = os.path.splitext(fn)
    # if ext.lower() not in ['.jpg', '.png']:
    #    continue

    gt_path = os.path.join(OUTPUT, "label", 'gt_' + bfn + '.txt')
    img_path = os.path.join(OUTPUT, "image", im_fn)
    return img_path, gt_path



# create window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW_NAME, 1000, 700)
cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

class_rgb = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
# class_rgb = np.array(class_rgb)

color = class_rgb[0]

#########################################################
im_fns = os.listdir(os.path.join(DATA_FOLDER, "image"))
im_fns.sort()

if not os.path.exists(os.path.join(OUTPUT, "image")):
    os.makedirs(os.path.join(OUTPUT, "image"))
if not os.path.exists(os.path.join(OUTPUT, "label")):
    os.makedirs(os.path.join(OUTPUT, "label"))
#########################################################

imgQueue = deque(im_fns)

img_path, gt_path = getInputImgAndLabelPaths(imgQueue[0])
img = cv2.imread(img_path)
img = resize_image(img)
loadPolygonsFromFiles(gt_path)

while True:

    tmp_img = img.copy()
    height, width = tmp_img.shape[:2]
    
    draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
    draw_polylines(tmp_img, mouse_x, mouse_y, points, closing_polygon, color)
    draw_polylines_from_list(tmp_img, img_objects)
    
    if selected_bbox >= 0 :
        draw_selected_anchors(tmp_img, img_objects, selected_bbox)
    
    if dragBBox.anchor_being_dragged is not None:
        dragBBox.handler_mouse_move(mouse_x, mouse_y)
    
    if len(points) == 4 :
        points=list()
    cv2.imshow(WINDOW_NAME, tmp_img)
 
    pressed_key = cv2.waitKey(DELAY)

    if pressed_key == ord('q') :
        break
    elif pressed_key == ord('s') :
        img_path, gt_path = getOutputImgAndLabelPaths(imgQueue[0])
        savePolygonsToFile(gt_path)
        cv2.imwrite(img_path, img)
        display_text("Image saved: "+img_path, 5000)
    elif pressed_key == ord('n') :
        display_text("Loading file: "+img_path, 5000)
        imgQueue.rotate(1)
        img_path, gt_path = getInputImgAndLabelPaths(imgQueue[0])

        img = cv2.imread(img_path)
        img = resize_image(img)
        loadPolygonsFromFiles(gt_path)
    elif pressed_key == ord('p') :
        display_text("Loading file: "+img_path, 5000)
        imgQueue.rotate(-1)
        img_path, gt_path = getInputImgAndLabelPaths(imgQueue[0])

        img = cv2.imread(img_path)
        img = resize_image(img)
        loadPolygonsFromFiles(gt_path)
    # help key listener
    elif pressed_key == ord('h'):
        text = ('[s] save image and ground truth text;\n'
                '[q] to quit;\n'
                '[p] or [n] to change Image;\n'
                )
        display_text(text, 5000)


cv2.destroyAllWindows()
