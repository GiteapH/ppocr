import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
import logging
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.logging import get_logger

import torch

logger = get_logger()


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = len(dt_boxes)
    # if abs(num_boxes - 2) < 1e-4:
    #     sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    # else:
    #     sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][0][1], x[0][0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][0][1] - _boxes[j][0][0][1]) < 10 and \
                    (_boxes[j + 1][0][0][0] < _boxes[j][0][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def get_rotate_crop_image(img, points, need_rotate=False):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    points = np.array(points)
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    invert_M = cv2.getPerspectiveTransform(pts_std, points)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    is_rot90 = False
    if need_rotate and dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
        is_rot90 = True
    return dst_img, invert_M, is_rot90


class TextSystem(object):
    def __init__(self, args, pt=None):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        if pt is None:
            self.yolov5_detector = torch.hub.load('utils/', 'custom', path='./exp21/weights/best.pt',
                                              source='local')  # local repo
        else:
            self.yolov5_detector = torch.hub.load('utils/', 'custom', path=pt,
                                                  source='local')  # local repo
        print('TextSystem __init__ test')

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args

    def __call__(self, image, cls=True):
        start = time.time()
        H, W = img.shape[:2]
        ori_im = image.copy()
        # 第一次检测
        pred = self.yolov5_detector(image).pred[0].cpu().numpy()  # numpy.array, [[x,y,x,y,conf,cls], ...]. 需要 NMS ?
        elapse = time.time() - start
        dt_boxes = []
        idx_array = []
        for item in pred:
            x1, y1, x2, y2 = item[:4]
            # x1左 x2右 y1上 y2下
            w = x2 - x1  # 宽
            h = y2 - y1  # 高
            x = x1 + w / 2  # 中心点
            y = y1 + h / 2
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            cls = round(item[5])
            # 除以各自的防止超过1
            lovo = [cls, x / W, y / H, w / W, h / H]
            dt_boxes.append([box, cls])
            idx_array.append(lovo)
        self.write_file(imgPath, file_name, idx_array)
        # dt_boxes_1 = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in pred[..., :4]])

        logger.debug("detect step 1: dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))

        dt_boxes = sorted_boxes(dt_boxes)
        dt_boxes_crop_list = []

        for bno in range(len(dt_boxes)):
            box, cls = copy.deepcopy(dt_boxes[bno])
            img_crop, _, _ = get_rotate_crop_image(ori_im, box)
            dt_boxes_crop_list.append([img_crop, cls])
        return dt_boxes, dt_boxes_crop_list

    def write_file(self, dirPath, filename, idx_array):
        if not os.path.exists('labels'):
            os.mkdir('labels')
        with open('labels/' + filename + ".txt", 'w') as f:
            for cls, x, y, w, h in idx_array:
                str = '{} {} {} {} {}\n'.format(int(cls), x, y, w, h)
                f.write(str)


class ocrClass:

    def __init__(self) -> None:
        args = utility.parse_args()
        args.det_model_dir = './trainedmodels/det_db_inference_2/'
        args.rec_model_dir = './trainedmodels/rec_db_inference/'
        args.use_angle_cls = False
        args.det_limit_side_len = 736
        args.det_limit_type = 'min'
        args.rec_char_dict_path = 'ppocr/utils/en_dict.txt'
        args.use_space_char = False
        args.rec_batch_num = 8
        args.det_db_box_thresh = 0.5

        self.text_sys = TextSystem(args)
        self.crop_save_dir = imgPath + '/detected'

        print('ocrClass __init__ test')

    def draw_boxes(self, frame, dt_boxes):
        code_plate_color = (0, 0, 255)
        manual_code_color = (0, 255, 0)
        box_color = (0, 0, 0)
        for box, cls in dt_boxes:
            box = np.array(box, dtype=int)
            if cls == 0:
                box_color = code_plate_color
            elif cls == 1:
                box_color = manual_code_color
            else:
                print('class of box error:', cls)
                continue
            cv2.polylines(frame, [box], isClosed=True, color=box_color, thickness=2)
        return frame

    def to_label(self, image):
        dt_boxes, dt_boxes_crop_list = self.text_sys(image, cls=False)

        if len(dt_boxes) == 0:
            print(image,'未发现需要标注的地方')
            return image, 0

        return self.draw_boxes(img, dt_boxes), len(dt_boxes)


if __name__ == "__main__":
    imgPath = 'E:\\yolov5-master\\datasets\\test\\3a_p\\'
    images = os.listdir(imgPath)
    print('ocrAfterYolo test:')
    model = ocrClass()
    classes = ['m', 'd']
    for f in images:
        f_pname = f.split('.')
        file_name = f_pname[0]
        post_name = f_pname[-1]
        if post_name == 'jpg':
            img = cv2.imread(imgPath + f)
            res_img, box_num = model.to_label(img)
            win_name = 'OCR: ' + f
            res_img = cv2.resize(res_img, (800, 800))
            cv2.imshow('ocr', res_img)
            cv2.waitKey(10)
            #         创建classes.txt
            with open('labels/classes.txt', 'w') as f:
                for i in range(len(classes)):
                    f.write(classes[i] + "\n")
