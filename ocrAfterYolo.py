import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

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



def box_in_img_rot90(box, k, img_w, img_h):
    box = copy.deepcopy(box)
    dst_box = []
    for x, y in box:
        assert  0<= x <= img_w and 0<= y <= img_h
    for i in range(k):
        dst_box = []
        # for x, y in box:
        #     dst_box.append([y, img_w - x])
        p1, p2, p3, p4 = box
        p1 = [p1[1], img_w - p1[0]]
        p2 = [p2[1], img_w - p2[0]]
        p3 = [p3[1], img_w - p3[0]]
        p4 = [p4[1], img_w - p4[0]]
        
        dst_box = [p2, p3, p4, p1]

        img_w, img_h = img_h, img_w
        box = dst_box
    
    return np.array(dst_box)


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    # if abs(num_boxes - 2) < 1e-4:
    #     sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    # else:
    #     sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, 0, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes

def get_rotate_crop_image(img, points):
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
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
        is_rot90 = True
    return dst_img, invert_M, is_rot90

def box_perspective(M, box):
    box_T = [[[point[0]],[point[1]],[1]] for point in box]
    box_dst_T = []
    for point_T in box_T:
        dst_point_T = M @ point_T
        box_dst_T.append(dst_point_T)
    box_dst = [[point_T[0][0], point_T[1][0]] for point_T in box_dst_T]
    return np.array(box_dst).round().astype(int)

class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)
        # self.yolov5_detector = torch.hub.load('ultralytics/yolov5',source='local','custom','exp23/weights/best.pt' )
        # self.yolov5_detector = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # local model
        self.yolov5_detector = torch.hub.load('./', 'custom', path='./exp23/weights/best.pt', source='local')  # local repo

        print('TextSystem __init__ test')

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num




    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'csl': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        # 第一次检测
        pred = self.yolov5_detector(img).pred[0].cpu().numpy() # numpy.array, [[x,y,x,y,conf,cls], ...]. 需要 NMS ?
        elapse = time.time() - start
        dt_boxes_1 = np.array([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]] for x1,y1,x2,y2 in pred[..., :4]])
        
        time_dict['det1'] = elapse
        logger.debug("detect step 1: dt_boxes_1 num : {}, elapse : {}".format(
            len(dt_boxes_1), elapse))
        if dt_boxes_1 is None:
            return None, None, None, None, time_dict
        

        dt_boxes_1 = sorted_boxes(dt_boxes_1)
        img_crop_list_1 = []
        img_crop_list_1_rot180 = []
        invertPT_list = []

        for bno in range(len(dt_boxes_1)):
            tmp_box = copy.deepcopy(dt_boxes_1[bno])
            img_crop, invert_M, is_rot90 = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list_1.append(img_crop)
            img_crop_list_1_rot180.append(np.rot90(img_crop.copy(), 2))
            invertPT_list.append([img_crop, invert_M, is_rot90, ori_im.shape[1], ori_im.shape[0]])
        
        # 第二次检测
        dt_boxes_2 = [] # n * m * 4 * 2, n == len(dt_boxes_1)
        img_crop_list_2 = [] # n * m * 4 * 2, n == len(dt_boxes_1)
        dt_boxes_2_rot180 = [] # n * m * 4 * 2, n == len(dt_boxes_1)
        img_crop_list_2_rot180 = [] # n * m * 4 * 2, n == len(dt_boxes_1)
        elapse = 0.
        for i, img in enumerate(img_crop_list_1):
            img_rot180 = img_crop_list_1_rot180[i]
            _dt_boxes, _elapse = self.text_detector(img)
            _dt_boxes_rot180, _elapse_rot180 = self.text_detector(img_rot180)
            elapse += _elapse + _elapse_rot180

            if _dt_boxes is None:
                _dt_boxes = []
            if _dt_boxes_rot180 is None:
                _dt_boxes_rot180 = []

            dt_boxes_2.append(_dt_boxes)
            dt_boxes_2_rot180.append(_dt_boxes_rot180)

            tmp_crop_list = []
            for bno in range(len(_dt_boxes)):
                tmp_box = copy.deepcopy(_dt_boxes[bno])
                img_crop, _, _ = get_rotate_crop_image(img, tmp_box)
                tmp_crop_list.append(img_crop)
                # print('type(img_crop) = ', type(img_crop))
            img_crop_list_2.append(tmp_crop_list)

            tmp_crop_list_rot180 = []
            for bno in range(len(_dt_boxes_rot180)):
                tmp_box = copy.deepcopy(_dt_boxes_rot180[bno])
                img_crop, _, _ = get_rotate_crop_image(img_rot180, tmp_box)
                tmp_crop_list_rot180.append(img_crop)
                # print('type(img_crop) = ', type(img_crop))
            img_crop_list_2_rot180.append(tmp_crop_list_rot180)
        
        assert len(dt_boxes_1) == len(dt_boxes_2) == len(dt_boxes_2_rot180)
        
        time_dict['det2'] = elapse
        logger.debug("detect step 2: dt_boxes_2 num : {}, elapse : {}".format(
            sum([len(i) for i in dt_boxes_2]), elapse))
        
        if dt_boxes_2 is None:
            return dt_boxes_1, invertPT_list, None, None, time_dict

        if self.use_angle_cls and cls:
            img_crop_list_2, angle_list, elapse = self.text_classifier(
                img_crop_list_2)
            time_dict['cls'] = elapse
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list_2), elapse))

        rec_res = []
        elapse = 0
        for i, img_crop_batch in enumerate(img_crop_list_2):
            _rec_res, _elapse = self.text_recognizer(img_crop_batch)
            score_sum = sum([rec_result[1] for rec_result in _rec_res])
            _rec_res_rot180, _elapse_rot180 = self.text_recognizer(img_crop_list_2_rot180[i])
            score_sum_rot180 = sum([rec_result[1] for rec_result in _rec_res_rot180])
            if score_sum < score_sum_rot180:
                _rec_res = _rec_res_rot180
                rot180_box_invert_list = []
                for box in dt_boxes_2_rot180[i]:
                    img_crop_w, img_crop_h = img_crop_list_1_rot180[i].shape[1], img_crop_list_1_rot180[i].shape[0]
                    rot180_box_invert_list.append(box_in_img_rot90(box, 2, img_crop_w, img_crop_h))
                dt_boxes_2[i] = rot180_box_invert_list
            rec_res.append(_rec_res)
            elapse += _elapse + _elapse_rot180
        
        
        time_dict['rec'] = elapse
        logger.debug("rec_res num  : {}, elapse : {}".format(
            sum([len(i) for i in rec_res]), elapse))
        logger.debug("rec_res : {}".format(rec_res))

        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list_2,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for i, (box_batch, rec_result_batch) in enumerate(zip(dt_boxes_2, rec_res)):
            _filter_boxes_batch, _filter_rec_res_batch = [], []
            for box, rec_result in zip(box_batch, rec_result_batch):
                text, score = rec_result
                if score >= self.drop_score:
                    _filter_boxes_batch.append(box)
                    _filter_rec_res_batch.append(rec_result)
            filter_boxes.append(_filter_boxes_batch)
            filter_rec_res.append(_filter_rec_res_batch)
            # print('_filter_boxes_batch =', _filter_boxes_batch)
            # print('_filter_rec_res_batch =', _filter_rec_res_batch)
                
        
        end = time.time()
        time_dict['all'] = end - start
        return dt_boxes_1, invertPT_list, filter_boxes, filter_rec_res, time_dict




class ocrClass:

    def __init__(self) -> None:
        # --det_model_dir="./trainedmodels/det_student_inference/" 
        # --rec_model_dir="./trainedmodels/rec_db_inference/" 
        # --use_angle_cls=false 
        # --det_limit_side_len 736 
        # --det_limit_type min 
        # --rec_char_dict_path ppocr/utils/en_dict.txt 
        # --use_space_char false 
        # --rec_batch_num 8 
        # --det_db_box_thresh 0.5

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

        print('ocrClass __init__ test')

    def draw_boxes(self, frame, dt_boxes_1, invertPT_list, dt_boxes_2, rec_res):
        dt_boxes_1 = np.array(dt_boxes_1).astype(int)
        # dt_boxes_2 = np.array(dt_boxes_2).astype(int)

        # int_boxes = []
        # for idx, box in enumerate(boxes):
        #     int_box = [list(map(int, pt)) for pt in box]
        #     int_boxes.append(int_box)
        # int_boxes = np.array(int_boxes)

        for i, box in enumerate(dt_boxes_1):
                cv2.polylines(frame, [box], isClosed=True, color=(0, 70, 250), thickness=2)

        for i, boxes_batch in enumerate(dt_boxes_2):
            img_crop, invert_M, is_rot90, ori_im_width, ori_im_height = invertPT_list[i]
            txts = rec_res[i]
            for j, box in enumerate(boxes_batch):
                if is_rot90:
                    # box = np.rot90(box, axes=(1,0))
                    box = box_in_img_rot90(box, 3, img_crop.shape[1], img_crop.shape[0])
                box = box_perspective(invert_M, box)
                # print(txts[j],tuple(box[0]),box)
                cv2.polylines(frame, [box], isClosed=True, color=(0, 150, 250), thickness=1)
                cv2.putText(frame, txts[j][0], tuple(box[0]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)


        return frame

    def detect(self,frame):
        dt_boxes_1, invertPT_list, dt_boxes_2, rec_res, time_dict = self.text_sys(frame, cls=False)

        if(len(dt_boxes_1) == 0):
            print('--ocr failed')
            cv2.putText(frame, '--ocr failed', (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
            return frame

        return self.draw_boxes(frame, dt_boxes_1, invertPT_list, dt_boxes_2, rec_res)





if __name__ == "__main__":
    # imgPath= 'D:/work/PaddleOCR-release-2.6/train_data/single_character_data_det-by-yolov5_bunch/eval/'
    # imgPath= 'D:/work/PaddleOCR-release-2.6/train_data/txt_data/train/imgs/'
    imgPath= 'F:/Code/jianhuaOCR/data/oriimg/'
    images = os.listdir(imgPath)
    print('ocrAfterYolo test:')
    model=ocrClass()
    for f in images:
        f_pname = f.split('.')
        file_name=f_pname[0]
        post_name=f_pname[-1]
        if(post_name=='jpg'):
            print(f)
            img=cv2.imread(imgPath+f)

            # res_img=img
            img=cv2.resize(img,(800,800))
            res_img = model.detect(img)

            win_name = 'OCR: ' + f
            # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            # cv2.imshow('ocr', res_img)
            save_name = imgPath+'/detected/' + file_name + '_ocr.' + post_name
            cv2.imshow('ocr', res_img)
            cv2.imwrite(save_name, res_img)
            print('save_name =', save_name)
            cv2.waitKey(10)

