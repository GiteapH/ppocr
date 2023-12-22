# -- coding: utf-8 --
import sys
import threading
import msvcrt
import tkinter.messagebox
import numpy as np
import cv2
import time
import sys, os
import datetime
import inspect
import ctypes
import random
from ctypes import *
import ocrAfterYolo

sys.path.append("./MvImport")
from MvCameraControl_class import *


def Async_raise(tid, exctype):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def Stop_thread(thread):
    Async_raise(thread.ident, SystemExit)


class CameraOperation():

    def __init__(self, obj_cam, st_device_list, n_connect_num=0, b_open_device=False, b_start_grabbing=False,
                 h_thread_handle=None, \
                 b_thread_closed=False, st_frame_info=None, b_exit=False, b_save_bmp=False, b_save_jpg=False,
                 buf_save_image=None, \
                 n_save_image_size=0, n_win_gui_id=0, frame_rate=0, exposure_time=0, gain=0):

        self.obj_cam = obj_cam
        self.st_device_list = st_device_list
        self.n_connect_num = n_connect_num
        self.b_open_device = b_open_device
        self.b_start_grabbing = b_start_grabbing
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain

    def To_hex_str(self, num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2 ** 32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr

    def Open_device(self):
        if False == self.b_open_device:
            # ch:选择设备并创建句柄 | en:Select device and create handle
            nConnectionNum = int(self.n_connect_num)
            stDeviceList = cast(self.st_device_list.pDeviceInfo[int(nConnectionNum)],
                                POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                tkinter.messagebox.showerror('show error', 'create handle fail! ret = ' + self.To_hex_str(ret))
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'open device fail! ret = ' + self.To_hex_str(ret))
                return ret
            print("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                    if ret != 0:
                        print("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret = self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    def Start_grabbing(self, root, panel):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'start grabbing fail! ret = ' + self.To_hex_str(ret))
                return
            self.b_start_grabbing = True
            print("start grabbing successfully!")
            try:
                self.n_win_gui_id = random.randint(1, 10000)
                self.h_thread_handle = threading.Thread(target=CameraOperation.Work_thread, args=(self, root, panel))
                self.h_thread_handle.start()
                self.b_thread_closed = True
            except:
                tkinter.messagebox.showerror('show error', 'error: unable to start thread')
                False == self.b_start_grabbing

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            # 退出线程
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'stop grabbing fail! ret = ' + self.To_hex_str(ret))
                return
            print("stop grabbing successfully!")
            self.b_start_grabbing = False
            self.b_exit = True

    def Close_device(self):
        if True == self.b_open_device:
            # 退出线程
            if True == self.b_thread_closed:
                Stop_thread(self.h_thread_handle)
                self.b_thread_closed = False
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'close deivce fail! ret = ' + self.To_hex_str(ret))
                return

        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit = True
        print("close device successfully!")

    def Set_trigger_mode(self, strMode):
        if True == self.b_open_device:
            if "continuous" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 0)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggermode fail! ret = ' + self.To_hex_str(ret))
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", 1)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggermode fail! ret = ' + self.To_hex_str(ret))
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource", 7)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'set triggersource fail! ret = ' + self.To_hex_str(ret))

    def Trigger_once(self, nCommand):
        if True == self.b_open_device:
            if 1 == nCommand:
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret != 0:
                    tkinter.messagebox.showerror('show error',
                                                 'set triggersoftware fail! ret = ' + self.To_hex_str(ret))

    def Get_parameter(self):
        if True == self.b_open_device:
            stFloatParam_FrameRate = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_FrameRate), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_exposureTime = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_exposureTime), 0, sizeof(MVCC_FLOATVALUE))
            stFloatParam_gain = MVCC_FLOATVALUE()
            memset(byref(stFloatParam_gain), 0, sizeof(MVCC_FLOATVALUE))
            ret = self.obj_cam.MV_CC_GetFloatValue("AcquisitionFrameRate", stFloatParam_FrameRate)
            if ret != 0:
                tkinter.messagebox.showerror('show error',
                                             'get acquistion frame rate fail! ret = ' + self.To_hex_str(ret))
            self.frame_rate = stFloatParam_FrameRate.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam_exposureTime)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'get exposure time fail! ret = ' + self.To_hex_str(ret))
            self.exposure_time = stFloatParam_exposureTime.fCurValue
            ret = self.obj_cam.MV_CC_GetFloatValue("Gain", stFloatParam_gain)
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'get gain fail! ret = ' + self.To_hex_str(ret))
            self.gain = stFloatParam_gain.fCurValue
            tkinter.messagebox.showinfo('show info', 'get parameter success!')

    def Set_parameter(self, frameRate, exposureTime, gain):
        if '' == frameRate or '' == exposureTime or '' == gain:
            tkinter.messagebox.showinfo('show info', 'please type in the text box !')
            return
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime", float(exposureTime))
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'set exposure time fail! ret = ' + self.To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("Gain", float(gain))
            if ret != 0:
                tkinter.messagebox.showerror('show error', 'set gain fail! ret = ' + self.To_hex_str(ret))

            ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frameRate))
            if ret != 0:
                tkinter.messagebox.showerror('show error',
                                             'set acquistion frame rate fail! ret = ' + self.To_hex_str(ret))

            tkinter.messagebox.showinfo('show info', 'set parameter success!')

    def Work_thread(self, root, panel):
        stOutFrame = MV_FRAME_OUT()
        img_buff = None
        buf_cache = None
        numArray = None
        while True:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if 0 == ret:
                if None == buf_cache:
                    buf_cache = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
                # 获取到图像的时间开始节点获取到图像的时间开始节点
                self.st_frame_info = stOutFrame.stFrameInfo
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, self.st_frame_info.nFrameLen)
                print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                self.st_frame_info.nWidth, self.st_frame_info.nHeight, self.st_frame_info.nFrameNum))
                self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
                if img_buff is None:
                    img_buff = (c_ubyte * self.n_save_image_size)()

                if True == self.b_save_jpg:
                    self.Save_jpg(buf_cache)  # ch:保存Jpg图片 | en:Save Jpg
                if True == self.b_save_bmp:
                    self.Save_Bmp(buf_cache)  # ch:保存Bmp图片 | en:Save Bmp
            else:
                print("no data, nret = " + self.To_hex_str(ret))
                continue

            # 转换像素结构体赋值
            # print('imgtype:')
            # print(PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_Mono8, self.st_frame_info.enPixelType)

            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))
            stConvertParam.nWidth = self.st_frame_info.nWidth
            stConvertParam.nHeight = self.st_frame_info.nHeight
            stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = self.st_frame_info.nFrameLen
            stConvertParam.enSrcPixelType = self.st_frame_info.enPixelType

            mode = None  # array转为Image图像的转换模式
            # RGB8直接显示
            if PixelType_Gvsp_RGB8_Packed == self.st_frame_info.enPixelType:
                numArray = CameraOperation.Color_numpy(self, buf_cache, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
                mode = "RGB"

            # Mono8直接显示
            elif PixelType_Gvsp_Mono8 == self.st_frame_info.enPixelType:
                numArray = CameraOperation.Mono_numpy(self, buf_cache, self.st_frame_info.nWidth,
                                                      self.st_frame_info.nHeight)
                mode = "L"

            # 如果是彩色且非RGB则转为RGB后显示
            elif self.Is_color_data(self.st_frame_info.enPixelType):
                # print('color test:', self.st_frame_info.enPixelType)
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3
                stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                time_start = time.time()
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                time_end = time.time()
                # print('MV_CC_ConvertPixelType to RGB s:', time_end - time_start)
                if ret != 0:
                    tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
                    continue
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = CameraOperation.Color_numpy(self, img_buff, self.st_frame_info.nWidth,
                                                       self.st_frame_info.nHeight)
                # print("Is_color_data test:")
                # print(numArray.shape,numArray.size)
                # print(numArray)
                mode = "RGB"

            # 如果是黑白且非Mono8则转为Mono8后显示
            elif self.Is_mono_data(self.st_frame_info.enPixelType):
                nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight
                stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
                stConvertParam.nDstBufferSize = nConvertSize
                time_start = time.time()
                ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                time_end = time.time()
                print('MV_CC_ConvertPixelType to Mono8:', time_end - time_start)
                if ret != 0:
                    # tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
                    continue
                cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
                numArray = CameraOperation.Mono_numpy(self, img_buff, self.st_frame_info.nWidth,
                                                      self.st_frame_info.nHeight)
                mode = "L"

            # 合并OpenCV显示图像

            image = cv2.cvtColor(numArray, cv2.COLOR_RGB2BGR)
            image_resize=cv2.resize(image,(800,800))



            #yolo,paddleocr image
            if(self.st_frame_info.nFrameNum%5==0):
                res_img=model.detect(image_resize)
                cv2.imshow('ocr',res_img)
                cv2.waitKey(10)
                # self.Save_jpg(img_buff)

            cv2.putText(image_resize, str(datetime.datetime.now()), (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(image_resize, str(self.st_frame_info.nFrameNum), (30,50), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 255, 0), 1)
            # cv2.imshow('my',image_resize)
            # cv2.waitKey(10)


            # image_control(numArray,st_frame_info)

            # current_image = Image.frombuffer(mode, (self.st_frame_info.nWidth, self.st_frame_info.nHeight),
            #                                  numArray.astype('uint8')).resize((800, 600), Image.ANTIALIAS)



            # imgtk = ImageTk.PhotoImage(image=current_image, master=root)
            # panel.imgtk = imgtk
            # panel.config(image=imgtk)
            # root.obr = imgtk
            nRet = self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
            if self.b_exit == True:
                if img_buff is not None:
                    del img_buff
                if buf_cache is not None:
                    del buf_cache
                break

    def Save_jpg(self, buf_cache):
        if (None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)

        if return_code != 0:
            # tkinter.messagebox.showerror('show error', 'save jpg fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False

            # tkinter.messagebox.showinfo('show info', 'save jpg success!')
        except:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Save_Bmp(self, buf_cache):
        if (0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp;  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = self.st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = self.st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = self.st_frame_info.nFrameLen
        stParam.pData = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer = cast(byref(self.buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = self.n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)
        if return_code != 0:
            tkinter.messagebox.showerror('show error', 'save bmp fail! ret = ' + self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            tkinter.messagebox.showinfo('show info', 'save bmp success!')
        except:
            self.b_save_bmp = False
            raise Exception("get one frame failed:%s" % e.message)
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Is_mono_data(self, enGvspPixelType):
        if PixelType_Gvsp_Mono8 == enGvspPixelType or PixelType_Gvsp_Mono10 == enGvspPixelType \
                or PixelType_Gvsp_Mono10_Packed == enGvspPixelType or PixelType_Gvsp_Mono12 == enGvspPixelType \
                or PixelType_Gvsp_Mono12_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Is_color_data(self, enGvspPixelType):
        if PixelType_Gvsp_BayerGR8 == enGvspPixelType or PixelType_Gvsp_BayerRG8 == enGvspPixelType \
                or PixelType_Gvsp_BayerGB8 == enGvspPixelType or PixelType_Gvsp_BayerBG8 == enGvspPixelType \
                or PixelType_Gvsp_BayerGR10 == enGvspPixelType or PixelType_Gvsp_BayerRG10 == enGvspPixelType \
                or PixelType_Gvsp_BayerGB10 == enGvspPixelType or PixelType_Gvsp_BayerBG10 == enGvspPixelType \
                or PixelType_Gvsp_BayerGR12 == enGvspPixelType or PixelType_Gvsp_BayerRG12 == enGvspPixelType \
                or PixelType_Gvsp_BayerGB12 == enGvspPixelType or PixelType_Gvsp_BayerBG12 == enGvspPixelType \
                or PixelType_Gvsp_BayerGR10_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG10_Packed == enGvspPixelType \
                or PixelType_Gvsp_BayerGB10_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG10_Packed == enGvspPixelType \
                or PixelType_Gvsp_BayerGR12_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG12_Packed == enGvspPixelType \
                or PixelType_Gvsp_BayerGB12_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG12_Packed == enGvspPixelType \
                or PixelType_Gvsp_YUV422_Packed == enGvspPixelType or PixelType_Gvsp_YUV422_YUYV_Packed == enGvspPixelType:
            return True
        else:
            return False

    def Mono_numpy(self, data, nWidth, nHeight):
        data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
        data_mono_arr = data_.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 1], "uint8")
        numArray[:, :, 0] = data_mono_arr
        return numArray

    def Color_numpy(self, data, nWidth, nHeight):
        # print('Color_numpy-data test:', data)
        # print(nWidth, nWidth)
        # print(int(nWidth * nHeight * 3))
        # count = int(nWidth * nHeight * 3)
        # print(count)

        data_ = np.frombuffer(data, count=int(nWidth * nHeight * 3), dtype=np.uint8, offset=0)
        data_r = data_[0:nWidth * nHeight * 3:3]
        data_g = data_[1:nWidth * nHeight * 3:3]
        data_b = data_[2:nWidth * nHeight * 3:3]

        data_r_arr = data_r.reshape(nHeight, nWidth)
        data_g_arr = data_g.reshape(nHeight, nWidth)
        data_b_arr = data_b.reshape(nHeight, nWidth)
        numArray = np.zeros([nHeight, nWidth, 3], "uint8")

        numArray[:, :, 0] = data_r_arr
        numArray[:, :, 1] = data_g_arr
        numArray[:, :, 2] = data_b_arr
        return numArray



    # # 需要显示的图像数据转换
    # def image_control(self,data, stFrameInfo):
    #     if stFrameInfo.enPixelType == 17301505:
    #         image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    #         print(image.shape)
    #         image_show(self,image=image, name=stFrameInfo.nHeight)
    #     elif stFrameInfo.enPixelType == 17301513:
    #         # print('before:',data.shape,data.size)
    #         data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
    #         # print('after:',data.shape,data.size)
    #         # image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
    #         image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    #         # print(image.shape,image)
    #         image_show(self,image=image, name=stFrameInfo.nHeight)
    #         # image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
    #         # image_show(image=image, name=stFrameInfo.nHeight)
    #     elif stFrameInfo.enPixelType == 35127316:
    #         data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
    #         image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    #         image_show(self,image=image, name=stFrameInfo.nHeight)
    #     elif stFrameInfo.enPixelType == 34603039:
    #         data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
    #         image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
    #         image_show(self,image=image, name=stFrameInfo.nHeight)
    #
    # # 显示图像
    # def image_show(self,image, name):
    #     image = cv2.resize(image, (600, 400), interpolation=cv2.INTER_AREA)
    #     name = str(name)
    #     cv2.imshow(name, image)
    #     # cv2.imwrite("name.bmp", image)
    #     k = cv2.waitKey(1) & 0xff

if __name__ == "__main__":

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)

    # nConnectionNum = input("please input the number of the device to connect:")
    print("please input the number of the device to connect:")
    nConnectionNum=0

    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()
    # cam = MvCamera()


    # ch:选择设备并创建句柄 | en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    stBool = c_bool(False)
    ret = cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
    if ret != 0:
        print("get AcquisitionFrameRateEnable fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:初始化检测模型
    model= ocrAfterYolo.ocrClass()
    try:
        obj_cam_operation = CameraOperation(cam, deviceList, 0)
        hThreadHandle = threading.Thread(target=CameraOperation.Work_thread, args=(obj_cam_operation,None, None))
        hThreadHandle.start()
    except:
        print("error: unable to start thread")

    print("press a key to stop grabbing.")
    msvcrt.getch()

    g_bExit = True
    hThreadHandle.join()

    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        sys.exit()
