import base64
import cv2

class BASE64C:
    def __init__(self,format="jpg"):
        self.head=f"data:image/{format};base64,"
        self.format="."+format

    def encode(self,path=None,ndarray=None):
        """

        :param path: absolute paht of img
        :param ndarray: ndarray
        :return: 字符串类型base64
        """
        if path:
            with open(path, 'rb') as f:
                base64_data = base64.b64encode(f.read())
                image_base64 = str(base64_data)[2:-1]  # 转字符串
                return self.head+image_base64
        if ndarray is not None:
            image = cv2.imencode(self.format, ndarray)[1] # return:Boolen,data
            base64_data = base64.b64encode(image)   # 二进制格式
            image_base64 = str(base64_data)[2:-1] # 转字符串

            return self.head+image_base64



    def decode(self,base64):
        pass
