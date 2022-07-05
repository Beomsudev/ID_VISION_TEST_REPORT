import cv2
from pylibdmtx.pylibdmtx import decode
import numpy as np
import time

class IdVision:
    def __init__(self):
        pass    # 이미지 경로, 출력 파일 전달 경로 등등.. 추가 필요

    def hook(self):
        start_time = time.time()                                    # START

        img = IdVision.read_image()                                 # 이미지 불러 오기

        pre_img = self.image_preprocessing(img)                     # 이미지 전처리(배경 삭제, 사이즈 조정, 선명 하게, 흑백)

        dmtx = self.read_data_matrix(pre_img)                       # 데이터 매트 릭스 디코딩

        code = self.get_code(dmtx)                                  # 코드값 읽어 오기(LIST 자료)

        coordinate = self.get_coordinate(dmtx)                      # 코드 좌표값 읽어 오기(LIST 자료)

        draw_rect_img = self.draw_rect(coordinate, pre_img, code)   # 코드에 값, 사각형 그리기

        end_time = time.time()
        tact_time = end_time - start_time                           # 디코딩 + 이미지 전처리 시간 출력
        print(f"\n\ntact time {tact_time} : sec")

        self.show_image(draw_rect_img)                              # 이미지 출력 & 저장(result.png)

        print('\n\n***** VISION END *****')                         # E N D

    @staticmethod
    def read_image():
        img = cv2.imread("image/ACL2-7.bmp")

        return img


    def image_preprocessing(self, img):
        # resize, 색 변경 등등등 가장 중요한 부분!
        print('*****PREPROCESSING_START*****')
        img = cv2.resize(img, dsize=(0, 0), fx=0.6, fy=0.6)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 50)
        edges = cv2.dilate(edges, None)
        edges = cv2.erode(edges, None)

        contour_info = []
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # print(len(contours))
        for c in contours:
            contour_info.append((
                c,
                cv2.isContourConvex(c),
                cv2.contourArea(c),
            ))

        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

        mask = np.zeros(edges.shape)
        count_num = 0
        for i in contour_info:
            mask = cv2.fillConvexPoly(mask, i[0], 255)
            count_num = count_num+1


        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)
        mask_stack = np.dstack([mask] * 3)

        mask_stack = mask_stack.astype('float32') / 255.0
        img = img.astype('float32') / 255.0

        MASK_COLOR = (0, 0, 0)
        masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
        masked = (masked * 255).astype('uint8')

        filter_img = cv2.GaussianBlur(masked, (0, 0), 2)
        dst = cv2.addWeighted(masked, 2, filter_img, -1, 0)

        pre_image = dst

        print('\n*****PREPROCESSING_E N D*****\n')
        return pre_image

    def read_data_matrix(self, pre_img):
        print('*****DECODING_START*****')

        dmtx = decode(pre_img, max_count=20) # 옵션 알아 보기! 속도 줄일기 필수!

        print('\n*****DECODING_E N D*****\n')
        return dmtx

    def get_code(self, dmtx):
        print('*****GET_CODE_START*****')

        code_list = list()
        count_num = 0
        for i in dmtx:
            code = dmtx[count_num][0]
            # print('CODE : ', code)
            # print('COUNT :', count_num + 1)

            count_num = count_num + 1
            code_list.append(code)
        print('\n*****GET_CODE_E N D*****\n')

        return code_list



    def get_coordinate(self, dmtx):
        print('*****GET_COORDINATE_START*****')

        count_num = 0
        coordinate = list()
        for i in dmtx:
            temp_coordinate = list()
            start_x = dmtx[count_num][1][0]
            start_y = dmtx[count_num][1][1]
            end_x = start_x + dmtx[count_num][1][2]
            end_y = start_y + dmtx[count_num][1][3]

            temp_coordinate.append(start_x)
            temp_coordinate.append(start_y)
            temp_coordinate.append(end_x)
            temp_coordinate.append(end_y)
            coordinate.append(temp_coordinate)
            count_num = count_num + 1

        print('\n*****GET_COORDINATE_E N D*****\n')

        return coordinate

    def draw_rect(self, coordinate, pre_img, code):
        print('*****DRAW_RECT_START*****')
        blue_color = (255, 0, 0)
        count_num = 0
        for i in coordinate:

            draw_rect_img = cv2.rectangle(pre_img, (coordinate[count_num][0]-100, coordinate[count_num][1]-100),
                                                (coordinate[count_num][2]+100, coordinate[count_num][3]+100),
                                                blue_color, 3)
            count_num = count_num + 1

        font = cv2.FONT_HERSHEY_PLAIN
        blue = (255, 0, 0)
        count_num = 0
        for i in code:
            code_text = str(code[count_num])
            draw_rect_img = cv2.putText(draw_rect_img, code_text,
                                        (coordinate[count_num][0]-50, coordinate[count_num][1]-50),
                                        font, 1, blue, 1, cv2.LINE_AA)
            count_num = count_num + 1

        draw_rect_img = cv2.resize(pre_img, dsize=(0, 0), fx=0.5, fy=0.5)

        print('\n*****DRAW_RECT_E N D*****\n')

        return draw_rect_img

    def show_image(self, draw_rect_img):
        cv2.imshow('result.png', draw_rect_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('result.png', draw_rect_img)


if __name__ == "__main__":
    IdVision().hook()