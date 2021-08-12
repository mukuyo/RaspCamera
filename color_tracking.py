import cv2
import numpy as np
import math

LOW_COLOR = np.array([0, 35, 204])
HIGH_COLOR = np.array([32, 255, 255])

AREA_RATIO_THRESHOLD = 0.005

def find_specific_color(frame,AREA_RATIO_THRESHOLD,LOW_COLOR,HIGH_COLOR):

    # 高さ，幅，チャンネル数
    h,w,c = frame.shape
    # h = 480, w =  640find_specific_color
    # hsv色空間に変換
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    # 色を抽出する
    ex_img = cv2.inRange(hsv,LOW_COLOR,HIGH_COLOR)

    # 輪郭抽出
    contours,hierarchy = cv2.findContours(ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # 面積を計算
    areas = np.array(list(map(cv2.contourArea,contours)))

    if len(areas) == 0 or np.max(areas) / (h*w) < AREA_RATIO_THRESHOLD:
        # 見つからなかったらNoneを返す
        print("the area is too small")
        return None
    else:
        # 面積が最大の塊の重心を計算し返す
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]
        result = cv2.moments(contours[max_idx])
        x = int(result["m10"]/result["m00"])
        y = int(result["m01"]/result["m00"])
        return (x,y)



def main():
    # webカメラを扱うオブジェクトを取得
    cap = cv2.VideoCapture(-1)

    while(cap.isOpened()):
        ret,frame = cap.read()

        if ret is False:
            print("cannot read image")
            continue

        # 位置を抽出
        pos = find_specific_color(
            frame,
            AREA_RATIO_THRESHOLD,
            LOW_COLOR,
            HIGH_COLOR
        )

        if pos is not None:
            # 抽出した座標に丸を描く
            cv2.circle(frame,pos,10,(0,0,255),-1)
            #print(pos[0])
            degree = math.degrees(math.atan2(240 - pos[1], 320 - pos[0])) - 90
            print(degree)
        # 画面に表示する
        cv2.imshow('frame',frame)

        # キーボード入力待ち
        key = cv2.waitKey(1) & 0xFF

        # qが押された場合は終了する
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
