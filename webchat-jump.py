import os,sys,time,random,cv2
import numpy as np
from sys import exit
from random import choice
class WechatAutoJump(object):
    def __init__(self):
        self.step = 1
    def get_current_img(self):         #从安卓设备拉取游戏截图
        os.system('adb shell screencap -p /sdcard/1.png')
        os.system('adb pull /sdcard/1.png state.png')
        img = cv2.imread('state.png')
        self.GameOver(img)             #检测游戏结束的end界面
        return img    
    def get_player_position(self,img):
        region_upper=int(img.shape[0]*0.3)
        region_lower=int(img.shape[0]*0.7)          
        region=img[region_upper:region_lower]          #原图上下各截取三分之一，缩小查找范围
        hsv_img=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)  #图片颜色转为HSV
        color_lower=np.int32([105,25,45])
        color_upper=np.int32([135,125,130])             #小人颜色区间，copy大神的
        color_mask = cv2.inRange(hsv_img, color_lower, color_upper)
        color_mask = open_op(color_mask)
        color_mask = close_op_large(color_mask)         #通过颜色区间二值化图片
        contours= cv2.findContours(color_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]   #寻找包含小人在内的所有轮廓
        if len(contours)>0:
            max_contour = max(contours, key=cv2.contourArea)      #找出轮廓面积最大的那个
            max_contour = cv2.convexHull(max_contour)             #找出将小人轮廓全包住的凸包轮廓
            rect = cv2.boundingRect(max_contour)                #获取轮廓的最小矩形框（非旋转矩形）
            x,y,w,h = rect
            self.body = (x,y,w,h)
            (a,b)=(int(x+w/2),int(y+h+region_upper - 15))         #小人的底部落点坐标，最下方的位置偏上15
            cv2.circle(img, (a,b), 5, (0,255,0), -1)              #画个绿点表示一下
            return np.array([a, b])
    def get_board_img(self, img, player_pos):           #获取目标方块图片
        self.region_upper=int(img.shape[0]*0.3)
        self.region_lower=int(img.shape[0]*0.6)
        body_position = player_pos                        #压缩查找空间
        if body_position[0]<(img.shape[1]/2.0):             #小人在左边的情况
            self.region_left=body_position[0]+15
            self.region_right=img.shape[1]
            self.po = True
        else:
            self.region_left=0                         #小人在右边的情况
            self.region_right=body_position[0]-15
            self.po = False
        region = img[self.region_upper:self.region_lower, self.region_left:self.region_right]
        return region
    def get_board_position(self, img, player_pos):        #获取目标中心坐标
        region = self.get_board_img(img, player_pos)
        region_gray=cv2.GaussianBlur(region,(3,3),0)       #高斯平滑图片
        region_canny=cv2.Canny(region,6,10)                 #边缘检测二值化图片，6-10比1-10准确
        if self.po:
            region_canny[:,0:self.body[2]]=0                 #去掉小人在左边时，自身所在区间的所有边缘
        else:
            region_canny[:,region_canny.shape[1]-self.body[2]:region_canny.shape[1]]=0   #去掉小人在右边时，自身所在区间的所有边缘
        cv2.imwrite('tyt/img'+str(self.step)+'c.png',region_canny)
        most_up_point_sum=np.int32([0,0])
        most_up_point_sum[1] =np.nonzero([max(i) for i in region_canny])[0][0]           #不为0的坐标中y轴最大的那个点，即最高点纵坐标
        most_up_point_sum[0]= int(np.mean(np.nonzero(region_canny[most_up_point_sum[1]])))  #最高点所有点的横坐标平均值，即最高点横坐标
        most_up_point=most_up_point_sum
        most_up_point[0]+=self.region_left        #最高点实际坐标
        if self.po:
            most_up_point[1]=int(1317-643* most_up_point[0]/1080)           #根据小人方向，选择固定的直线，使用最高点纵坐标获得目标中心点横坐标
        else:
            most_up_point[1]=int(most_up_point[0]*640/1080+648)
        return np.array(most_up_point)      
    def jump(self, player_pos, target_pos):
        self.distance = np.linalg.norm(player_pos - target_pos)
        x = round(self.distance,2)
        press_time = np.e**(-15)*1.969*x**(3)-np.e**(-8)*2.96*x*x+1.813806*x   #最终拟合出来的，按压系数曲线 * 距离，曲线为二次多项式
        cv2.imwrite('tyt/img'+str(self.step)+'a.png',self.img)
        press_time = int(press_time)
        press_h, press_w = random.randint(300,800), random.randint(200,800)    #随机按压位置，防BAN，聊胜于无
        cmd = 'adb shell input swipe {} {} {} {} {}'.format(press_w, press_h, press_w, press_h, press_time)
        print('distance',self.distance,cmd)
        os.system(cmd)
        self.step += 1
        t = round(random.uniform(1,2),2)             #随机按压时间，防BAN，聊胜于无
        time.sleep(t)
       
    def GameOver(self,img):    #匹配结束的图片
        img2 = cv2.imread('1e.png')
        img_e_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_s_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        res2 = cv2.matchTemplate(img_e_gray,img_s_gray,cv2.TM_CCOEFF_NORMED)
        min_val2,max_val2,min_loc2,max_loc2=cv2.minMaxLoc(res2)
        if max_val2 > 0.9:
            input('.....')
    def play(self):
        self.img = self.get_current_img()
        self.player_pos = self.get_player_position(self.img)
        self.target_pos = self.get_board_position(self.img, self.player_pos)
        print(self.player_pos,self.target_pos)
        cv2.circle(self.img, tuple(self.target_pos), 6, (0,0,255), -1)
        self.jump(self.player_pos, self.target_pos)
      
    def run(self):
        try:
            while True:
                self.play()
        except KeyError:
            input('按任意键继续！')
def open_op(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opened_mask

def close_op(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask

def close_op_large(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 12))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask

if __name__ == "__main__":
    AI = WechatAutoJump()
    AI.run()
