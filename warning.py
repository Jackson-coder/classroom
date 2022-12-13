import numpy as np

action_cls = {0:"sitting",1:"standing",2:"lying",3:"other"}

class warning_v1(object):
    def __init__(self, detect_results, sigma=3) -> None:
        """全员坐满进行初始化

        Args:
            detect_results (np.array): detect results, Nx6
            sigma (np.array): moving scale ratio
        Returns:
            np.array: seat_information, NX9
        """
        center_x = (detect_results[..., 0] + detect_results[..., 2]) / 2
        center_y = (detect_results[..., 1] + detect_results[..., 3]) / 2
        w = abs(detect_results[..., 0] - detect_results[..., 2])
        h = abs(detect_results[..., 1] - detect_results[..., 3])
        state = []
        for detectState in detect_results[..., -1]:# 躺坐为0， 离席为1
            state.append(1 if detectState == 1 else 0)
        # state = np.array([detectState==1 for detectState in detect_results[..., -1]]) 
        
        center_x = np.expand_dims(center_x, axis=1)
        center_y = np.expand_dims(center_y, axis=1)
        w = np.expand_dims(w, axis=1)
        h = np.expand_dims(h, axis=1)
        state = np.expand_dims(state, axis=1)
        
        
        self.seat = []
        for seat in np.concatenate((center_x,center_y,w,h,detect_results[..., :4],state),axis=1):
            if seat[-1]==0:
                self.seat.append(seat)
        
        self.seat = np.array(self.seat)
        print(self.seat)
        print('************************************************************')
        # self.conf = conf
        self.sigma = sigma


    def warningAction(self, detect_result, distance_vector):
        # if detect_result[-2] < self.conf:
        #     return
        detect_center = ((detect_result[0] + detect_result[2]) / 2, (detect_result[1] + detect_result[3]) / 2)
        
        distance = abs(self.seat[:,0]-detect_center[0]) + abs(self.seat[:,1]-detect_center[1])
        min_index = distance.argmin()
        min_value = distance.min()
        # print(detect_result, min_value)
        # print(distance[min_index], min_value)
        # print(min_index, min_value)
        # print((self.seat[min_index][2]+self.seat[min_index][3]))
        # print(min_value, self.seat[min_index][2]+self.seat[min_index][3])
        if min_value < (self.seat[min_index][2]+self.seat[min_index][3]) and min_value < distance_vector[min_index]:
            distance_vector[min_index] = min_value
            # print(distance_vector)
            if detect_result[-1] == 2:
                self.seat[min_index][-1] = 0
                return "lying"
            elif detect_result[-1] == 0:
                self.seat[min_index][-1] = 0
                return "sitting"
            elif detect_result[-1] == 1:
                self.seat[min_index][-1] = 1
                # print("standing")
                return "standing"
        elif detect_result[-1] == 2:
            return "lying"
        else:
            return 

        
    def countAndAction(self, detect_results, person_number=None):
        seat_number = self.seat.shape[0] if person_number==None or self.seat.shape[0]>person_number else person_number
        distance_vector = np.ones((seat_number,))*1000
        person_number = 0
        detect_lying = []
        for detect_result in detect_results:
            state = self.warningAction(detect_result, distance_vector)
            # print(distance_vector)
            if state is not None:
                person_number += 1
            if state == "lying":
                detect_lying.append(detect_result)
        
        print(person_number, seat_number)
        flag = True if person_number < seat_number else False
        
        # 趴桌(人)， 缺席（状态值）, 离席（座位，可以通过最有一个元素查看离席的座位）
        return detect_lying, [flag,person_number], self.seat
    
    
class warning(object):
    def __init__(self) -> None:
        """全员坐满进行初始化
        """
        self.state_dict = {} # id:state
        self.leaving_frame = {}
        self.lying_frame = {}
        self.sitting_frame = {}
        
    def countAndAction(self, detect_results, seat_number=30):
        """判断三种报警事务

        Args:
            detect_results (list): Nx6, (bboxes, id, cls) 
            seat_number (int, optional): Defaults to 30.

        Returns:
            detect_lying, [flag,person_number], self.seat
        """
        detect_lying = []
        detect_leaving = []
        for detect_result in detect_results:
            if action_cls[int(detect_result[5])] == "lying":
                self.leaving_frame[int(detect_result[4])] = 0
                self.lying_frame[int(detect_result[4])] = 1 if int(detect_result[4]) not in self.lying_frame.keys() else self.lying_frame[int(detect_result[4])]+1
                if self.lying_frame[int(detect_result[4])]>10:
                    detect_lying.append(int(detect_result[4]))
                    self.state_dict[int(detect_result[4])] = True
                # self.state_dict[int(detect_result[4])] = self.state_dict[int(detect_result[4])] + 1 if int(detect_result[4]) in self.state_dict else 1
            elif action_cls[int(detect_result[5])] == "sitting":
                self.lying_frame[int(detect_result[4])] = 0
                self.leaving_frame[int(detect_result[4])] = 0
                self.sitting_frame[int(detect_result[4])] = 1 if int(detect_result[4]) not in self.sitting_frame.keys() else self.sitting_frame[int(detect_result[4])]+1
                if self.sitting_frame[int(detect_result[4])]>10:
                    self.state_dict[int(detect_result[4])] = True
                # self.state_dict[int(detect_result[4])] = self.state_dict[int(detect_result[4])] + 1 if int(detect_result[4]) in self.state_dict else 1
            elif action_cls[int(detect_result[5])] == "standing" and (int(detect_result[4]) in self.state_dict.keys()):
                self.lying_frame[int(detect_result[4])] = 0
                self.leaving_frame[int(detect_result[4])] = self.leaving_frame[int(detect_result[4])] + 1 if int(detect_result[4]) in self.leaving_frame else 1
                # self.leaving_frame[int(detect_result[4])] = 0 if int(detect_result[4]) not in self.leaving_frame.keys() else self.leaving_frame[int(detect_result[4])]
                if self.leaving_frame[int(detect_result[4])]<12 and self.leaving_frame[int(detect_result[4])]>4 and self.state_dict[int(detect_result[4])]: #8 0 3
                    detect_leaving.append(int(detect_result[4]))
            
            
            
        person_number = len(detect_results)
        flag = True if person_number < seat_number else False
        
        # 趴桌(人)， 缺席（状态值）, 离席（座位，可以通过最有一个元素查看离席的座位）
        return detect_lying, [flag,person_number], detect_leaving