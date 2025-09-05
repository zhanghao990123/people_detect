from ultralytics import YOLO
import cv2
import numpy as np
import time
import paho.mqtt.client as mqtt
import json
import base64

class CustomAreaPeopleCounter:
    def __init__(self, model_path='yolov8n.pt', mqtt_broker='localhost', mqtt_port=1883):
        """
        :param model_path: 模型路径
        :param mqtt_broker: MQTT代理地址
        :param mqtt_port: MQTT代理端口
        """
        self.model = YOLO(model_path)
        self.person_class_id = 0  # COCO数据集中person的类别ID

        # 自定义检测区域（支持两种格式）
        # 格式1: 实际像素坐标 [x1, y1, x2, y2]
        # 格式2: 归一化坐标（0-1）[[x1, y1], [x2, y2], ...]
        self.detection_area = None  # 初始化为None，后续通过方法设置

        # MQTT客户端
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.connect(mqtt_broker, mqtt_port)
        self.mqtt_topic = "alert"

        # 记录上一帧的人员数量
        self.prev_people_count = -1

    def set_detection_area_pixels(self, pixel_coords):
        """设置检测区域（实际像素坐标）"""
        self.detection_area = {
            'type': 'pixel',
            'coords': np.array(pixel_coords, dtype=np.int32)
        }

    def set_detection_area_normalized(self, normalized_coords):
        """设置检测区域（归一化坐标，0-1）"""
        self.detection_area = {
            'type': 'normalized',
            'coords': np.array(normalized_coords, dtype=np.float32)
        }

    def is_in_detection_area(self, bbox, frame_shape):
        """检查目标是否在检测区域内"""
        if self.detection_area is None:
            return True  # 如果未设置区域，则检测整个画面

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 转换为实际像素坐标
        if self.detection_area['type'] == 'normalized':
            h, w = frame_shape[:2]
            area_coords = (self.detection_area['coords'] * np.array([w, h])).astype(np.int32)
        else:
            area_coords = self.detection_area['coords']

        # 检查中心点是否在多边形区域内
        return cv2.pointPolygonTest(area_coords, (center_x, center_y), False) >= 0

    def process_frame(self, frame):
        """处理单帧图像"""
        # 执行检测
        results = self.model(frame)
        # results = self.model(frame,imgsz=320)  #imgsz=320

        # 筛选在检测区域内的人员
        people_in_area = []
        for result in results:
            for box in result.boxes:
                if int(box.cls) == self.person_class_id:
                    bbox = [int(x) for x in box.xyxy[0].tolist()]
                    if self.is_in_detection_area(bbox, frame.shape):
                        people_in_area.append({
                            'bbox': bbox,
                            'confidence': float(box.conf)
                        })

        # 绘制检测区域
        if self.detection_area is not None:
            if self.detection_area['type'] == 'normalized':
                h, w = frame.shape[:2]
                area_coords = (self.detection_area['coords'] * np.array([w, h])).astype(np.int32)
            else:
                area_coords = self.detection_area['coords']

            cv2.polylines(frame, [area_coords], True, (0, 0, 255), 2)

        # 绘制检测结果
        for person in people_in_area:
            x1, y1, x2, y2 = person['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{person["confidence"]:.2f}',
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 显示人数统计
        people_count = len(people_in_area)
        cv2.putText(frame, f'People in area: {people_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 将当前帧图片编码为Base64字符串
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # 只有当人员数量发生变化时才发送MQTT消息
        if people_count != self.prev_people_count:
            timestamp_unix = int(time.time())
            message = json.dumps({
                "signTime": timestamp_unix,
                "video_count": people_count,
                "type": 89,
                "typeName": "区域人员计数",
                "rtspUrl": "192.168.1.5",
                "signBigAvatar": frame_base64  # 新增字段，存储Base64编码的图片
            }, ensure_ascii=False)

            # 直接发送，无需再次json.dumps()
            self.mqtt_client.publish(self.mqtt_topic, message)

            # 更新上一帧的人员数量
            self.prev_people_count = people_count

        return frame, people_count

    def process_video(self, video_path, output_path=None):
        """处理视频文件（循环播放检测）"""

        while True:  # 增加外层循环，实现视频循环播放
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error opening video file")
                return

            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置输出视频
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            frame_skip = 5
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # 当前视频播放完毕，跳出内层循环

                if frame_count % frame_skip == 0:
                    processed_frame, count = self.process_frame(frame)
                    cv2.imshow('Custom Area People Counter', processed_frame)
                    if output_path and out:
                        out.write(processed_frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    if out:
                        out.release()
                    cv2.destroyAllWindows()
                    self.mqtt_client.disconnect()
                    return  # 按q退出整个程序

            # 释放当前视频资源
            cap.release()
            if out:
                out.release()
            print("视频播放完毕，重新开始检测...")


# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = CustomAreaPeopleCounter(model_path='./peoplen.pt', mqtt_broker='localhost',
                                       mqtt_port=1883)

    # 设置检测区域（两种方式任选其一）

    # 方式1: 使用实际像素坐标 [x1,y1,x2,y2]
    # detector.set_detection_area_pixels([300, 200, 1000, 800])  # 示例坐标

    # 方式2: 使用归一化坐标 [[x1,y1], [x2,y2], ...] (0-1范围)
    detector.set_detection_area_normalized([   #可以画多边形区域
        [0, 0],  # 左上
        [0, 1],  # 右上
        [1, 1],  # 右下
        [1, 0]  # 左下
    ])

    # 处理视频
    detector.process_video("./test.mp4",
                           'output.mp4')
