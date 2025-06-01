import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import queue
from matplotlib.animation import FuncAnimation
import threading
import sys

model_path = './hand_landmarker.task'

mp_hand = mp.solutions.hands

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# グローバル変数
latest_result = None
landmark_queue = queue.Queue(maxsize=10)
camera_running = True

# EMAフィルタのクラス
class EMALandmarkFilter:
    def __init__(self, alpha=0.3, num_landmarks=21):
        self.alpha = alpha
        self.num_landmarks = num_landmarks
        self.smoothed_landmarks = None
        self.is_initialized = False
    
    def update(self, landmarks):
        if landmarks is None or len(landmarks) == 0:
            return None
            
        current_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        if not self.is_initialized:
            self.smoothed_landmarks = current_landmarks.copy()
            self.is_initialized = True
        else:
            self.smoothed_landmarks = (self.alpha * current_landmarks + 
                                     (1 - self.alpha) * self.smoothed_landmarks)
        
        return self.smoothed_landmarks
    
    def reset(self):
        self.smoothed_landmarks = None
        self.is_initialized = False

# 3Dプロット用のクラス
class Hand3DPlotter:
    def __init__(self):
        # 3Dプロットの設定
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # プロットの設定
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z (Depth)')
        self.ax.set_title('Hand Landmarks in 3D Space (Real-time)')
        
        # 軸の範囲設定
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(-0.3, 0.3)
        
        # 手のランドマーク接続情報
        self.connections = list(mp_hand.HAND_CONNECTIONS)
        
        # ランドマークの色設定（指ごとに色分け）
        self.landmark_colors = self._get_landmark_colors()
        
        # 現在のランドマーク
        self.current_landmarks = None
        
        # fps計算用
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
    def _get_landmark_colors(self):
        """各ランドマークの色を定義"""
        colors = []
        # 手首 (0)
        colors.append('red')
        # 親指 (1-4)
        colors.extend(['orange'] * 4)
        # 人差し指 (5-8)
        colors.extend(['yellow'] * 4)
        # 中指 (9-12)
        colors.extend(['green'] * 4)
        # 薬指 (13-16)
        colors.extend(['blue'] * 4)
        # 小指 (17-20)
        colors.extend(['purple'] * 4)
        return colors
    
    def animate(self, frame):
        """アニメーション用の更新関数"""
        # キューから最新データを取得
        try:
            while not landmark_queue.empty():
                self.current_landmarks = landmark_queue.get_nowait()
        except queue.Empty:
            pass
        
        # FPS計算
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # プロットをクリア
        self.ax.clear()
        
        # 軸とタイトルの再設定
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y') 
        self.ax.set_zlabel('Z (Depth)')
        self.ax.set_title(f'Hand Landmarks in 3D Space (3D FPS: {self.fps:.1f})')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(-0.3, 0.3)
        
        if self.current_landmarks is not None:
            # Y軸を反転（画像座標系に合わせる）
            landmarks_3d_flipped = self.current_landmarks.copy()
            landmarks_3d_flipped[:, 1] = 1 - landmarks_3d_flipped[:, 1]
            
            # ランドマークをプロット
            for i, (x, y, z) in enumerate(landmarks_3d_flipped):
                self.ax.scatter(x, y, z, c=self.landmark_colors[i], s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # 接続線を描画
            for connection in self.connections:
                start_idx, end_idx = connection
                start_point = landmarks_3d_flipped[start_idx]
                end_point = landmarks_3d_flipped[end_idx]
                
                self.ax.plot3D([start_point[0], end_point[0]],
                              [start_point[1], end_point[1]],
                              [start_point[2], end_point[2]], 
                              'gray', alpha=0.7, linewidth=2)
            
            # 手首から各指の付け根への線を強調
            wrist_point = landmarks_3d_flipped[0]
            finger_bases = [5, 9, 13, 17]  # 各指の付け根
            for base_idx in finger_bases:
                base_point = landmarks_3d_flipped[base_idx]
                self.ax.plot3D([wrist_point[0], base_point[0]],
                              [wrist_point[1], base_point[1]],
                              [wrist_point[2], base_point[2]], 
                              'red', alpha=0.5, linewidth=3)
        else:
            # データがない場合のメッセージ
            self.ax.text(0.5, 0.5, 0, 'No hand detected', fontsize=16, ha='center', color='red')
        
        # グリッドを表示
        self.ax.grid(True, alpha=0.3)
        
        return []

# EMAフィルタのインスタンス
ema_filter = EMALandmarkFilter(alpha=0.3)

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result
    
    # 3D座標をキューに送信
    if result and result.hand_landmarks:
        hand_landmark = result.hand_landmarks[0]
        smoothed_landmarks = ema_filter.update(hand_landmark)
        if smoothed_landmarks is not None:
            try:
                landmark_queue.put_nowait(smoothed_landmarks)
            except queue.Full:
                try:
                    landmark_queue.get_nowait()
                    landmark_queue.put_nowait(smoothed_landmarks)
                except queue.Empty:
                    pass
    else:
        ema_filter.reset()
        try:
            landmark_queue.put_nowait(None)
        except queue.Full:
            try:
                landmark_queue.get_nowait()
                landmark_queue.put_nowait(None)
            except queue.Empty:
                pass

class CameraProcessor:
    def __init__(self):
        self.cap = None
        self.landmarker = None
        self.is_running = False
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
    def initialize(self):
        """カメラとランドマーカーを初期化"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("エラー: カメラを開くことができません")
                return False
                
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                result_callback=print_result)
            
            self.landmarker = HandLandmarker.create_from_options(options)
            return True
        except Exception as e:
            print(f"初期化エラー: {e}")
            return False
    
    def process_frame(self):
        """フレームを処理"""
        if not self.cap or not self.landmarker:
            return None, False
            
        ret, frame = self.cap.read()
        if not ret:
            return None, False
            
        # フレームのリサイズ
        frame = cv2.resize(frame, (1280, 720))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipeで処理
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp = int(time.time() * 1000)
        self.landmarker.detect_async(mp_image, frame_timestamp)
        
        # ランドマークを描画
        frame = self.draw_landmarks(frame)
        
        # FPS計算
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # 情報を表示
        cv2.putText(frame, f'Camera FPS: {self.fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'EMA Alpha: {ema_filter.alpha:.2f}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, '3D Plot: Active', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame, True
    
    def draw_landmarks(self, frame):
        """ランドマークを描画"""
        if not latest_result or not latest_result.hand_landmarks:
            return frame

        if ema_filter.smoothed_landmarks is None:
            return frame
            
        smoothed_landmarks = ema_filter.smoothed_landmarks
        image_height, image_width, _ = frame.shape
        
        # 接続線を描画
        for connection in mp_hand.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = smoothed_landmarks[start_idx]
            end = smoothed_landmarks[end_idx]
            x0, y0 = int(start[0] * image_width), int(start[1] * image_height)
            x1, y1 = int(end[0] * image_width), int(end[1] * image_height)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # 関節を描画
        for i, landmark in enumerate(smoothed_landmarks):
            x = int(landmark[0] * image_width)
            y = int(landmark[1] * image_height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 5, (0, 0, 0), 1)
        
        return frame
    
    def cleanup(self):
        """リソースを解放"""
        if self.cap:
            self.cap.release()
        if self.landmarker:
            self.landmarker.close()

def main():
    """メイン関数"""
    print("=== 3D Hand Landmark Visualization ===")
    print("操作方法:")
    print("  マウスドラッグ: 3D視点変更")
    print("  キーボード操作は3Dプロットウィンドウで:")
    print("  '+' : EMA alpha値を上げる（応答性向上）")
    print("  '-' : EMA alpha値を下げる（滑らか）")
    print("  'r' : フィルタをリセット")
    print("  'q' または ウィンドウを閉じる: 終了")
    print(f"初期alpha値: {ema_filter.alpha:.2f}")
    
    # カメラプロセッサを初期化
    camera_processor = CameraProcessor()
    if not camera_processor.initialize():
        print("カメラの初期化に失敗しました")
        return
    
    # 3Dプロッターを初期化
    plotter = Hand3DPlotter()
    
    # キーボードイベント処理
    def on_key(event):
        global camera_running
        if event.key == 'q':
            camera_running = False
            plt.close('all')
        elif event.key == '+' or event.key == '=':
            ema_filter.alpha = min(1.0, ema_filter.alpha + 0.05)
            print(f"Alpha値を上げました: {ema_filter.alpha:.2f}")
        elif event.key == '-':
            ema_filter.alpha = max(0.01, ema_filter.alpha - 0.05)
            print(f"Alpha値を下げました: {ema_filter.alpha:.2f}")
        elif event.key == 'r':
            ema_filter.reset()
            print("EMAフィルタをリセットしました")
    
    # カメラ処理関数
    def camera_loop():
        global camera_running
        try:
            while camera_running:
                frame, success = camera_processor.process_frame()
                if not success:
                    break
                    
                # OpenCVウィンドウ表示はメインスレッドでないと問題が起きる場合があるので
                # フレームデータのみ処理して、表示は別の方法を検討
                time.sleep(0.01)  # CPU使用率を下げる
        except Exception as e:
            print(f"カメラループエラー: {e}")
        finally:
            camera_processor.cleanup()
    
    # カメラ処理を別スレッドで開始
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    
    # キーボードイベントを接続
    plotter.fig.canvas.mpl_connect('key_press_event', on_key)
    
    # アニメーションを開始
    ani = FuncAnimation(plotter.fig, plotter.animate, interval=33, blit=False, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        camera_running = False
        camera_processor.cleanup()
        plt.close('all')
        print("終了しました")

if __name__ == "__main__":
    main()