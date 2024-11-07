import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import pytesseract
import re
import json
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import csv

class PowerOCRApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Power Number OCR - Video Enhanced")
        
        # Define parameters
        self.param_configs = {
            # Basic parameter
            'scale_factor': {'type': 'float', 'min': 1.0, 'max': 5.0, 'default': 3.0, 
                           'label': 'Scale Factor'},
                           
            # HSV parameters - Red
            'red_hue_low1': {'type': 'int', 'min': 0, 'max': 179, 'default': 0, 
                           'label': 'Red Hue Low1'},
            'red_hue_high1': {'type': 'int', 'min': 0, 'max': 179, 'default': 10, 
                            'label': 'Red Hue High1'},
            'red_hue_low2': {'type': 'int', 'min': 0, 'max': 179, 'default': 160, 
                           'label': 'Red Hue Low2'},
            'red_hue_high2': {'type': 'int', 'min': 0, 'max': 179, 'default': 179, 
                            'label': 'Red Hue High2'},
            'red_sat_low': {'type': 'int', 'min': 0, 'max': 255, 'default': 100, 
                          'label': 'Red Saturation Low'},
            'red_sat_high': {'type': 'int', 'min': 0, 'max': 255, 'default': 255, 
                           'label': 'Red Saturation High'},
            'red_val_low': {'type': 'int', 'min': 0, 'max': 255, 'default': 100, 
                          'label': 'Red Value Low'},
            'red_val_high': {'type': 'int', 'min': 0, 'max': 255, 'default': 255, 
                           'label': 'Red Value High'},
            
            # HSV parameters - White
            'white_sat_high': {'type': 'int', 'min': 0, 'max': 255, 'default': 83, 
                             'label': 'White Saturation High'},
            'white_val_low': {'type': 'int', 'min': 0, 'max': 255, 'default': 151, 
                            'label': 'White Value Low'},
            
            # Region parameters for red numbers
            'split_ratio': {'type': 'float', 'min': 0.1, 'max': 0.9, 'default': 0.4, 
                          'label': 'Split Ratio'},
            'upper_red_val_offset': {'type': 'int', 'min': -100, 'max': 100, 'default': -20, 
                                   'label': 'Upper Value Offset'},
            'lower_red_sat_offset': {'type': 'int', 'min': -100, 'max': 100, 'default': 25, 
                                   'label': 'Lower Saturation Offset'}
        }
        
        # Initialize parameters
        self.params = {}
        for name, config in self.param_configs.items():
            if config['type'] == 'int':
                self.params[name] = tk.IntVar(value=config['default'])
            elif config['type'] == 'float':
                self.params[name] = tk.DoubleVar(value=config['default'])

        # Add video processing variables
        self.cap = None
        self.video_path = None
        self.is_processing = True
        self.is_paused = False
        self.current_frame = None
        self.roi = None
        self.frame_count = 0
        self.total_frames = 0
        self.time_series_data = []
        self.total_frames = 0
        self.valid_readings = 0
        self.interpolated_readings = 0
        self.recognition_history = []  # 存储识别历史 (time, power, is_valid)
        self.total_frames = 0
        self.valid_readings = 0
        self.interpolated_readings = 0
        self.recognition_history = []  # (time, power, is_valid)
        self.energy_history = []  # (time, instant_power, cumulative_energy)
        self.last_time = None
        self.total_energy = 0  # 总能量消耗(Wh)
        self.last_reset_time = None  # 上次重置时间
        
        self.create_gui()

    def extract_power_value(self, text):
        """
        从OCR文本中提取功率值
        
        Args:
            text (str): OCR识别的文本
            
        Returns:
            float or None: 提取的功率值
        """
        if not text:
            return None
            
        match = re.search(r'(\d+\.?\d*)W?', text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                # 基本范围检查
                if 0 <= value <= 150:
                    return value
            except ValueError:
                pass
        return None

    def prepare_image(self, frame, roi, scale_factor):
        """
        准备图像用于OCR识别
        
        Args:
            frame: 原始帧
            roi: 感兴趣区域
            scale_factor: 缩放因子
            
        Returns:
            处理后的图像
        """
        if roi:
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w].copy()
            return cv2.resize(roi_frame, None, 
                            fx=scale_factor, 
                            fy=scale_factor,
                            interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.resize(frame, None, 
                            fx=scale_factor, 
                            fy=scale_factor,
                            interpolation=cv2.INTER_CUBIC)

    def analyze_number_color(self, hsv, p):
        """
        分析数字颜色（红色或白色），对红色数字进行区域分割处理
        
        Args:
            hsv: HSV颜色空间的图像
            p: 参数字典
            
        Returns:
            tuple: (is_red, binary_mask)
        """
        # 创建基本的红色和白色掩码
        red_mask1 = cv2.inRange(hsv, 
                            np.array([p['red_hue_low1'], p['red_sat_low'], p['red_val_low']]),
                            np.array([p['red_hue_high1'], p['red_sat_high'], p['red_val_high']]))
        red_mask2 = cv2.inRange(hsv,
                            np.array([p['red_hue_low2'], p['red_sat_low'], p['red_val_low']]),
                            np.array([p['red_hue_high2'], p['red_sat_high'], p['red_val_high']]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        # 创建白色掩码
        white_mask = cv2.inRange(hsv,
                                np.array([0, 0, p['white_val_low']]),
                                np.array([180, p['white_sat_high'], 255]))
        
        # 合并掩码进行颜色分析
        possible_digits = cv2.bitwise_or(red_mask, white_mask)
        
        # 分析连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(possible_digits, connectivity=8)
        
        # 统计红色和白色像素
        total_red_pixels = 0
        total_white_pixels = 0
        
        if num_labels > 1:  # 跳过背景
            valid_components = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                if area > 10 and 0.33 < width/height < 3:
                    valid_components.append((i, area))
            
            valid_components = sorted(valid_components, key=lambda x: x[1], reverse=True)[:3]
            for label, _ in valid_components:
                component_mask = (labels == label).astype(np.uint8) * 255
                red_pixels = cv2.countNonZero(cv2.bitwise_and(component_mask, red_mask))
                white_pixels = cv2.countNonZero(cv2.bitwise_and(component_mask, white_mask))
                total_red_pixels += red_pixels
                total_white_pixels += white_pixels
        
        # 判断是否为红色数字
        is_red = False
        if total_red_pixels + total_white_pixels > 0:
            is_red = total_red_pixels / (total_red_pixels + total_white_pixels) > 0.5
        
        # 如果是红色数字，进行区域分割处理
        if is_red:
            height = hsv.shape[0]
            split_y = int(height * p['split_ratio'])
            
            # 分割上下区域
            upper_region = hsv[:split_y, :]
            lower_region = hsv[split_y:, :]
            
            # 处理上部区域
            upper_val_low = p['red_val_low'] + p['upper_red_val_offset']
            upper_val_low = np.clip(upper_val_low, 0, 255)
            
            upper_mask1 = cv2.inRange(upper_region,
                                    np.array([p['red_hue_low1'], p['red_sat_low'], upper_val_low]),
                                    np.array([p['red_hue_high1'], p['red_sat_high'], p['red_val_high']]))
            upper_mask2 = cv2.inRange(upper_region,
                                    np.array([p['red_hue_low2'], p['red_sat_low'], upper_val_low]),
                                    np.array([p['red_hue_high2'], p['red_sat_high'], p['red_val_high']]))
            upper_mask = cv2.bitwise_or(upper_mask1, upper_mask2)
            
            # 处理下部区域
            lower_sat_low = p['red_sat_low'] + p['lower_red_sat_offset']
            lower_sat_low = np.clip(lower_sat_low, 0, 255)
            
            lower_mask1 = cv2.inRange(lower_region,
                                    np.array([p['red_hue_low1'], lower_sat_low, p['red_val_low']]),
                                    np.array([p['red_hue_high1'], p['red_sat_high'], p['red_val_high']]))
            lower_mask2 = cv2.inRange(lower_region,
                                    np.array([p['red_hue_low2'], lower_sat_low, p['red_val_low']]),
                                    np.array([p['red_hue_high2'], p['red_sat_high'], p['red_val_high']]))
            lower_mask = cv2.bitwise_or(lower_mask1, lower_mask2)
            
            # 合并上下区域的掩码
            binary = np.vstack([upper_mask, lower_mask])
            return is_red, binary
        
        # 如果是白色数字，直接返回白色掩码
        return is_red, white_mask

    def calculate_energy(self, current_time, power):
        """
        计算能量消耗
        
        Args:
            current_time (float): 当前时间(秒)
            power (float): 当前功率(W)
        
        Returns:
            tuple: (instant_energy, cumulative_energy) in Wh
        """
        if self.last_time is None:
            self.last_time = current_time
            return 0, self.total_energy
        
        # 计算时间间隔(小时)
        time_delta = (current_time - self.last_time) / 3600  # 转换为小时
        
        # 计算这个时间间隔的能量消耗(Wh)
        instant_energy = power * time_delta
        self.total_energy += instant_energy
        
        # 更新上次时间
        self.last_time = current_time
        
        return instant_energy, self.total_energy

    def reset_energy_counter(self):
        """重置能量计数器"""
        self.last_reset_time = self.last_time
        self.total_energy = 0
        self.energy_history = [(t, p, e) for t, p, e in self.energy_history 
                            if t >= self.last_reset_time] if self.last_reset_time else []

    def process_image(self, *args):
        """处理图像并提取功率值"""
        if self.current_frame is None:
            return
            
        try:
            # 获取参数
            p = {k: v.get() for k, v in self.params.items()}
            
            # 准备图像
            img = self.prepare_image(self.current_frame, self.roi, p['scale_factor'])
            
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # 分析数字颜色和获取掩码
            is_red, binary = self.analyze_number_color(hsv, p)
            
            # OCR识别
            custom_config = (
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.W '
                '--user-patterns \\d+\\.\\d+W '
                '--user-patterns \\d+W '
            )
            text = pytesseract.image_to_string(binary, config=custom_config).strip()
            
            # 提取功率值
            number = self.extract_power_value(text)
            
            if self.cap is not None and not self.is_paused:
                time = self.frame_count / self.cap.get(cv2.CAP_PROP_FPS)
                validated_power = self.validate_and_interpolate_power(time, number, is_red)
                
                if validated_power is not None:
                    # 判断是否为有效识别
                    is_valid = (number is not None and 
                            not is_red and 
                            abs(number - validated_power) < 0.1)
                    
                    # 更新统计信息
                    self.update_statistics(time, validated_power, is_valid)
                    
                    # 计算实时能量消耗（如果不是第一个数据点）
                    if self.recognition_history:
                        prev_time, prev_power, _ = self.recognition_history[-1]
                        time_delta = (time - prev_time) / 3600  # 转换为小时
                        interval_energy = (validated_power + prev_power) / 2 * time_delta
                        self.total_energy += interval_energy
                    
                    # 更新识别历史
                    self.recognition_history.append((time, validated_power, is_valid))
                    
                    # 更新显示
                    success_rate = (self.valid_readings / self.total_frames * 100 
                                if self.total_frames > 0 else 0)
                    
                    display_text = (
                        f"Power: {validated_power:.1f}W | "
                        f"Energy: {self.total_energy:.2f}Wh | "
                        f"Success Rate: {success_rate:.1f}%"
                    )
                    
                    if is_valid:
                        status = "Valid"
                        color = 'green'
                    else:
                        status = "Interpolated"
                        color = 'orange'
                    
                    if is_red:
                        status += " (Red)"
                    
                    self.result_label.configure(
                        text=f"{display_text} - {status}",
                        foreground=color
                    )
            
            # 创建调试显示
            self.update_debug_display(img, binary, is_red)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image: {str(e)}")

    def update_debug_display(self, img, binary, is_red):
        """
        更新调试显示界面
        
        Args:
            img: 原始图像
            binary: 二值化图像
            is_red: 是否为红色数字
        """
        h, w = img.shape[:2]
        display_width = 400
        panel_height = int(h * (display_width / w))
        text_height = 40
        padding = 10
        
        # 创建调试显示图像
        debug_display = np.zeros_like(img)
        if is_red:
            debug_display[binary > 0] = [0, 0, 255]
        else:
            debug_display[binary > 0] = [255, 255, 255]
        
        # 调整显示大小
        display_original = cv2.resize(img, (display_width, panel_height))
        display_debug = cv2.resize(debug_display, (display_width, panel_height))
        binary_display = cv2.resize(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 
                                (display_width, panel_height))
        
        # 创建显示面板
        total_height = (panel_height + text_height + padding) * 3 + text_height
        display = np.ones((total_height, display_width, 3), dtype=np.uint8) * 255
        
        # 添加标题和图像
        titles = ["Original", "Mask Overlay", "Binary"]
        y_pos = text_height + padding
        
        for title, panel in zip(titles, [display_original, display_debug, binary_display]):
            cv2.putText(display, title, (10, y_pos-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            display[y_pos:y_pos+panel_height] = panel
            y_pos += panel_height + text_height + padding
        
        # 更新显示
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_rgb))
        self.image_canvas.config(width=display_width, height=total_height)
        self.image_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def select_roi(self):
        """Select ROI from current frame"""
        if self.current_frame is None:
            messagebox.showerror("Error", "No video loaded")
            return
            
        # Pause video while selecting ROI
        was_paused = self.is_paused
        self.is_paused = True
        
        # Create a window for ROI selection
        cv2.namedWindow("Select ROI")
        roi = cv2.selectROI("Select ROI", self.current_frame, 
                           fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")
        
        if roi[2] > 0 and roi[3] > 0:  # If valid ROI selected
            self.roi = roi
            messagebox.showinfo("Success", "ROI selected successfully")
        
        # Restore previous pause state
        self.is_paused = was_paused
        if not self.is_paused:
            self.process_video_frame()

    def update_statistics(self, time, power, is_valid):
        """
        更新识别统计信息
        
        Args:
            time (float): 时间戳
            power (float): 功率值
            is_valid (bool): 是否为有效识别
        """
        self.total_frames += 1
        if is_valid:
            self.valid_readings += 1
        else:
            self.interpolated_readings += 1
        
        self.recognition_history.append((time, power, is_valid))
        
        # 保持历史记录在合理范围内
        if len(self.recognition_history) > 1000:  # 保留最近1000帧
            self.recognition_history.pop(0)

    def clean_time_series_data(self):
        """Clean the time series data by removing outliers and smoothing"""
        if not self.time_series_data:
            return
        
        # 将数据转换为numpy数组以便处理
        times, values = zip(*self.time_series_data)
        times = np.array(times)
        values = np.array(values)
        
        # 使用移动中位数过滤异常值
        window_size = 5
        for i in range(len(values)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(values), i + window_size // 2 + 1)
            window = values[start_idx:end_idx]
            median = np.median(window)
            
            # 如果当前值与中位数差异太大，使用中位数替代
            if abs(values[i] - median) > 10:  # 10W的阈值
                values[i] = median
        
        # 使用移动平均平滑数据
        window_size = 3
        smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        smoothed_times = times[window_size-1:]
        
        # 更新时间序列数据
        self.time_series_data = list(zip(smoothed_times, smoothed_values))

    def plot_data(self):
        """绘制功率和能量曲线，在功率图中区分识别值和预测值"""
        if not self.recognition_history:
            messagebox.showwarning("Warning", "No data to plot")
            return
            
        # 创建新窗口
        plot_window = tk.Toplevel(self.window)
        plot_window.title("Power and Energy Analysis")
        
        # 创建图形：两个子图
        fig = plt.Figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])
        
        # 功率曲线子图
        ax1 = fig.add_subplot(gs[0])
        # 能量曲线子图
        ax2 = fig.add_subplot(gs[1])
        
        # 准备数据
        times, powers, valids = zip(*self.recognition_history)
        times = np.array(times)
        powers = np.array(powers)

        # 1. 绘制功率曲线
        # 分离正确识别和预测值
        correct_mask = np.array(valids)
        predicted_mask = ~correct_mask

        # 绘制连续的功率曲线作为背景
        ax1.plot(times, powers, 'k-', alpha=0.2, label='Power Trend', linewidth=1)
        
        # 标记正确识别的点
        if np.any(correct_mask):
            ax1.scatter(times[correct_mask], powers[correct_mask], 
                    color='green', label='Correct Reading', 
                    alpha=0.6, s=40, zorder=3)
        
        # 标记预测/插值的点
        if np.any(predicted_mask):
            ax1.scatter(times[predicted_mask], powers[predicted_mask], 
                    color='red', label='Interpolated', 
                    alpha=0.6, s=40, zorder=3)

        # 连接相邻点以显示趋势（用浅色线）
        for i in range(1, len(times)):
            color = 'lightgreen' if valids[i-1] and valids[i] else 'lightcoral'
            ax1.plot(times[i-1:i+1], powers[i-1:i+1], 
                    color=color, alpha=0.3, linewidth=2)

        # 设置功率图样式
        ax1.set_title("Power Over Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Power (W)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # 2. 计算和绘制能量曲线
        energy_values = np.zeros_like(times, dtype=float)
        total_energy = 0
        for i in range(1, len(times)):
            # 使用梯形法则计算能量
            dt = (times[i] - times[i-1]) / 3600  # 转换为小时
            avg_power = (powers[i] + powers[i-1]) / 2
            interval_energy = avg_power * dt
            total_energy += interval_energy
            energy_values[i] = total_energy

        # 绘制能量曲线
        ax2.plot(times, energy_values, 'b-', label='Cumulative Energy', linewidth=2)
        ax2.set_title("Energy Consumption Over Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy (Wh)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        # 添加统计信息
        success_rate = (self.valid_readings / self.total_frames * 100 
                    if self.total_frames > 0 else 0)
        total_time = (times[-1] - times[0]) / 3600  # 转换为小时
        avg_power = np.mean(powers)
        
        stats_text = (
            f"Statistics:\n"
            f"Recognition Rate: {success_rate:.1f}%\n"
            f"Total Time: {total_time:.2f}h\n"
            f"Total Energy: {total_energy:.2f}Wh\n"
            f"Average Power: {avg_power:.1f}W\n"
            f"Valid Readings: {sum(valids)}/{len(valids)}\n\n"
            f"Total Energy Consumption: {total_energy:.2f}Wh"
        )
        
        # 添加文本框，位置在图形右下角
        fig.text(0.98, 0.02, stats_text, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right')
        
        # 在窗口标题中显示总能量
        plot_window.title(f"Power and Energy Analysis - Total Energy: {total_energy:.2f}Wh")
        
        # 设置布局
        fig.tight_layout()
        
        # 创建canvas并添加工具栏
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()

    def save_data(self):
        """Save the collected data to CSV"""
        if not self.time_series_data:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Time (s)", "Power (W)"])
                    writer.writerows(self.time_series_data)
                messagebox.showinfo("Success", "Data saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def reset_video(self):
        """Reset video to beginning"""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.time_series_data = []
            self.process_video_frame()

    def create_gui(self):
        """Create GUI with video support"""
        # Create main frame
        self.main_frame = ttk.Frame(self.window, padding="5")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create left and right frames
        left_frame = ttk.Frame(self.main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        
        right_frame = ttk.Frame(self.main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Create parameter frame
        param_frame = ttk.LabelFrame(left_frame, text="Parameters", padding="5")
        param_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        
        # Create parameter controls
        for row, (param_name, config) in enumerate(self.param_configs.items()):
            self.create_parameter_control(param_frame, param_name, config, row)
        
        # Create button frame
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=1, column=0, pady=5)
        
        ttk.Button(button_frame, text="Open Image", 
                  command=self.open_image).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Open Video",
                  command=self.open_video).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Parameters",
                  command=self.save_params).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Load Parameters",
                  command=self.load_params).grid(row=0, column=3, padx=5)
        
        # Create image display area
        self.image_canvas = tk.Canvas(right_frame, bg='black')
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Create result display label
        self.result_label = ttk.Label(right_frame, text="Waiting for processing...", 
                                    font=('Arial', 12))
        self.result_label.grid(row=1, column=0, pady=5)
        
        # Configure grid weights
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.rowconfigure(0, weight=1)
        left_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)

    def create_parameter_control(self, parent, param_name, config, row):
        """Create control widget for a parameter"""
        label = ttk.Label(parent, text=config['label'])
        label.grid(row=row, column=0, sticky="w", padx=(0,10))
        
        if config['type'] in ['int', 'float']:
            # Create frame for scale and entry
            frame = ttk.Frame(parent)
            frame.grid(row=row, column=1, sticky="ew")
            
            # Scale
            scale = ttk.Scale(frame, 
                            from_=config['min'], 
                            to=config['max'],
                            variable=self.params[param_name],
                            command=self.process_image)
            scale.grid(row=0, column=0, sticky="ew")
            
            # Entry
            entry = ttk.Entry(frame, width=8)
            entry.grid(row=0, column=1, padx=5)
            
            # Synchronization functions
            def sync_scale_to_entry(*args):
                try:
                    value = float(entry.get()) if config['type'] == 'float' else int(entry.get())
                    if config['min'] <= value <= config['max']:
                        self.params[param_name].set(value)
                        self.process_image()
                except ValueError:
                    pass
            
            def sync_entry_to_scale(*args):
                entry.delete(0, tk.END)
                value = self.params[param_name].get()
                entry.insert(0, f"{value:.1f}" if config['type'] == 'float' else str(value))
            
            # Bind events
            entry.bind('<Return>', sync_scale_to_entry)
            entry.bind('<FocusOut>', sync_scale_to_entry)
            self.params[param_name].trace_add('write', lambda *args: sync_entry_to_scale())
            
            # Initial display
            sync_entry_to_scale()
            
            # Configure grid weight
            frame.columnconfigure(0, weight=1)

    def open_image(self):
        """Open and process single image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if file_path:
            try:
                self.current_frame = cv2.imread(file_path)
                if self.current_frame is None:
                    raise ValueError("Cannot read image file")
                self.process_image()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {str(e)}")

    def save_params(self):
        """Save parameters to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            try:
                params = {name: var.get() for name, var in self.params.items()}
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                messagebox.showinfo("Success", "Parameters saved")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save parameters: {str(e)}")

    def load_params(self):
        """Load parameters from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    params = json.load(f)
                for name, value in params.items():
                    if name in self.params:
                        self.params[name].set(value)
                self.process_image()
                messagebox.showinfo("Success", "Parameters loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {str(e)}")

    def reset_video(self):
        """Reset video processing"""
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.time_series_data = []
            ret, self.current_frame = self.cap.read()
            if ret:
                self.process_image()

    def cleanup(self):
        """Cleanup resources"""
        if self.cap is not None:
            self.cap.release()

    def open_video(self):
        """Open video file and optionally start ROI selection"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.avi *.mp4 *.mkv *.mov")]
        )
        if file_path:
            try:
                # Open video
                if self.cap is not None:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise ValueError("Cannot open video file")
                
                self.video_path = file_path
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_count = 0
                self.time_series_data = []
                self.is_paused = True
                
                # Read first frame
                ret, frame = self.cap.read()
                if not ret:
                    raise ValueError("Cannot read video frame")
                
                self.current_frame = frame
                
                # Ask user whether to use ROI
                use_roi = messagebox.askyesno("ROI Selection", 
                    "Would you like to select a Region of Interest (ROI)?\n\n"
                    "Yes - Select a specific area to process\n"
                    "No - Process the entire frame")
                
                if use_roi:
                    # Select ROI
                    cv2.namedWindow("Select ROI - Press SPACE when done")
                    self.roi = cv2.selectROI("Select ROI - Press SPACE when done", 
                                        frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Select ROI - Press SPACE when done")
                    
                    if self.roi[2] <= 0 or self.roi[3] <= 0:
                        messagebox.showerror("Error", "Invalid ROI selected")
                        self.cleanup()
                        return
                else:
                    # Use full frame
                    height, width = frame.shape[:2]
                    self.roi = (0, 0, width, height)
                
                # Create video control panel
                self.create_video_control_panel()
                # Start video processing
                self.is_paused = False
                self.process_video_frame()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open video: {str(e)}")
    
    def select_new_roi(self):
        """Select new ROI while maintaining video position"""
        if self.current_frame is not None:
            was_paused = self.is_paused
            self.is_paused = True
            
            # Ask user whether to use ROI
            use_roi = messagebox.askyesno("ROI Selection", 
                "Would you like to select a Region of Interest (ROI)?\n\n"
                "Yes - Select a specific area to process\n"
                "No - Process the entire frame")
            
            if use_roi:
                cv2.namedWindow("Select ROI - Press SPACE when done")
                self.roi = cv2.selectROI("Select ROI - Press SPACE when done", 
                                        self.current_frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select ROI - Press SPACE when done")
                
                if self.roi[2] <= 0 or self.roi[3] <= 0:
                    messagebox.showwarning("Warning", "Invalid ROI selected, keeping previous ROI")
                    return
            else:
                # Use full frame
                height, width = self.current_frame.shape[:2]
                self.roi = (0, 0, width, height)
            
            self.process_image()
            self.is_paused = was_paused
            if not self.is_paused:
                self.process_video_frame()

    def create_video_control_panel(self):
        """Create enhanced video control panel"""
        if hasattr(self, 'video_control_frame'):
            self.video_control_frame.destroy()
        
        # Create video control frame
        self.video_control_frame = ttk.LabelFrame(self.main_frame, text="Video Controls", padding="5")
        self.video_control_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Create progress bar and frame counter
        progress_frame = ttk.Frame(self.video_control_frame)
        progress_frame.grid(row=0, column=0, columnspan=6, sticky="ew", pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Scale(progress_frame, 
                                    from_=0, 
                                    to=self.total_frames-1,
                                    variable=self.progress_var,
                                    orient="horizontal",
                                    command=self.seek_frame)
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=5)
        
        self.frame_label = ttk.Label(progress_frame, 
                                    text=f"Frame: 0/{self.total_frames}")
        self.frame_label.grid(row=0, column=1, padx=5)
        
        progress_frame.columnconfigure(0, weight=1)
        
        # Create control buttons
        button_frame = ttk.Frame(self.video_control_frame)
        button_frame.grid(row=1, column=0, columnspan=6, pady=5)
        
        ttk.Button(button_frame, text="⏮ Start",
                command=self.goto_start).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="⏪ -100F",
                command=lambda: self.step_frames(-100)).grid(row=0, column=1, padx=5)
        self.play_pause_btn = ttk.Button(button_frame, text="⏸ Pause",
                                        command=self.toggle_pause)
        self.play_pause_btn.grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="⏩ +100F",
                command=lambda: self.step_frames(100)).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="⏭ End",
                command=self.goto_end).grid(row=0, column=4, padx=5)
        
        # Additional controls
        control_frame = ttk.Frame(self.video_control_frame)
        control_frame.grid(row=2, column=0, columnspan=6, pady=5)
        
        ttk.Button(control_frame, text="Select New ROI",
                command=self.select_new_roi).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Plot Data",
                command=self.plot_data).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Save Data",
                command=self.save_data).grid(row=0, column=2, padx=5)

    def seek_frame(self, value):
        """Seek to specific frame"""
        if self.cap is not None:
            frame_number = int(float(value))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, self.current_frame = self.cap.read()
            if ret:
                self.frame_count = frame_number
                self.frame_label.config(text=f"Frame: {frame_number}/{self.total_frames}")
                self.process_image()

    def step_frames(self, frames):
        """Step forward or backward by specified number of frames"""
        if self.cap is not None:
            new_frame = self.frame_count + frames
            new_frame = max(0, min(new_frame, self.total_frames - 1))
            self.progress_var.set(new_frame)
            self.seek_frame(new_frame)

    def goto_start(self):
        """Go to start of video"""
        if self.cap is not None:
            self.progress_var.set(0)
            self.seek_frame(0)

    def goto_end(self):
        """Go to end of video"""
        if self.cap is not None:
            self.progress_var.set(self.total_frames - 1)
            self.seek_frame(self.total_frames - 1)

    def toggle_pause(self):
        """Toggle video pause state"""
        self.is_paused = not self.is_paused
        self.play_pause_btn.config(text="⏸ Pause" if not self.is_paused else "▶ Play")
        if not self.is_paused:
            self.process_video_frame()

    def select_new_roi(self):
        """Select new ROI while maintaining video position"""
        if self.current_frame is not None:
            was_paused = self.is_paused
            self.is_paused = True
            
            cv2.namedWindow("Select ROI - Press SPACE when done")
            self.roi = cv2.selectROI("Select ROI - Press SPACE when done", 
                                    self.current_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI - Press SPACE when done")
            
            if self.roi[2] > 0 and self.roi[3] > 0:
                self.process_image()
                self.is_paused = was_paused
                if not self.is_paused:
                    self.process_video_frame()
            else:
                messagebox.showwarning("Warning", "Invalid ROI selected, keeping previous ROI")

    def process_video_frame(self):
        """Process video frames with progress update"""
        if self.cap is None or self.roi is None:
            return
            
        if not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.frame_count += 1
                
                # Update progress bar and frame counter
                self.progress_var.set(self.frame_count)
                self.frame_label.config(text=f"Frame: {self.frame_count}/{self.total_frames}")
                
                # Process frame
                self.process_image()
                
                # Schedule next frame
                self.window.after(10, self.process_video_frame)
            else:
                # Video ended
                self.is_paused = True
                self.play_pause_btn.config(text="▶ Play")
                self.plot_data()

    def validate_and_interpolate_power(self, time, power, is_red):
        """
        验证OCR识别的功率值并在必要时进行插值
        
        Args:
            time (float): 当前时间戳(秒)
            power (float or None): OCR识别出的功率值
            is_red (bool): 数字是否为红色
            
        Returns:
            float or None: 验证/插值后的功率值
        """
        # 处理首个数据点
        if not self.time_series_data:
            if power is not None and 0 <= power <= 150:
                self.last_valid_power = power
                return power
            return None
        
        # 获取最近的有效数据点
        recent_data = self.time_series_data[-5:]  # 取最近5个点
        recent_times, recent_powers = zip(*recent_data)
        recent_powers = np.array(recent_powers)
        
        # 计算基本统计值
        last_power = recent_powers[-1]
        
        # OCR结果基本验证
        is_invalid = (
            power is None or  # 未识别出数字
            power < 0 or power > 150  # 超出物理范围
        )
        
        # 对红色数字的特殊处理
        if is_red and not is_invalid:
            # 红色数字时，只有在与前值相差不大时才相信OCR结果
            # 因为红色数字识别准确率较低
            if abs(power - last_power) > 20:  # 允许20W的差异
                is_invalid = True
        
        if not is_invalid:
            # 有效读数，直接使用
            self.last_valid_power = power
            self.consecutive_invalid_count = 0
            return power
        
        # 处理无效读数
        self.consecutive_invalid_count += 1
        
        # 插值策略：
        # 1. 连续无效次数少时，保持使用上一个有效值
        # 2. 连续无效次数多时，使用局部平均值
        if self.consecutive_invalid_count <= 3:
            # 短期内保持使用最后一个有效值
            return last_power
        else:
            # 长期无效时使用最近值的平均
            return np.mean(recent_powers)
    
def main():
    root = tk.Tk()
    app = PowerOCRApp(root)
    
    # Set window size and position
    window_width = 1200
    window_height = 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    # Add cleanup on window close
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    
    # Run main loop
    root.mainloop()

if __name__ == "__main__":
    main()
