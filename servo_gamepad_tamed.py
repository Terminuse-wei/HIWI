#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
servo_gamepad_tamed.py
USB手柄控制连续旋转舵机（GPIO17:水平, GPIO27:垂直）
加入：死区、低通、速度系数、精细模式(LT)、限加速度、在线校中点(X/Y)
"""
import os, time, pygame, pigpio

# -- 在命令行/SSH下禁用窗口 --
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ====== 硬件引脚 ======
PIN_YAW   = 17
PIN_PITCH = 27

# ====== 行为参数（可现场调） ======
NEUTRAL_US       = 1500     # 你的“完全停止”脉宽，可用之前脚本校到比如 1492/1510
BASE_MAX_DELTA   = 220      # 最大速度上限的基线（越小越慢），原来300太快
gain             = 0.40     # 速度系数（0.10~1.00），可用十字键←/→调
DEADZONE         = 0.18     # 摇杆死区（0~1），大一些更稳
LPF_ALPHA        = 0.20     # 输入低通滤波系数（0.1~0.3）
SLEW_US_PER_SEC  = 500      # 限加速度：每秒最大改变的微秒数（越小越丝滑）
REFRESH_HZ       = 30       # 刷新率，30Hz足够

# ====== 按键映射（常见手柄） ======
BTN_A      = 0   # 居中
BTN_X      = 2   # 中点 -1us
BTN_Y      = 3   # 中点 +1us
BTN_LB     = 4   # 反转X（可选，不用也行）
BTN_RB     = 5   # 反转Y（可选，不用也行）
BTN_START  = 7   # 居中
AXIS_LX    = 0   # 左摇杆X
AXIS_LY    = 1   # 左摇杆Y
AXIS_LT    = 2   # 左扳机（有的手柄在 2/5/…，若无就按 BTN_LB 做精细模式）
# 方向键（HAT）
HAT_IDX    = 0

# ====== 工具函数 ======
def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def main():
    global gain, NEUTRAL_US

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("⚠️ 未检测到手柄"); return
    js = pygame.joystick.Joystick(0); js.init()
    name = js.get_name()
    print(f"✅ 已连接手柄: {name}")

    pi = pigpio.pi()
    if not pi.connected:
        print("❌ pigpio 未运行，请先: sudo pigpiod"); return

    # 方向反转开关（如你之前左右反了，设为 -1 即可）
    invert_x = -1    # 左右反向修正：-1 反向，1 正常
    invert_y =  1

    # 低通滤波后的输入 & 当前输出微秒值（用于限加速度）
    x_f, y_f = 0.0, 0.0
    yaw_us   = NEUTRAL_US
    pitch_us = NEUTRAL_US

    clock = pygame.time.Clock()
    print("操作：左摇杆 控制两轴；A/Start 居中；LT 精细模式；十字键←/→ 调速度系数；X/Y 微调中点；Ctrl+C 退出")
    print(f"初始: NEUTRAL={NEUTRAL_US}us, gain={gain:.2f}, BASE_MAX_DELTA={BASE_MAX_DELTA}us, deadzone={DEADZONE}")

    try:
        while True:
            pygame.event.pump()

            # ---- 读轴 ----
            x_raw = js.get_axis(AXIS_LX)
            y_raw = js.get_axis(AXIS_LY)

            # 死区
            x_raw = 0.0 if abs(x_raw) < DEADZONE else x_raw
            y_raw = 0.0 if abs(y_raw) < DEADZONE else y_raw

            # 低通滤波
            x_f = (1.0 - LPF_ALPHA) * x_f + LPF_ALPHA * x_raw
            y_f = (1.0 - LPF_ALPHA) * y_f + LPF_ALPHA * y_raw

            # 精细模式（按住 LT 时减速至 0.4x；若你的LT是轴，值通常在 -1..1）
            fine = False
            try:
                lt_val = js.get_axis(AXIS_LT)
                # 某些手柄LT松开= -1，按下趋近 +1；做个阈值判断
                if lt_val > 0.2:
                    fine = True
            except Exception:
                # 若没有 LT 轴，可用 LB 作为精细模式
                if js.get_button(BTN_LB):
                    fine = True
            gain_eff = gain * (0.40 if fine else 1.0)

            # ---- 计算目标脉宽 ----
            max_delta = int(BASE_MAX_DELTA * gain_eff)
            # 注意：y 轴通常上为负，取反；再应用反向开关
            tgt_yaw   = clamp(int(NEUTRAL_US + (invert_x * x_f) * max_delta),  900, 2100)
            tgt_pitch = clamp(int(NEUTRAL_US + (invert_y * -y_f) * max_delta), 900, 2100)

            # ---- 限加速度（slew rate）----
            max_step = int(SLEW_US_PER_SEC / REFRESH_HZ)
            if tgt_yaw > yaw_us:   yaw_us   = min(yaw_us + max_step, tgt_yaw)
            elif tgt_yaw < yaw_us: yaw_us   = max(yaw_us - max_step, tgt_yaw)
            if tgt_pitch > pitch_us:   pitch_us = min(pitch_us + max_step, tgt_pitch)
            elif tgt_pitch < pitch_us: pitch_us = max(pitch_us - max_step, tgt_pitch)

            # ---- 按键动作 ----
            # 居中
            if js.get_button(BTN_A) or js.get_button(BTN_START):
                yaw_us = pitch_us = NEUTRAL_US

            # 中点微调（每次 1us）
            if js.get_button(BTN_X):  # -1us
                NEUTRAL_US = clamp(NEUTRAL_US - 1, 1200, 1800)
            if js.get_button(BTN_Y):  # +1us
                NEUTRAL_US = clamp(NEUTRAL_US + 1, 1200, 1800)

            # 十字键(HAT)调 gain
            if js.get_numhats() > 0:
                hat = js.get_hat(HAT_IDX)  # (x,y)
                if hat[0] == 1:   # →
                    gain = clamp(gain + 0.02, 0.10, 1.00)
                elif hat[0] == -1: # ←
                    gain = clamp(gain - 0.02, 0.10, 1.00)

            # 可选：LB/RB 切换反向
            if js.get_button(BTN_LB):
                invert_x *= -1; time.sleep(0.15)
            if js.get_button(BTN_RB):
                invert_y *= -1; time.sleep(0.15)

            # ---- 输出 ----
            pi.set_servo_pulsewidth(PIN_YAW,   yaw_us)
            pi.set_servo_pulsewidth(PIN_PITCH, pitch_us)

            print(f"\rYaw={yaw_us}us  Pitch={pitch_us}us | NEU={NEUTRAL_US}  gain={gain:.2f}  fine={'ON' if fine else 'OFF'}   ", end="")
            clock.tick(REFRESH_HZ)

    except KeyboardInterrupt:
        print("\n退出，释放舵机...")
    finally:
        pi.set_servo_pulsewidth(PIN_YAW,   0)
        pi.set_servo_pulsewidth(PIN_PITCH, 0)
        pi.stop()
        pygame.quit()

if __name__ == "__main__":
    main()
