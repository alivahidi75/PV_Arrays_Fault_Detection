# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:52:10 2025

@author: ISD
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# پارامترهای سیستم
M = 10   # Inertia constant
D = 1    # Damping coefficient

# تابع تبدیل سیستم قدرت (فقط بلوک Gp(s) = 1 / (Ms + D))
numerator = [1]
denominator = [M, D]
Gp = ctrl.TransferFunction(numerator, denominator)

# تابع تبدیل ساده کنترل‌کننده (PI controller) - برای آزمایش
Kp = 1.0
Ki = 0.5
Gc = ctrl.TransferFunction([Kp, Ki], [1, 0])  # Kp + Ki/s

# حلقه بسته: Gc(s) * Gp(s) / (1 + Gc(s) * Gp(s))
open_loop = Gc * Gp
closed_loop = ctrl.feedback(open_loop)

# پاسخ سیستم به ورودی پله (تغییر ناگهانی بار)
t = np.linspace(0, 70, 500)
t, y = ctrl.step_response(closed_loop, t)

# رسم پاسخ فرکانس
plt.plot(t, y)
plt.title("LFC پاسخ فرکانس در سیستم تک‌ناحیه‌ای")
plt.xlabel("زمان (ثانیه)")
plt.ylabel("انحراف فرکانس (Hz)")
plt.grid(True)
plt.show()
