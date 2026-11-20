"""
Робо-рука SocialTech — параметрическая 3D-модель.

Запуск:  python3 model.py
Результат:
  ../stl/*.stl         — детали для печати (уже повёрнуты для печати)
  ../viewer/arm.bin.gz — геометрия для 3D-просмотрщика

Все размеры в миллиметрах. Размеры сервоприводов приблизительные:
перед печатью измерьте свои моторы штангенциркулем и поправьте SERVO_*.
"""
import gzip
import json
import math
import os
import struct

import numpy as np
from manifold3d import CrossSection as C
from manifold3d import Manifold as M

HERE = os.path.dirname(os.path.abspath(__file__))
STL_DIR = os.path.join(HERE, "..", "stl")
VIEW_DIR = os.path.join(HERE, "..", "viewer")
SEG = 64  # гладкость окружностей

# ---------------------------------------------------------------- параметры
TOL = 0.3            # зазор для посадки
M8 = 8.4             # отверстие под ось-болт М8
M3 = 3.2             # отверстие под винт М3
BALL = 6.0           # страйкбольный шарик 6 мм

BASE_R, BASE_H = 100, 60      # корпус основания
RACE_R = 80                   # радиус дорожки шариков
PLAT_T = 10                   # толщина поворотной платформы
PLAT_Z = BASE_H + 0.6         # низ платформы
SHOULDER_Z = 125              # высота оси плеча
L_UPPER = 130                 # плечо: ось-ось
L_FORE = 100                  # предплечье: ось-ось
UA_X, UA_Y = 35, 40           # сечение плеча
FA_X, FA_Y = 30, 24           # сечение предплечья

# стандартный сервопривод (MG996R / DS3218 / DS3235)
SERVO_STD = dict(L=40.7, W=19.7, H=40.0, ear_span=54.0, ear_t=2.5,
                 ear_from_top=10.0, shaft_off=10.0, hole_dx=49.5, hole_dy=10.0)
# большой сервопривод (DS5160 / DS51150) — для апгрейда плеча
SERVO_BIG = dict(L=65.5, W=30.0, H=48.0, ear_span=82.0, ear_t=3.0,
                 ear_from_top=12.0, shaft_off=17.0, hole_dx=74.0, hole_dy=18.0)

HORN_GAP = 4.0   # расстояние от щеки до верха сервопривода (тут сидит качалка)


# ---------------------------------------------------------------- helpers
def box(x0, x1, y0, y1, z0, z1):
    return M.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl_z(r, z0, z1, x=0, y=0, seg=SEG):
    return M.cylinder(z1 - z0, r, r, seg).translate((x, y, z0))


def cyl_y(r, y0, y1, x=0, z=0, seg=SEG):
    # цилиндр вдоль оси Y
    return M.cylinder(y1 - y0, r, r, seg).rotate((-90, 0, 0)).translate((x, y0, z))


def cyl_x(r, x0, x1, y=0, z=0, seg=SEG):
    return M.cylinder(x1 - x0, r, r, seg).rotate((0, 90, 0)).translate((x0, y, z))


def profile_y(cs, y0, y1):
    """Профиль в плоскости XZ (cs: x->X, y->Z), выдавленный по Y."""
    return M.extrude(cs, y1 - y0).rotate((90, 0, 0)).translate((0, y1, 0))


def circ(r, x=0, y=0, seg=SEG):
    return C.circle(r, seg).translate((x, y))


def union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def ring_z(r_out, r_in, z0, z1):
    return cyl_z(r_out, z0, z1) - cyl_z(r_in, z0 - 1, z1 + 1)


def gear_cs(teeth, module, bore=0.0, seg_per_tooth=10):
    """Эвольвентная шестерня (угол 20°)."""
    pa = math.radians(20)
    rp = teeth * module / 2
    rb = rp * math.cos(pa)
    ra = rp + module
    rf = max(rp - 1.25 * module, 0.5)

    def inv(a):
        return math.tan(a) - a

    # половина толщины зуба на делительной окружности (угол)
    half = math.pi / (2 * teeth) + inv(pa)
    pts = []
    for i in range(teeth):
        c = 2 * math.pi * i / teeth
        side = []
        for k in range(seg_per_tooth + 1):
            r = max(rb, rf) + (ra - max(rb, rf)) * k / seg_per_tooth
            a = math.acos(min(1.0, rb / r))
            side.append((r, half - inv(a)))
        left = [(r * math.cos(c - t), r * math.sin(c - t)) for r, t in side]
        right = [(r * math.cos(c + t), r * math.sin(c + t)) for r, t in reversed(side)]
        root_a = c - half - 0.6 * math.pi / teeth
        pts.append((rf * math.cos(root_a + 0.3 * math.pi / teeth), rf * math.sin(root_a + 0.3 * math.pi / teeth)))
        pts += left + right
        pts.append((rf * math.cos(c + half + 0.3 * math.pi / teeth), rf * math.sin(c + half + 0.3 * math.pi / teeth)))
    g = C([pts])
    if bore:
        g = g - circ(bore / 2, seg=32)
    return g


# ---------------------------------------------------------------- серво
def servo_body(s):
    """Сервопривод: вал по +Z в начале координат, верх корпуса в z=0."""
    L, W, H = s["L"], s["W"], s["H"]
    cx = -(L / 2 - s["shaft_off"])  # центр корпуса по X; вал ближе к +X концу
    body = box(cx - L / 2, cx + L / 2, -W / 2, W / 2, -H, 0)
    ez = -s["ear_from_top"]
    ears = box(cx - s["ear_span"] / 2, cx + s["ear_span"] / 2, -W / 2, W / 2, ez - s["ear_t"], ez)
    for sx in (-1, 1):
        for sy in (-1, 1):
            ears = ears - cyl_z(1.2, ez - 5, ez + 1, cx + sx * s["hole_dx"] / 2, sy * s["hole_dy"] / 2, 16)
    hub = cyl_z(W * 0.3, 0, 2) + cyl_z(2.9, 0, 4, seg=24)
    return body + ears + hub, cx


def servo_cradle(s, wall=3.5):
    """Держатель сервопривода. Ставится на щеку сустава.
    Система координат как у servo_body; плоскость крепления к щеке z=+HORN_GAP."""
    L, W, H = s["L"], s["W"], s["H"]
    _, cx = servo_body(s)
    ez = -s["ear_from_top"]
    outer = box(cx - s["ear_span"] / 2 - 4.5, cx + s["ear_span"] / 2 + 4.5, -W / 2 - wall, W / 2 + wall,
                -H - wall, HORN_GAP)
    pocket = box(cx - L / 2 - TOL, cx + L / 2 + TOL, -W / 2 - TOL, W / 2 + TOL, -H - 0.01, HORN_GAP + 1)
    ear_slot = box(cx - s["ear_span"] / 2 - TOL, cx + s["ear_span"] / 2 + TOL, -W / 2 - TOL, W / 2 + TOL,
                   ez - s["ear_t"] - TOL, HORN_GAP + 1)
    # окно для качалки
    horn = cyl_z(14, -1, HORN_GAP + 1)
    c = outer - pocket - ear_slot - horn
    # дно открыто для проводов
    c = c - box(cx - L / 2 + 6, cx + L / 2 - 6, -W / 2 + 3, W / 2 - 3, -H - wall - 1, -H + 1)
    # винты к щеке
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = c - cyl_z(M3 / 2, -H - wall - 1, HORN_GAP + 1,
                          cx + sx * (s["ear_span"] / 2 + 0.5), sy * (W / 2 + wall / 2 - 0.2), 20)
    # стойки под ушки сервопривода (вкручиваются саморезы)
    return c


def cradle_screw_holes(s, length=40):
    """Отверстия в щеке под винты держателя (в системе координат сервопривода)."""
    _, cx = servo_body(s)
    holes = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            holes.append(cyl_z(M3 / 2, -5, length, cx + sx * (s["ear_span"] / 2 + 2.4),
                               sy * (s["W"] / 2 + 1.55), 20))
    return union(holes)


def horn_boss(r=11.5, y0=0, y1=10):
    """Цилиндрическая бобышка на звене под крестовую/круглую металлическую качалку (4 винта М3)."""
    b = cyl_y(r, y0, y1)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        b = b - cyl_y(1.25, y1 - 8, y1 + 1, 7 * math.cos(a), 7 * math.sin(a), 16)
    b = b - cyl_y(4.0, y1 - 2, y1 + 1)  # под винт вала
    return b


def servo_pad(s, y0, y1, zc):
    """Площадка на щеке сустава, к которой привинчивается держатель сервопривода."""
    _, cx = servo_body(s)
    return box(cx - s["ear_span"] / 2 - 4.5, cx + s["ear_span"] / 2 + 4.5, y0, y1,
               zc - s["W"] / 2 - 3.5, zc + s["W"] / 2 + 3.5)


def servo_housing(s, top, wall=3.5):
    """(сплошное тело, полость) гнезда сервопривода: вал в (0,0) вверх по +Z, верх корпуса в z=top."""
    _, cx = servo_body(s)
    solid = box(cx - s["ear_span"] / 2 - 2, cx + s["ear_span"] / 2 + 2, -s["W"] / 2 - wall, s["W"] / 2 + wall,
                top - s["H"] - 2, top)
    cav = box(cx - s["L"] / 2 - TOL, cx + s["L"] / 2 + TOL, -s["W"] / 2 - TOL, s["W"] / 2 + TOL,
              top - s["H"], top + 1)
    cav = cav + box(cx - s["ear_span"] / 2 - TOL, cx + s["ear_span"] / 2 + TOL, -s["W"] / 2 - TOL,
                    s["W"] / 2 + TOL, top - s["ear_from_top"] - s["ear_t"] + 1.5, top + 1)
    cav = cav + box(cx - 5, cx + 5, -s["W"] / 2 - wall - 1, -s["W"] / 2 + 1, top - s["H"] + 1,
                    top - s["H"] + 9)  # выход проводов
    return solid, cav


# Поворот: сервопривод с валом вдоль -Y, верх корпуса на плоскости y = y_face + HORN_GAP
def place_servo_y(part, y_face):
    # servo frame: shaft +Z, top z=0  -> world: shaft -Y, top at y_face+HORN_GAP
    return part.rotate((90, 0, 0)).translate((0, y_face + HORN_GAP, 0))


# ================================================================= ДЕТАЛИ
def base_housing():
    h = cyl_z(BASE_R, 0, BASE_H)
    h = h - cyl_z(BASE_R - 4, 3, BASE_H - 6)            # полость
    h = h - cyl_z(RACE_R - 10, BASE_H - 7, BASE_H + 1)  # центр открыт (шкив)
    h = h - M.revolve(circ(BALL / 2 + 0.15, RACE_R, BASE_H, 32), SEG * 2)  # дорожка шариков
    h = h - (cyl_z(BASE_R + 1, 26, 34) - cyl_z(BASE_R - 2.5, 25, 35))       # канавка LED-кольца
    # центральная стойка под ось М8
    h = h + cyl_z(11, 3, 36) - cyl_z(M8 / 2, -1, 37)
    # салазки шагового мотора NEMA17: мотор сдвигается по X (центр от -40 до -70) для натяжения ремня
    for sy in (-1, 1):
        y0 = sy * (21.15 + 0.3)
        h = h + box(-92, -24, min(y0, y0 + sy * 4), max(y0, y0 + sy * 4), 3, 18)
    for k in range(3):
        h = h - box(-80 + k * 18, -74 + k * 18, -32, 32, 10, 14)   # окна под стяжки
    h = h - box(-75, -35, -3, 3, -1, 4)                            # паз под винт-фиксатор
    # вентиляция в дне
    for i in range(-3, 4):
        h = h - box(20, 60, i * 9 - 1.5, i * 9 + 1.5, -1, 4)
    # аварийная кнопка Ø22 и окно под разъёмы сзади
    h = h - cyl_y(11.25, -BASE_R - 2, -BASE_R + 6, 0, 30).rotate((0, 0, -35))
    h = h - box(-BASE_R - 2, -BASE_R + 6, -14, 14, 12, 26)
    # проушины для крепления к столу
    for k in range(4):
        a = math.radians(45 + 90 * k)
        x, y = 88 * math.cos(a), 88 * math.sin(a)
        h = h + cyl_z(6, 3, 10, x, y, 24) - cyl_z(2.2, -1, 11, x, y, 16)
    return h


def led_ring():
    return cyl_z(BASE_R + 0.6, 26.3, 33.7) - cyl_z(BASE_R - 2.2, 25, 35)


def pulley_100t():
    """Шкив GT2 100 зубьев. z: -1.5..16, ступица сверху (крепится к платформе 4×М3)."""
    r = 100 * 2 / math.pi / 2 - 0.254
    p = cyl_z(r, 0, 8, seg=200)
    teeth = [cyl_z(0.6, -1, 9, (r + 0.1) * math.cos(2 * math.pi * i / 100),
                   (r + 0.1) * math.sin(2 * math.pi * i / 100), 10) for i in range(100)]
    p = p - union(teeth)
    p = p + cyl_z(r + 1.5, 8, 9.5, seg=200) + cyl_z(r + 1.5, -1.5, 0, seg=200) + cyl_z(15, 9.5, 16)
    p = p - cyl_z(M8 / 2, -2, 17)
    for k in range(4):
        a = math.radians(90 * k)
        p = p - cyl_z(M3 / 2, -2, 17, 10 * math.cos(a), 10 * math.sin(a), 16)
    for k in range(6):
        a = math.radians(60 * k)
        p = p - cyl_z(6.5, -2, 17, 22 * math.cos(a), 22 * math.sin(a), 32)
    return p


def turntable():
    """Поворотная платформа со стойками плеча. Начало = ось вращения на полу."""
    z0, z1 = PLAT_Z, PLAT_Z + PLAT_T
    gap, t = UA_Y + 2, 12
    p = cyl_z(92, z0, z1)
    p = p - M.revolve(circ(BALL / 2 + 0.15, RACE_R, z0 - 0.3, 32), SEG * 2)
    p = p + cyl_z(15, z0 - 4, z0 + 0.1)
    # стойки
    prof = C.batch_hull([C.square((80, 1)).translate((-40, z1 - 1)), circ(30, 0, SHOULDER_Z)])
    p = p + profile_y(prof, gap / 2, gap / 2 + t) + profile_y(prof, -gap / 2 - t, -gap / 2)
    pad = servo_pad(SERVO_BIG, gap / 2, gap / 2 + t, SHOULDER_Z)
    p = p + M.batch_hull([pad, box(-40, 40, gap / 2, gap / 2 + t, z1 - 1, z1)])
    # ребро жёсткости сзади
    p = p + box(-40, -34, -gap / 2 - t, gap / 2 + t, z1 - 1, SHOULDER_Z - 30)
    # стойка пружины-противовеса
    p = p + cyl_z(6, z1 - 1, z1 + 30, -70, 0)
    # --- вырезы
    p = p - cyl_z(M8 / 2, z0 - 10, z1 + 1)
    for k in range(4):
        a = math.radians(90 * k)
        p = p - cyl_z(M3 / 2, z0 - 5, z0 + 6, 10 * math.cos(a), 10 * math.sin(a), 16)
    p = p - cyl_x(1.8, -80, -60, 0, z1 + 25, 16)
    p = p - cyl_y(M8 / 2, -gap / 2 - t - 1, 0, 0, SHOULDER_Z)
    p = p - cyl_y(12.2, 0, gap / 2 + t + 1, 0, SHOULDER_Z)
    p = p - cradle_screw_holes(SERVO_BIG, 60).rotate((90, 0, 0)).translate((0, gap / 2 + t + HORN_GAP, SHOULDER_Z))
    # каналы для проводов
    p = p - cyl_z(5, z0 - 1, z1 + 1, -55, 30, 24)
    return p, gap, t


def upper_arm():
    """Плечо. Начало — ось плеча, звено вдоль +Z, оси вдоль Y."""
    hx, hy = UA_X / 2, UA_Y / 2
    slot = FA_Y / 2 + 1
    a = profile_y(C.batch_hull([circ(hx), circ(hx, 0, L_UPPER)]), -hy, hy)
    a = a + servo_pad(SERVO_STD, slot, hy, L_UPPER)              # площадка серво локтя
    a = a + horn_boss(11.5, hy, UA_Y / 2 + 1 + 12 - 0.3)           # под качалку плеча
    a = a + box(-hx - 10, -hx + 1, -5, 5, 30, 42)                  # крюк пружины
    # --- вырезы
    a = a - box(-60, hx + 1, -slot, slot, L_UPPER - 22, L_UPPER + 40)  # вилка локтя
    a = a - cyl_y(M8 / 2, -hy - 1, hy + 14, 0, 0)
    a = a - cyl_y(M8 / 2, -hy - 1, -slot + 0.01, 0, L_UPPER)
    a = a - cyl_y(12.2, slot - 0.01, hy + 1, 0, L_UPPER)
    a = a - cradle_screw_holes(SERVO_STD).rotate((90, 0, 0)).translate((0, hy + HORN_GAP, L_UPPER))
    for sx in (-1, 1):
        a = a - cyl_z(4.25, 18, L_UPPER - 21, sx * 10, 0, 24)       # шпильки М8
    win = C.batch_hull([circ(4.5, 0, 38), circ(4.5, 0, L_UPPER - 45)])
    a = a - profile_y(win, -hy - 1, -6) - profile_y(win, 6, hy + 1)
    a = a - cyl_y(1.8, -6, 6, -hx - 5, 36, 16)
    return a


def forearm():
    """Предплечье с вилкой запястья. Начало — ось локтя."""
    hx, hy = FA_X / 2, FA_Y / 2
    wy, slot = 20, 10
    f = profile_y(C.batch_hull([circ(hx), circ(hx, 0, L_FORE * 0.55)]), -hy, hy)
    f = f + M.batch_hull([profile_y(circ(hx, 0, L_FORE * 0.5), -hy, hy), profile_y(circ(18, 0, L_FORE), -wy, wy)])
    f = f + servo_pad(SERVO_STD, slot, wy, L_FORE)
    f = f + horn_boss(11.5, hy, UA_Y / 2 - 0.3)
    f = f - box(-60, 20, -slot, slot, L_FORE - 20, L_FORE + 40)
    f = f - cyl_y(M8 / 2, -hy - 1, hy + 1, 0, 0)
    f = f - cyl_y(M8 / 2, -wy - 1, -slot + 0.01, 0, L_FORE)
    f = f - cyl_y(12.2, slot - 0.01, wy + 1, 0, L_FORE)
    f = f - cradle_screw_holes(SERVO_STD).rotate((90, 0, 0)).translate((0, wy + HORN_GAP, L_FORE))
    for sx in (-1, 1):
        f = f - cyl_z(4.25, 16, L_FORE - 19, sx * 8, 0, 24)
    return f


WR_H = 68  # от оси наклона запястья до выхода вала вращения


def wrist_block():
    """Блок запястья: язычок на оси наклона + гнездо сервопривода вращения (вал по +Z)."""
    ty = 9
    s = SERVO_STD
    top = WR_H - 3
    w = profile_y(C.batch_hull([circ(15), C.square((30, 1)).translate((-15, 20))]), -ty, ty)
    w = w + horn_boss(11.5, ty, 19.7)
    solid, cav = servo_housing(s, top)
    _, cx = servo_body(s)
    w = w + M.batch_hull([box(-15, 15, -ty, ty, 20, 21),
                          box(cx - s["ear_span"] / 2 - 2, cx + s["ear_span"] / 2 + 2, -s["W"] / 2 - 3.5,
                              s["W"] / 2 + 3.5, top - s["H"] - 2, top - s["H"] - 1)])
    w = w + solid
    w = w - cav - cyl_y(M8 / 2, -ty - 1, 21, 0, 0)
    return w, top


# ------------------------------------------------------------ захват
G_MOD = 1.25
G_T = 16
G_R = G_MOD * G_T / 2        # 10 мм
G_Z = 62                     # высота осей шестерён над выходом вала вращения
LINK = 40                    # длина поводков
LINK_DZ = 16                 # расстояние между осями пальца
FINGER_IN = 12               # вынос губки пальца внутрь
WORM_R = 6.0
G_GRIP_TOP = 46              # верх корпуса серво захвата
WX = -G_R - (G_R + WORM_R)   # ось червяка
WY = -2.25
GRIP_SERVO_ROT = 90          # сервопривод захвата повёрнут вдоль Y


def gripper_body():
    s = SERVO_STD
    top = G_GRIP_TOP
    solid, cav = servo_housing(s, top)
    solid = solid.rotate((0, 0, GRIP_SERVO_ROT)).translate((WX, WY, 0))
    cav = cav.rotate((0, 0, GRIP_SERVO_ROT)).translate((WX, WY, 0))
    sb = solid.bounding_box()
    flange = cyl_z(16, 0, 5)
    base = M.batch_hull([flange, box(sb[0], sb[3], sb[1], sb[4], 0, 5)])
    b = base + solid
    for (y0, y1) in ((-9.5, -6.0), (6.5, 10.0)):
        plate = C.batch_hull([circ(7, -G_R, G_Z), circ(7, G_R, G_Z), circ(6, -G_R, G_Z + LINK_DZ),
                              circ(6, G_R, G_Z + LINK_DZ), C.square((32, 1)).translate((-16, 3))])
        b = b + profile_y(plate, y0, y1)
    # верхняя опора червяка
    b = b + box(WX - 6, WX + 6, -15, -10.5, 3, 81)
    b = b + box(WX - 6, WX + 6, -15, WY + 6, 77, 81)
    # --- вырезы
    b = b - cav
    b = b - cyl_z(2.1, 75, 82, WX, WY, 24)
    for k in range(4):
        a = math.radians(45 + 90 * k)
        b = b - cyl_z(1.6, -1, 6, 7 * math.cos(a), 7 * math.sin(a), 16)
    b = b - cyl_z(4.2, -1, 6, seg=24)
    for x in (-G_R, G_R):
        for z in (G_Z, G_Z + LINK_DZ):
            b = b - cyl_y(M3 / 2, -12, 12, x, z, 20)
    return b


def worm():
    """Червяк (однозаходный, модуль 1.25). Ось Z; снизу фланец под круглую качалку."""
    lead = math.pi * G_MOD
    ra, rf = WORM_R + G_MOD, WORM_R - 1.25 * G_MOD
    n = 96
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        d = abs(((th / (2 * math.pi)) % 1.0) - 0.5) * 2
        r = ra if d < 0.3 else (ra - (ra - rf) * (d - 0.3) / 0.4 if d < 0.7 else rf)
        pts.append((r * math.cos(th), r * math.sin(th)))
    length = 20
    w = M.extrude(C([pts]), length, int(length / lead * 24), -360 * length / lead).translate((0, 0, 7))
    w = w + cyl_z(10, 0, 3) + cyl_z(rf, 3, 7.01, seg=32) + cyl_z(1.9, 26.9, 33, seg=24)
    for k in range(2):
        a = math.radians(180 * k)
        w = w - cyl_z(1.1, -1, 4, 7 * math.cos(a), 7 * math.sin(a), 12)
    w = w - cyl_z(1.6, -1, 10, seg=16)
    return w  # z 0..33


def gear_with_arm(worm_layer, phase=0.0):
    """Шестерня + поводок пальца. Ось вдоль Y в начале координат, поводок вверх (+Z)."""
    spur = profile_y(gear_cs(G_T, G_MOD).rotate(90 + phase), 0.5, 6.0)
    arm = profile_y(C.batch_hull([circ(5), circ(4, 0, LINK)]), 0.5, 4.5)
    g = spur + arm
    if worm_layer:
        lead_ang = math.atan(math.pi * G_MOD / (2 * math.pi * WORM_R))
        twist = math.degrees(5.5 * math.tan(lead_ang) / G_R)
        wheel = M.extrude(gear_cs(G_T, G_MOD), 5.5, 6, twist).rotate((90, 0, 0)).translate((0, 0.5, 0))
        g = g + wheel
    g = g - cyl_y(M3 / 2 + 0.05, -6, 7, 0, 0, 20) - cyl_y(M3 / 2 + 0.05, -1, 7, 0, LINK, 20)
    return g


def idle_link():
    return profile_y(C.batch_hull([circ(4.5), circ(4.5, 0, LINK)]) - circ(M3 / 2 + 0.05, 0, 0, 16)
                     - circ(M3 / 2 + 0.05, 0, LINK, 16), 0.5, 4.5)


JAW_Z0, JAW_Z1 = LINK_DZ + 4, LINK_DZ + 44


def finger():
    """Левый палец (губка смотрит в +X). Оси: (0,0) и (0,LINK_DZ) в плоскости XZ."""
    prof = C.batch_hull([circ(5), circ(5, 0, LINK_DZ)])
    prof = prof + C.batch_hull([circ(4, 0, LINK_DZ), C.square((6, 1)).translate((FINGER_IN - 6, JAW_Z0 + 2))])
    prof = prof + C([[(FINGER_IN - 6, JAW_Z0), (FINGER_IN, JAW_Z0), (FINGER_IN, JAW_Z1 - 4),
                      (FINGER_IN - 2, JAW_Z1), (FINGER_IN - 6, JAW_Z1 - 6)]])
    prof = prof - circ(M3 / 2 + 0.05, 0, 0, 16) - circ(M3 / 2 + 0.05, 0, LINK_DZ, 16)
    f = profile_y(prof, -5.5, 0.5)
    f = f - box(FINGER_IN - 1.2, FINGER_IN + 1, -4.5, -0.5, JAW_Z0 + 4, JAW_Z1 - 6)  # паз под накладку
    return f


def finger_pad():
    p = box(0, 2.2, -4.3, -0.7, JAW_Z0 + 4.2, JAW_Z1 - 6.2)
    n = int((JAW_Z1 - JAW_Z0 - 11) // 4.2)
    for i in range(n):
        p = p + box(2.1, 3.4, -5.2, 0.2, JAW_Z0 + 5 + i * 4.2, JAW_Z0 + 7 + i * 4.2)
    return p.translate((FINGER_IN - 1.0, 0, 0))


# ------------------------------------------------------------ декор
def armor(half_w, depth, z0, z1, x_face):
    """Выпуклая накладка-броня на переднюю грань звена (клеится)."""
    R = (half_w ** 2 + depth ** 2) / (2 * depth)
    lens = circ(R, x_face - R + depth, 0, 160) ^ C.square((depth + 0.01, 2 * half_w + 2)).translate(
        (x_face, -half_w - 1))
    a = M.extrude(lens, z1 - z0).translate((0, 0, z0))
    ell = M.sphere(1, 96).scale((depth * 6, half_w * 1.6, (z1 - z0) / 2 * 1.08)).translate((x_face, 0, (z0 + z1) / 2))
    a = a ^ ell
    for k in (-1, 1):
        a = a - box(x_face + depth - 1.0, x_face + depth + 2, k * half_w * 0.45 - 0.8, k * half_w * 0.45 + 0.8,
                    z0 + 12, z1 - 12)
    return a


def joint_cap(r=15, t=3):
    c = cyl_y(r, 0, t) + cyl_y(r * 0.55, t, t + 1.2)
    c = c - (cyl_y(r * 0.78, t - 0.8, t + 2) - cyl_y(r * 0.66, t - 1, t + 2))
    c = c - cyl_y(7.5, -0.01, t - 1.2)  # под головку болта
    return c


# ------------------------------------------------------------ не печатаемое (для картинки)
def nema17():
    return box(-21.15, 21.15, -21.15, 21.15, 0, 40) + cyl_z(11, 40, 42) + cyl_z(2.5, 42, 66, seg=24)


def gt2_20t():
    r = 20 * 2 / math.pi / 2 - 0.254
    return cyl_z(r, -7, 0, seg=40) + cyl_z(r + 1.5, -8, -7, seg=40) + cyl_z(r + 1.5, 0, 1, seg=40) + cyl_z(8, 1, 8)


def belt(c_dist, r1, r2, z0, z1):
    outer = C.batch_hull([circ(r1 + 1.4, 0, 0, 160), circ(r2 + 1.4, c_dist, 0, 48)])
    inner = C.batch_hull([circ(r1 + 0.2, 0, 0, 160), circ(r2 + 0.2, c_dist, 0, 48)])
    return M.extrude(outer - inner, z1 - z0).translate((0, 0, z0))


def balls():
    n = int(2 * math.pi * RACE_R / (BALL + 0.4))
    return union([M.sphere(BALL / 2, 12).translate((RACE_R * math.cos(2 * math.pi * i / n),
                                                     RACE_R * math.sin(2 * math.pi * i / n), BASE_H + 0.3))
                  for i in range(n)])


def bolt_y(y0, y1, x=0, z=0, r=4):
    return cyl_y(r, y0, y1, x, z, 24) + cyl_y(r * 1.65, y0 - 5, y0, x, z, 6)


def rod_z(z0, z1, x=0, y=0):
    return cyl_z(4, z0, z1, x, y, 16)


# ================================================================= сборка
def build():
    parts = {}

    def add(name, m, link, mat, printable=True):
        parts[name] = dict(m=m, link=link, mat=mat, printable=printable)

    gap, t = UA_Y + 2, 12
    ss, _ = servo_body(SERVO_STD)
    sb, _ = servo_body(SERVO_BIG)

    # ---------- 0: основание
    add("01_base_housing", base_housing(), 0, "dark")
    add("02_led_ring", led_ring(), 0, "glow")
    add("nema17", nema17().translate((-50, 0, 3)), 0, "motor", False)
    add("gt2_20t", gt2_20t().translate((-50, 0, 48)), 0, "metal", False)
    add("belt", belt(50, 100 / math.pi, 20 / math.pi, 41, 48).rotate((0, 0, 180)), 0, "rubber", False)
    add("balls", balls(), 0, "metal", False)
    add("estop", (cyl_y(11, -4, 0) + cyl_y(20, -14, -4) + cyl_y(15, -17, -14))
        .translate((0, -BASE_R, 30)).rotate((0, 0, -35)), 0, "estop", False)

    # ---------- 1: платформа
    tt, gap, t = turntable()
    yf = gap / 2 + t
    add("03_turntable_shoulder", tt, 1, "dark")
    add("04_pulley_100T", pulley_100t().translate((0, 0, PLAT_Z - 4 - 16)), 1, "gold")
    add("servo_shoulder_DS5160", place_servo_y(sb, yf).translate((0, 0, SHOULDER_Z)), 1, "servo", False)
    add("05_cradle_shoulder_big", place_servo_y(servo_cradle(SERVO_BIG), yf).translate((0, 0, SHOULDER_Z)), 1, "dark")
    add("bolt_shoulder", bolt_y(-yf, UA_Y / 2 - 2, 0, SHOULDER_Z), 1, "metal", False)
    add("21a_cap_shoulder", joint_cap(17).rotate((0, 0, 180)).translate((0, -yf - 5, SHOULDER_Z)), 1, "gold")

    # ---------- 2: плечо
    add("06_upper_arm", upper_arm(), 2, "dark")
    add("rods_upper", union([rod_z(19, L_UPPER - 22, sx * 10) for sx in (-1, 1)]), 2, "metal", False)
    add("20a_armor_upper_front", armor(15, 7, 22, L_UPPER - 26, UA_X / 2), 2, "red")
    add("20b_armor_upper_back", armor(15, 6, 48, L_UPPER - 26, UA_X / 2).mirror((1, 0, 0)), 2, "red")
    add("servo_elbow", place_servo_y(ss, UA_Y / 2).translate((0, 0, L_UPPER)), 2, "servo", False)
    add("07_cradle_elbow", place_servo_y(servo_cradle(SERVO_STD), UA_Y / 2).translate((0, 0, L_UPPER)), 2, "dark")
    add("bolt_elbow", bolt_y(-UA_Y / 2, FA_Y / 2 - 2, 0, L_UPPER), 2, "metal", False)
    add("21b_cap_elbow", joint_cap(14).rotate((0, 0, 180)).translate((0, -UA_Y / 2 - 5, L_UPPER)), 2, "gold")

    # ---------- 3: предплечье
    add("08_forearm", forearm(), 3, "dark")
    add("rods_fore", union([rod_z(17, L_FORE - 20, sx * 8) for sx in (-1, 1)]), 3, "metal", False)
    add("20c_armor_fore", armor(12, 6, 14, 52, FA_X / 2), 3, "red")
    add("servo_wrist_pitch", place_servo_y(ss, 20).translate((0, 0, L_FORE)), 3, "servo", False)
    add("09_cradle_wrist", place_servo_y(servo_cradle(SERVO_STD), 20).translate((0, 0, L_FORE)), 3, "dark")
    add("bolt_wrist", bolt_y(-20, 7, 0, L_FORE), 3, "metal", False)
    add("21c_cap_wrist", joint_cap(12).rotate((0, 0, 180)).translate((0, -25, L_FORE)), 3, "gold")

    # ---------- 4: запястье
    wb, wtop = wrist_block()
    add("10_wrist_block", wb, 4, "dark")
    add("servo_wrist_roll", ss.translate((0, 0, wtop)), 4, "servo", False)
    add("22_diffuser_wrist", cyl_z(19, wtop - 1.5, wtop) - cyl_z(15, wtop - 2, wtop + 1), 4, "glow")

    # ---------- 5: захват
    add("11_gripper_body", gripper_body(), 5, "dark")
    add("servo_gripper", ss.rotate((0, 0, GRIP_SERVO_ROT)).translate((WX, WY, G_GRIP_TOP)), 5, "servo", False)
    add("12_worm", worm().translate((WX, WY, G_GRIP_TOP + 3.0)), 5, "gold")
    add("13_gear_left_worm", gear_with_arm(True), 6, "gold")
    add("14_gear_right", gear_with_arm(False, 360 / G_T / 2).mirror((1, 0, 0)), 7, "gold")
    add("15a_link_left", idle_link(), 8, "dark")
    add("15b_link_right", idle_link().mirror((1, 0, 0)), 9, "dark")
    add("16a_finger_left", finger(), 10, "dark")
    add("16b_finger_right", finger().mirror((1, 0, 0)), 11, "dark")
    add("17a_pad_left", finger_pad(), 10, "tpu")
    add("17b_pad_right", finger_pad().mirror((1, 0, 0)), 11, "tpu")
    add("pins_gripper", union([cyl_y(1.5, -11, 11, x, z, 12) for x in (-G_R, G_R) for z in (G_Z, G_Z + LINK_DZ)]),
        5, "metal", False)
    return parts


# Положение звеньев (для просмотрщика): родитель, точка оси в системе родителя, ось вращения
LINKS = {
    0: dict(parent=None, origin=(0, 0, 0), axis=None, name="Основание"),
    1: dict(parent=0, origin=(0, 0, 0), axis=(0, 0, 1), name="Поворот 360°"),
    2: dict(parent=1, origin=(0, 0, SHOULDER_Z), axis=(0, 1, 0), name="Плечо"),
    3: dict(parent=2, origin=(0, 0, L_UPPER), axis=(0, 1, 0), name="Локоть"),
    4: dict(parent=3, origin=(0, 0, L_FORE), axis=(0, 1, 0), name="Запястье: наклон"),
    5: dict(parent=4, origin=(0, 0, WR_H), axis=(0, 0, 1), name="Запястье: вращение"),
    6: dict(parent=5, origin=(-G_R, 0, G_Z), axis=(0, 1, 0), name="grip"),
    7: dict(parent=5, origin=(G_R, 0, G_Z), axis=(0, 1, 0), name="grip"),
    8: dict(parent=5, origin=(-G_R, 0, G_Z + LINK_DZ), axis=(0, 1, 0), name="grip"),
    9: dict(parent=5, origin=(G_R, 0, G_Z + LINK_DZ), axis=(0, 1, 0), name="grip"),
    10: dict(parent=5, origin=(0, 0, 0), axis=None, name="finger"),
    11: dict(parent=5, origin=(0, 0, 0), axis=None, name="finger"),
}
# начало координат каждого звена в его собственной геометрии: геометрия строится
# с осью сустава в (0,0,0), кроме звеньев 0 и 1 (мир) и 5..11 (в системе звена 5)
LOCAL_AT_ORIGIN = {0: False, 1: False, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: True,
                   10: True, 11: True}


def mesh_arrays(m):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties[:, :3], dtype=np.float32)
    f = np.asarray(mesh.tri_verts, dtype=np.uint32)
    return v, f


def write_stl(path, v, f):
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln == 0, 1, ln)
    with open(path, "wb") as fh:
        fh.write(b"SocialTech robo arm".ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        data = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        data["n"] = n
        data["v"] = tri
        fh.write(data.tobytes())


# поворот для печати: кладём деталь самой большой плоской стороной на стол
PRINT_ROT = {
    "01_base_housing": (0, 0, 0),
    "02_led_ring": (0, 0, 0),
    "03_turntable_shoulder": (0, 0, 0),
    "04_pulley_100T": (180, 0, 0),
    "05_cradle_shoulder_big": (90, 0, 0),
    "06_upper_arm": (90, 0, 0),
    "07_cradle_elbow": (90, 0, 0),
    "08_forearm": (90, 0, 0),
    "09_cradle_wrist": (90, 0, 0),
    "10_wrist_block": (90, 0, 0),
    "11_gripper_body": (0, 0, 0),
    "12_worm": (0, 0, 0),
    "13_gear_left_worm": (-90, 0, 0),
    "14_gear_right": (-90, 0, 0),
    "15a_link_left": (90, 0, 0),
    "15b_link_right": (90, 0, 0),
    "16a_finger_left": (90, 0, 0),
    "16b_finger_right": (90, 0, 0),
    "17a_pad_left": (0, -90, 0),
    "17b_pad_right": (0, 90, 0),
    "20a_armor_upper_front": (0, -90, 0),
    "20b_armor_upper_back": (0, 90, 0),
    "20c_armor_fore": (0, -90, 0),
    "21a_cap_shoulder": (90, 0, 0),
    "21b_cap_elbow": (90, 0, 0),
    "21c_cap_wrist": (90, 0, 0),
    "22_diffuser_wrist": (0, 0, 0),
}


def main():
    os.makedirs(STL_DIR, exist_ok=True)
    os.makedirs(VIEW_DIR, exist_ok=True)
    parts = build()

    # ---- STL для печати
    report = []
    for name, p in parts.items():
        if not p["printable"]:
            continue
        m = p["m"].rotate(PRINT_ROT.get(name, (0, 0, 0)))
        bb = m.bounding_box()
        m = m.translate((-(bb[0] + bb[3]) / 2, -(bb[1] + bb[4]) / 2, -bb[2]))
        v, f = mesh_arrays(m)
        write_stl(os.path.join(STL_DIR, name + ".stl"), v, f)
        bb = m.bounding_box()
        size = (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
        report.append((name, size, p["m"].volume() / 1000.0, m.genus(), str(m.status())))
        p["size"] = [round(x, 1) for x in size]
        p["vol"] = round(p["m"].volume() / 1000.0, 1)

    # ---- геометрия для просмотрщика
    blob = bytearray()
    meta = []
    for name, p in parts.items():
        v, f = mesh_arrays(p["m"])
        meta.append(dict(name=name, link=p["link"], mat=p["mat"], printable=p["printable"],
                         size=p.get("size"), vol=p.get("vol"),
                         nv=int(len(v)), nf=int(len(f)), off=len(blob)))
        blob += v.astype("<f4").tobytes()
        idx = f.astype("<u2") if len(v) < 65536 else f.astype("<u4")
        meta[-1]["idx32"] = bool(len(v) >= 65536)
        blob += idx.tobytes()
        while len(blob) % 4:
            blob += b"\0"
    import base64
    with open(os.path.join(VIEW_DIR, "arm.b64.txt"), "w") as fh:
        fh.write(base64.b64encode(gzip.compress(bytes(blob), 9)).decode())
    links = {k: dict(parent=v["parent"], origin=v["origin"], axis=v["axis"]) for k, v in LINKS.items()}
    with open(os.path.join(VIEW_DIR, "arm.json"), "w") as fh:
        json.dump(dict(parts=meta, links=links,
                       consts=dict(G_R=G_R, G_Z=G_Z, LINK=LINK, LINK_DZ=LINK_DZ, FINGER_IN=FINGER_IN,
                                   SHOULDER_Z=SHOULDER_Z, L_UPPER=L_UPPER, L_FORE=L_FORE, WR_H=WR_H)),
                  fh, ensure_ascii=False)

    print(f"{'деталь':32s} {'размер для печати, мм':>26s} {'объём, см3':>10s}")
    tot = 0
    for name, size, vol, genus, st in report:
        tot += vol
        flag = "" if st.endswith("NoError") else "  !! " + st
        big = "  !! не влезает в 256" if max(size[0], size[1]) > 250 or size[2] > 250 else ""
        print(f"{name:32s} {size[0]:7.1f} x {size[1]:6.1f} x {size[2]:6.1f}  {vol:8.1f}{flag}{big}")
    print(f"всего объём деталей: {tot:.0f} см3 (сплошных); вес при 40% заполнении ~{tot * 1.27 * 0.55:.0f} г PETG")


if __name__ == "__main__":
    main()
