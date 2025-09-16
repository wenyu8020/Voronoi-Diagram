# $LAN=Python$
"""
版本宣告
學號：M123040016
姓名：張紋瑜
"""
from tkinter import *
from tkinter import filedialog
# from calculate import *

import numpy as np
import os
import threading
import copy


# 求中點
def midpoint(data_a, data_b):
    return [(data_a[0] + data_b[0]) / 2, (data_a[1] + data_b[1]) / 2]


# 求斜率
def slope(line):
    if line[1][0] - line[0][0] == 0:
        return None
    return (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])


# 求兩點向量
def vector(a, b):
    return [b[0] - a[0], b[1] - a[1]]


# 求兩點法向量（交換一變號）
def normal_vector(a, b):
    vector_ab = vector(a, b)

    temp = vector_ab[0]
    vector_ab[0] = vector_ab[1]
    vector_ab[1] = temp
    vector_ab[0] *= -1

    return vector_ab


# 外積
def outer_product(cen, a, b):
    return (a[0] - cen[0]) * (b[1] - cen[1]) - (a[1] - cen[1]) * (b[0] - cen[0])


# 內積
def inner_product(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


# 確定是否共線
def collinearity(a, b, c):
    if (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0]) == 0:
        return True
    return False


# 確定點是否有在線上
def point_on_line_segment(point, line_segment):
    P1, P2 = line_segment[0], line_segment[1]

    vec_P1P = [point[0] - P1[0], point[1] - P1[1]]
    vec_P1P2 = [P2[0] - P1[0], P2[1] - P1[1]]
    vec_P2P = [point[0] - P2[0], point[1] - P2[1]]
    vec_P2P1 = [P1[0] - P2[0], P1[1] - P2[1]]

    dot1 = vec_P1P[0] * vec_P1P2[0] + vec_P1P[1] * vec_P1P2[1]
    dot2 = vec_P2P[0] * vec_P2P1[0] + vec_P2P[1] * vec_P2P1[1]

    if dot1 >= 0 and dot2 >= 0:
        return True
    return False


# 距離公式
def get_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# 求外心，套公式
def get_circumcentre(a, b, c):
    if a == b == c:
        return a

    d1 = (b[0] ** 2 + b[1] ** 2) - (a[0] ** 2 + a[1] ** 2)
    d2 = (c[0] ** 2 + c[1] ** 2) - (b[0] ** 2 + b[1] ** 2)
    fm = 2 * ((c[1] - b[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - b[0]))
    x = ((c[1] - b[1]) * d1 - (b[1] - a[1]) * d2) / fm
    y = ((b[0] - a[0]) * d2 - (c[0] - b[0]) * d1) / fm
    return [int(x), int(y)]


# 求重心，套公式
def get_centroid(point):
    cx = 0
    cy = 0
    for i in range(len(point)):
        cx += point[i][0]
        cy += point[i][1]
    return [cx / len(point), cy / len(point)]


# 求線段兩邊的點
def get_side_point(a, b):
    # 邊界
    max_x = 600
    max_y = 600
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    # 是否共點
    if dx == 0 and dy == 0:
        return a
    # 是否共線（垂直）
    elif dx == 0:
        if dy > 0:
            return [a[0], max_y]
        else:
            return [a[0], 0]
    # 是否共線（水平）
    elif dy == 0:
        if dx > 0:
            return [max_x, a[1]]
        else:
            return [0, a[1]]
    else:
        m = float(dy) / float(dx)  # 斜率
        d = a[1] - m * a[0]  # 截距
        # 斜截式線性方程式 y=mx+d
        # 斜率正：\ 斜率負：/ （跟一般座標不同，因畫布座標上下顛倒）
        # x = 0, y = d
        # 碰到畫布左邊
        if 0 <= d <= max_y:
            # 往左下 or 往左上
            if (m < 0 and dy > 0) or (m > 0 and dy < 0):
                return [0, int(d)]
        # x = max_x, y = max_x*m + d
        # 碰到畫布右邊
        if 0 <= max_x * m + d <= max_y:
            # 往右上 or 往右下
            if (m < 0 and dy < 0) or (m > 0 and dy > 0):
                return [max_x, int(max_x * m + d)]
        # x = -d/m, y = 0
        # 碰到畫布上邊
        if 0 < -d / m < max_x:
            # 往右上 or 往左上
            if (m < 0 and dy < 0) or (m > 0 and dy < 0):
                return [int(-d / m), 0]
        # x = (max_y-d)/m, y = max_y
        # 碰到畫布下邊
        if 0 < (max_y - d) / m < max_x:
            # 往右下 or 往左下
            if (m < 0 and dy > 0) or (m > 0 and dy > 0):
                return [int((max_y - d) / m), max_y]
        return [-1, -1]


# 求中垂線
def get_midperpendicular(line):
    mid = midpoint(line[0], line[1])
    t1 = [mid[i] + normal_vector(line[0], line[1])[i] * 600 for i in range(len(normal_vector(line[0], line[1])))]
    t2 = [mid[i] + normal_vector(line[1], line[0])[i] * 600 for i in range(len(normal_vector(line[1], line[0])))]

    return [t2, t1, line[0], line[1]]


# 求兩線交點
def get_intersection_point(line1, line2):
    # ax + by + c = 0
    a1 = line1[0][1] - line1[1][1]
    b1 = line1[1][0] - line1[0][0]
    c1 = line1[0][0] * line1[1][1] - line1[1][0] * line1[0][1]
    a2 = line2[0][1] - line2[1][1]
    b2 = line2[1][0] - line2[0][0]
    c2 = line2[0][0] * line2[1][1] - line2[1][0] * line2[0][1]

    D = a1 * b2 - a2 * b1
    if D == 0:
        return []
    x = (b1 * c2 - b2 * c1) / D
    y = (a2 * c1 - a1 * c2) / D

    # 確定點是否都在兩條線上
    if point_on_line_segment([x, y], line1) and point_on_line_segment([x, y], line2):
        return [np.round(x), np.round(y)]
    else:
        return []


# 求Convex Hull
def get_convexhull(point):
    lower = []
    for p in point:
        while len(lower) >= 2 and outer_product(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(point):
        while len(upper) >= 2 and outer_product(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper


# 點以逆時針排序
def sort_counterclockwise(point):
    # 重心是為了得到點的向量（類似時鐘）
    centroid = get_centroid(point)
    for i in range(len(point) - 1):
        for j in range(len(point) - i - 1):
            # 兩向量外積大於0，表示順時針
            if outer_product(centroid, point[j], point[j + 1]) > 0:
                temp = point[j + 1]
                point[j + 1] = point[j]
                point[j] = temp
    return point


# 排序點
def sort_point(point):
    point = set([tuple(t) for t in point])
    point = list([list(t) for t in point])
    point.sort(key=lambda arr: (arr[0], arr[1]))
    return point


# 線段E x1 y1 x2 y2，座標須滿足x1≦x2 或 x1=x2, y1≦y2
# 不同線段之間，依照x1, y1, x2, y2的順序進行排序
def sort_edge(edge):
    for i in range(len(edge)):
        if edge[i][0][0] > edge[i][1][0] or (edge[i][0][0] == edge[i][1][0] and edge[i][0][1] > edge[i][1][1]):
            temp = edge[i][0]
            edge[i][0] = edge[i][1]
            edge[i][1] = temp
    edge.sort(key=lambda arr: (arr[0], arr[1]))
    return edge


# 將點分成兩部分
def divide(point, side):
    side_point = []
    if side == 0:
        for i in range(len(point) // 2):
            side_point.append(point[i])
    elif side == 1:
        for i in range(len(point) // 2, len(point)):
            side_point.append(point[i])
    return side_point


# 原本得出的線是無限延伸的，為了輸出文字檔，將線段的點變成在邊界內
def edge_adjustment(edge):
    for i in range(len(edge)):
        mid = midpoint(edge[i][2], edge[i][3])
        t1 = edge[i][0][:]
        t2 = edge[i][1][:]
        if edge[i][0][0] > 600 or edge[i][0][0] < 0 or edge[i][0][1] > 600 or edge[i][0][1] < 0:
            edge[i][0] = get_side_point(mid, t1)
        if edge[i][1][0] > 600 or edge[i][1][0] < 0 or edge[i][1][1] > 600 or edge[i][1][1] < 0:
            edge[i][1] = get_side_point(mid, t2)
        edge[i][0][0] = int(edge[i][0][0])
        edge[i][0][1] = int(edge[i][0][1])
        edge[i][1][0] = int(edge[i][1][0])
        edge[i][1][1] = int(edge[i][1][1])

    return edge


# 初始化Voronoi Diagram
class Voronoi:
    # 點、邊、點的數量
    def __init__(self, point_list=None, edge_list=None):
        if point_list is None:
            point_list = []
        if edge_list is None:
            edge_list = []
        self.diagram = []
        self.data = point_list
        self.edge = edge_list
        self.hyperplane_edge = []
        self.n = len(point_list)


def add_point(p):
    # 放入list中
    v.data.append(p)
    update_coordinates_text(v)
    draw_point(p, "black")


# 在用滑鼠點擊時畫出點
def mouse_click(event):
    global v, v_list
    if len(v_list):
        if count < len(v_list):
            v = v_list[count]
    # 在畫布範圍內畫點
    if 0 <= event.x <= 600 and 0 <= event.y <= 600:
        # 放入list中
        add_point([event.x, event.y])
        update_coordinates_text(v)


# 畫點
def draw_point(p, color):
    canvas.create_oval(p[0], p[1], p[0], p[1], fill=color, width=3)


# 畫線
def draw_edge(a, b, color):
    canvas.create_line(a[0], a[1], b[0], b[1], fill=color, width=1.5, smooth=True)


# 畫Convex Hull
def draw_convexhull(point):
    for i in range(len(point)):
        draw_edge(point[i], point[(i + 1) % len(point)], '#90baf5')


# 畫整個圖
def draw_diagram(e, color):
    # 有共點的情況，需要將不能成為edge刪除
    i = len(e) - 1
    while i >= 0:
        if e[i][0] == e[i][1]:
            del e[i]
        i -= 1
    # 畫線
    for edge in e:
        draw_edge(edge[0], edge[1], color)


# 將畫布上的點更新在右欄，以便查看有哪些點
def update_coordinates_text(vor):
    coordinates_text.delete(1.0, END)
    for i in range(len(vor.data)):
        coordinates_text.insert(END, f"({vor.data[i][0]}, {vor.data[i][1]})\n")


# 執行
def run():
    global count, v, v_list, click_run, click_step
    click_run = True
    click_step = False
    clear_canvas()
    if len(v.data) > 0:
        for i in range(len(v.data)):
            draw_point(v.data[i], "black")
            update_coordinates_text(v)
        v.data = sort_point(v.data)
        draw_convexhull(get_convexhull(v.data))
        v = form_diagram(v.data)
        v.edge = edge_adjustment(v.edge)
        draw_diagram(v.edge, "red")


def form_diagram(point):
    global click_step
    vor = Voronoi(point)
    # 只有一點，不須畫線
    if len(point) == 1:
        return vor
    # 兩點的情況
    elif len(point) == 2:
        vor = voronoi_two_nodes(point[0], point[1])
        if click_step:
            step.append([vor, 1])
        sort_edge(vor.edge)
        return vor
    # 三點的情況
    elif len(point) == 3:
        vor = voronoi_three_nodes(point[0], point[1], point[2])
        if click_step:
            step.append([vor, 1])
        sort_edge(vor.edge)
        return vor
    # 三點以上
    else:
        left_point = divide(point, 0)
        right_point = divide(point, 1)
        # 需要合併
        vor = merge(form_diagram(left_point), form_diagram(right_point))
        if click_step:
            step.append([vor, 3])
        return vor


# 兩點的情況
def voronoi_two_nodes(a, b):
    mid = midpoint(a, b)  # 求中點
    vor = Voronoi()
    vor.data.append(a)
    vor.data.append(b)

    # 沒有延伸edge超出邊界的情況
    # 求出兩個法向量座標
    t1 = [mid[i] + normal_vector(a, b)[i]*600 for i in range(min(len(mid), len(normal_vector(a, b))))]
    t2 = [mid[i] + normal_vector(b, a)[i]*600 for i in range(min(len(mid), len(normal_vector(b, a))))]

    # 從中點延伸，往法線方向延伸edge到邊界
    vor.edge.append([t1, t2, a, b])

    return vor


# 三點
def voronoi_three_nodes(a, b, c):
    vor = Voronoi()
    vor.data.append(a)
    vor.data.append(b)
    vor.data.append(c)
    point_list = list(vor.data)
    # 將點照逆時針方向排序，才能將法線往外延伸
    point_list = sort_counterclockwise(point_list)

    # 判斷三點共線
    if collinearity(a, b, c):
        mid = midpoint(a, b)  # 中點

        # 第一條edge
        # 沒有延伸edge超出邊界的情況
        # 求出兩個法向量座標
        t1 = [mid[i] + normal_vector(a, b)[i]*600 for i in range(min(len(mid), len(normal_vector(a, b))))]
        t2 = [mid[i] + normal_vector(b, a)[i]*600 for i in range(min(len(mid), len(normal_vector(b, a))))]
        # 從中點延伸，往法線方向延伸edge無限延伸
        vor.edge.append([t1, t2, a, b])

        # 第二條edge，同上
        mid = midpoint(b, c)
        t3 = [mid[i] + normal_vector(c, b)[i]*600 for i in range(min(len(mid), len(normal_vector(c, b))))]
        t4 = [mid[i] + normal_vector(b, c)[i]*600 for i in range(min(len(mid), len(normal_vector(b, c))))]
        vor.edge.append([t3, t4, b, c])
    else:
        # 非三點共線
        # 求出外心，到三個點等距離
        circumcentre = get_circumcentre(a, b, c)
        # 畫出三條edge
        for i in range(len(point_list)):
            mid = midpoint(point_list[i], point_list[(i + 1) % 3])  # 中點
            # p1p2, p2p3, p3p1都有法向量
            t1 = [mid[j] + normal_vector(point_list[i], point_list[(i + 1) % 3])[j]*600 for j in range(len(mid))]
            # 從外心延伸，往法向量的方向無限延伸
            vor.edge.append([circumcentre, t1, point_list[i], point_list[(i + 1) % 3]])

    return vor


# 合併
def merge(v1, v2):
    global click_step
    vor = Voronoi()
    temp_vor = Voronoi()
    left_ch = get_convexhull(v1.data)
    right_ch = get_convexhull(v2.data)
    tangent, ch = get_tangent(left_ch, right_ch)
    down_tangent = tangent[1][:]
    down_tangent.reverse()

    # 進行深層複製以確保元素是獨立的
    vor.data = copy.deepcopy(v1.data) + copy.deepcopy(v2.data)
    vor.edge = copy.deepcopy(v1.edge) + copy.deepcopy(v2.edge)
    temp_vor.data = copy.deepcopy(vor.data)
    temp_vor.edge = copy.deepcopy(vor.edge)

    hp_edge = []
    ip_list = []
    last_intersection_point = get_midperpendicular(tangent[0])[0]
    intersection_point = []
    last_intersection_line = []
    intersection_line = []
    touched_line = [False] * len(vor.edge)

    # 兩個VD之間的連線，從最上面的切線開始
    connected_line = tangent[0][:]
    while connected_line != tangent[1] and connected_line != down_tangent:
        hyperplane = get_midperpendicular(connected_line)
        intersection_point = []
        for i in range(len(vor.edge)):
            if len(last_intersection_line) and last_intersection_line == vor.edge[i]:
                continue
            p = get_intersection_point(hyperplane, vor.edge[i])
            # 往下看有交點
            if len(p):
                if last_intersection_point[1] <= p[1]:
                    if len(intersection_point) == 0:
                        intersection_point = p
                        intersection_line = vor.edge[i]
                        continue
                    # 求距離最短的線
                    if get_distance(hyperplane[0], p) < get_distance(hyperplane[0], intersection_point):
                        intersection_point = p
                        intersection_line = vor.edge[i]

        if len(last_intersection_point):
            hyperplane[0] = last_intersection_point
        hp_edge.append([hyperplane[0], intersection_point, connected_line[0], connected_line[1]])
        ip_list.append(intersection_point)
        last_intersection_point = intersection_point
        last_intersection_line = intersection_line

        # 找下一個VD之間的連接線
        if connected_line[0] == intersection_line[2]:
            connected_line[0] = intersection_line[3][:]
        elif connected_line[0] == intersection_line[3]:
            connected_line[0] = intersection_line[2][:]
        elif connected_line[1] == intersection_line[2]:
            connected_line[1] = intersection_line[3][:]
        elif connected_line[1] == intersection_line[3]:
            connected_line[1] = intersection_line[2][:]

        # 根據hyper plane跟接觸線的斜率來判斷要刪線的部分
        index = vor.edge.index(intersection_line)
        if len(intersection_point) == 0:
            continue
        else:
            mh = slope(hp_edge[len(hp_edge) - 1])
            ml = slope(intersection_line)
            if mh is None:
                if intersection_line in v1.edge:
                    temp_vor.edge[index][1] = intersection_point
                elif intersection_line in v2.edge:
                    temp_vor.edge[index][0] = intersection_point
            else:
                if intersection_line in v1.edge:
                    if ml is None:
                        temp_vor.edge[index][1] = intersection_point
                    elif ml / mh < 0:
                        temp_vor.edge[index][1] = intersection_point
                    elif ml / mh >= 0:
                        if ml > 0:
                            if ml > mh:
                                temp_vor.edge[index][0] = intersection_point
                            elif ml < mh:
                                temp_vor.edge[index][1] = intersection_point
                        else:
                            if ml > mh:
                                temp_vor.edge[index][1] = intersection_point
                            elif ml < mh:
                                temp_vor.edge[index][0] = intersection_point
                elif intersection_line in v2.edge:
                    if ml is None:
                        temp_vor.edge[index][0] = intersection_point
                    elif ml / mh < 0:
                        temp_vor.edge[index][0] = intersection_point
                    elif ml / mh >= 0:
                        if ml >= 0:
                            if ml > mh:
                                temp_vor.edge[index][1] = intersection_point
                            elif ml < mh:
                                temp_vor.edge[index][0] = intersection_point
                        else:
                            if ml > mh:
                                temp_vor.edge[index][0] = intersection_point
                            elif ml < mh:
                                temp_vor.edge[index][1] = intersection_point

            touched_line[index] = True

    if tangent[0] == tangent[1] or tangent[0] == down_tangent:
        hp_edge.append([get_midperpendicular(tangent[0])[0], get_midperpendicular(tangent[0])[1], connected_line[0],
                        connected_line[1]])
    else:
        hp_edge.append([intersection_point, get_midperpendicular(tangent[1])[0], connected_line[0], connected_line[1]])

    # 紀錄步驟
    if click_step:
        step.append([vor, 2])
    num = 0
    touchedNode = []
    ch_edge = []
    # 再確定是否有需要刪線的部分
    for i in range(len(ch)):
        ch_edge.append([ch[i], ch[(i + 1) % len(ch)]])

    for i in range(len(touched_line)):
        if touched_line[i]:
            touchedNode.append(temp_vor.edge[i][0])
            touchedNode.append(temp_vor.edge[i][1])
    for e in temp_vor.edge[:]:
        check = False
        if len(touched_line) <= num:
            break
        if touched_line[num] is False:
            if e[0] in touchedNode or e[1] in touchedNode:
                num += 1
                continue
            for i in range(len(ch_edge)):
                if inner_product(vector(e[0], e[1]), vector(ch_edge[i][0], ch_edge[i][1])) == 0 and len(
                        get_intersection_point(e, ch_edge[i])):
                    num += 1
                    check = True
                    break
            if not check:
                temp_vor.edge.remove(e)
        num += 1

    vor.edge = copy.deepcopy(temp_vor.edge + hp_edge)
    vor.hyperplane_edge = hp_edge
    sort_edge(vor.edge)

    return vor


# 得上下切線
def get_tangent(lch, rch):
    t = []
    point = lch + rch
    point = sort_point(point)
    whole_ch = get_convexhull(point)

    for i in range(len(whole_ch)):
        if (whole_ch[i] in lch and whole_ch[(i + 1) % len(whole_ch)] in rch) or (
                whole_ch[i] in rch and whole_ch[(i + 1) % len(whole_ch)] in lch):
            t.append([whole_ch[i], whole_ch[(i + 1) % len(whole_ch)]])

    return t, whole_ch


# 清空所有東西
def clear_canvas():
    canvas.delete(ALL)
    coordinates_text.delete("1.0", "end")


def clear():
    global v, count, v_list, step_num, step
    v = Voronoi()
    step = []
    step_num = 0
    if len(v_list):
        v_list[count] = Voronoi()
    clear_canvas()


# 一步一步做
def step_by_step():
    global step, step_num, color_num, v, click_run, click_step
    click_step = True
    # 清空
    if len(step) <= 0 or step_num >= len(step) or click_run:
        clear_canvas()
        step = []
        step_num = 0
        color_num = 0
        click_run = False

        v.data = sort_point(v.data)
        v = form_diagram(v.data)

    if step_num == 0:
        clear_canvas()

    if len(v.data) > 0:
        for i in range(len(v.data)):
            draw_point(v.data[i], "black")
            update_coordinates_text(v)

    # 1: 畫左右兩邊VD, 2: Hyper Plane, 3: Merge
    if step[step_num][1] == 1:
        color_num += 1
        if color_num % 2 == 1:
            draw_diagram(step[step_num][0].edge, '#722cab')
            draw_convexhull(get_convexhull(step[step_num][0].data))
        else:
            draw_diagram(step[step_num][0].edge, '#cc88fc')
            draw_convexhull(get_convexhull(step[step_num][0].data))
    elif step[step_num][1] == 2:
        for i in range(len(step[step_num][0].hyperplane_edge)):
            draw_edge(step[step_num][0].hyperplane_edge[i][0], step[step_num][0].hyperplane_edge[i][1], '#f0963c')
    elif step[step_num][1] == 3:
        canvas.delete(ALL)
        for i in range(len(v.data)):
            draw_point(v.data[i], "black")
        draw_diagram(step[step_num][0].edge, "red")
        draw_convexhull(get_convexhull(step[step_num][0].data))

    if step_num < len(step):
        step_num += 1
    else:
        step_num = 0


# 下一個測資
def next_case():
    global count, v, v_list
    canvas.delete(ALL)
    coordinates_text.delete("1.0", "end")
    if count < len(v_list):
        count += 1
        if count < len(v_list):
            v = v_list[count]
            update_coordinates_text(v)
            for j in range(len(v.data)):
                draw_point(v.data[j], "black")
            draw_diagram(v.edge, "red")


# 讀檔
def read_file():
    global v_list, count, v
    v_list = []
    count = 0
    clear_canvas()
    file_path = filedialog.askopenfilename()  # 開資料夾選檔案
    out_file = False
    if file_path:
        with open(file_path, "r", encoding="utf-8") as file:
            i = 0
            vor = Voronoi()
            for line in file.readlines():
                if line[0][0].isdigit():  # 讀in檔
                    if i == 0:
                        if int(line) == 0:
                            break
                        vor.n = int(line)
                    else:
                        line = line.split()
                        vor.data.append([int(line[0]), int(line[1])])
                    i += 1
                    if i == vor.n + 1:
                        i = 0
                        v_list.append(vor)
                        vor = Voronoi()
                elif line[0][0].isalpha():  # 讀out檔
                    out_file = True
                    if line[0][0] == 'P':  # point
                        line = line.split()
                        v_list[0].data.append([int(line[1]), int(line[2])])
                    if line[0][0] == 'E':  # edge
                        line = line.split()
                        v_list[0].edge.append([[int(line[1]), int(line[2])], [int(line[3]), int(line[4])]])
            v = v_list[0]
            if out_file:
                draw_diagram(v.edge, "red")
            update_coordinates_text(v)
            for j in range(len(v.data)):
                draw_point(v.data[j], "black")


# 存檔
def save_file():
    file = filedialog.asksaveasfilename(initialfile="vd_testdata.out")  # 打開資料夾
    with open(file, 'w') as f:
        if len(v.data) > 0:
            for i in range(len(v.data)):
                f.write("P " + str(v.data[i][0]) + " " + str(v.data[i][1]) + '\n')
            for i in range(len(v.edge)):
                f.write(
                    "E " + str(v.edge[i][0][0]) + " " + str(v.edge[i][0][1]) + " " + str(v.edge[i][1][0]) + " " + str(
                        v.edge[i][1][1]) + '\n')
    f.close()


if __name__ == '__main__':
    # 視窗
    window = Tk()
    window.title("Voronoi")
    window.geometry("700x640")
    window.minsize(width=700, height=640)
    window.resizable(width=False, height=False)

    v_list = []
    count = 0
    v = Voronoi()
    step = []
    step_num = 0
    color_num = 0
    click_run = False
    click_step = False

    # 畫布
    canvas = Canvas(window, width=600, height=600, background="white")
    canvas.bind('<Button-1>', mouse_click)
    canvas.place(x=0, y=0)
    # 右欄座標框
    coordinates_text = Text(window, width=10, height=550)
    coordinates_text.place(x=610, y=0)

    btn_frame = Frame(window)
    btn_frame.place(x=0, y=610)

    # 下個case
    next_btn = Button(btn_frame, text="Next", command=next_case)
    next_btn.pack(side=LEFT)
    # 執行
    btn1 = Button(btn_frame, text="Run", command=run)
    btn1.pack(side=LEFT)
    # 一步一步做
    btn2 = Button(btn_frame, text="Step by step", command=step_by_step)
    btn2.pack(side=LEFT)
    # 清理畫布
    btn3 = Button(btn_frame, text="Clear", command=clear)
    btn3.pack(side=LEFT)
    # 讀取檔案
    btn4 = Button(btn_frame, text="Open", command=read_file)
    btn4.pack(side=LEFT)
    # 存檔
    btn5 = Button(btn_frame, text="Save", command=save_file)
    btn5.pack(side=LEFT)

    window.mainloop()
