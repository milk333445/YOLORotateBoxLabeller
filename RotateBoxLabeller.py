import cv2
import numpy as np
import copy
from RotateBoxLabeller_config import ConfigManager
from pathlib import Path
import argparse
import os
import sys



def get_key_action(key):
    return key_actions.get(key, 'invalid key')


def show_xy_rotated_rect(event, x, y, flags, param):
    global drawing, center_point, right_clicking_dragging, rotated_pts, prev_mouse_move
    
    
    expand_image = copy.deepcopy(param[1])
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            drawing = True
            center_point = (x, y)
            # 繪製中心點
            cv2.circle(expand_image, center_point, 5, (0, 0, 255), -1)
            
            current_category = obj[param[4]]
            
            info_text = f'current category: {current_category}'
            
            cv2.putText(expand_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow(param[2], expand_image)
        else:
            drawing = False
            new_obj = param[5] + [param[4]]
            param[0].append(rotated_pts.tolist())
            param[3].append(center_point)
            param[5] = new_obj
            print('param[0]:', param[0])
            print('param[5]:', param[5])
            
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        right_clicking_dragging = True
    
    
    elif event == cv2.EVENT_RBUTTONUP:
        right_clicking_dragging = False
        prev_mouse_move = None
    
    
    elif event == cv2.EVENT_MOUSEMOVE:
        temp_image = expand_image.copy()
        if drawing:
            
            if len(param[0]) > 0:
                for i, pts in enumerate(param[0]):
                    cv2.polylines(temp_image, [np.array(param[0][i], dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
                    
                    label = obj[param[5][i]]
                    label_position = (int(pts[0][0]), int(pts[0][1]) - 5)
                    cv2.putText(temp_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    
                 
            dx = x - center_point[0]
            dy = y - center_point[1]
            distance = np.sqrt(dx**2 + dy**2)
            angle = np.degrees(np.arctan2(dy, dx))
        
            
            # 計算旋轉矩形的四個點
            rect_w = int(distance * 2)
            rect_h = int(distance)
            angle_rad = np.radians(angle)
            pts = np.array([
                [-rect_w//2, -rect_h//2],
                [rect_w//2, -rect_h//2],
                [rect_w//2, rect_h//2],
                [-rect_w//2, rect_h//2]
            ])

            rotated_pts = []
            for pt in pts:
                x_val = pt[0] * np.cos(angle_rad) - pt[1] * np.sin(angle_rad) + center_point[0]
                y_val = pt[0] * np.sin(angle_rad) + pt[1] * np.cos(angle_rad) + center_point[1]
                rotated_pts.append([x_val, y_val])
            rotated_pts = np.array(rotated_pts, dtype=np.int32)
            
            
            # 繪製旋轉的矩形和中心點
            
            cv2.polylines(temp_image, [rotated_pts], isClosed=True, color=(255, 0, 0), thickness=2)
            
            left_top_x = min([pt[0] for pt in rotated_pts])
            left_top_y = min([pt[1] for pt in rotated_pts])
            category = obj[param[4]]
            
            cv2.putText(temp_image, category, (left_top_x, left_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            
            cv2.circle(temp_image, center_point, 5, (0, 0, 255), -1)
            
            
            current_category = obj[param[4]]
            info_text = f'current category: {current_category}'
            cv2.putText(temp_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            
            

            cv2.imshow(param[2], temp_image)
            
            
            if right_clicking_dragging:
                if not prev_mouse_move:
                    prev_mouse_move = (x, y)
                
                dx = x - prev_mouse_move[0]
                dy = y - prev_mouse_move[1]
                
                center_point = (center_point[0] + dx, center_point[1] + dy)
                
                # 跟新所有點
                for i, pt in enumerate(rotated_pts):
                    rotated_pts[i] = (pt[0] + dx, pt[1] + dy)
                    
                prev_mouse_move = (x, y)
                
                temp_image = expand_image.copy()
                
                if len(param[0]) > 0:
                    for i, pt in enumerate(param[0]):
                        cv2.polylines(temp_image, [np.array(param[0][i], dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
                        label = obj[param[5][i]]
                        label_position = (int(pt[0][0]), int(pt[0][1]) - 5)
                        cv2.putText(temp_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.polylines(temp_image, [rotated_pts], isClosed=True, color=(255, 0, 0), thickness=2)
                left_top_x = min([pt[0] for pt in rotated_pts])
                left_top_y = min([pt[1] for pt in rotated_pts])
                category = obj[param[4]]
            
                cv2.putText(temp_image, category, (left_top_x, left_top_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                current_category = obj[param[4]]
                
                info_text = f'current category: {current_category}'
                
                cv2.putText(temp_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                
                cv2.circle(temp_image, center_point, 5, (0, 0, 255), -1)
                cv2.imshow(param[2], temp_image)
            
    
   
   
def move_edge(points, edge_index1, edge_index2, offset):
    # 計算多邊形的中心點
    centroid_x = sum([pt[0] for pt in points]) / len(points)
    centroid_y = sum([pt[1] for pt in points]) / len(points)

    # 計算所選邊的中點
    midpoint_x = (points[edge_index1][0] + points[edge_index2][0]) / 2
    midpoint_y = (points[edge_index1][1] + points[edge_index2][1]) / 2

    # 計算從中心點到所選邊的中點的向量
    direction_dx = midpoint_x - centroid_x
    direction_dy = midpoint_y - centroid_y

    # 正規化該向量
    magnitude = np.sqrt(direction_dx**2 + direction_dy**2)
    direction_dx /= magnitude
    direction_dy /= magnitude

    # 使用該向量和offset進行移動
    points[edge_index1][0] += offset * direction_dx
    points[edge_index1][1] += offset * direction_dy
    points[edge_index2][0] += offset * direction_dx
    points[edge_index2][1] += offset * direction_dy



   
  
  
def draw_selected_edge(lst_a_full, selected_edge):
    points = lst_a_full[0][-1]
    img = lst_a_full[1].copy()
  
    if len(lst_a_full[0]) > 0:
        for i, pt in enumerate(lst_a_full[0]):
            cv2.polylines(img, [np.array(lst_a_full[0][i], dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
            label = obj[lst_a_full[5][i]]
            cv2.putText(img, label, (int(pt[0][0]), int(pt[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.polylines(img, [np.array(points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    
    
    # 畫選擇的邊
    if selected_edge == 'top':
        cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), color=(0, 0, 255), thickness=2)
        
    elif selected_edge == 'bottom':
        cv2.line(img, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), color=(0, 0, 255), thickness=2)
        
    elif selected_edge == 'left':
        cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[3][0]), int(points[3][1])), color=(0, 0, 255), thickness=2)
        
    elif selected_edge == 'right':
        cv2.line(img, (int(points[1][0]), int(points[1][1])), (int(points[2][0]), int(points[2][1])), color=(0, 0, 255), thickness=2)
        
    return img
    
    
    
def update_image_and_label(lst_a, obj, count2, action):
    item = count2 % len(obj)
    current_category = obj[item]
    lst_a[4] = item
    img_tmp = lst_a[1].copy()
    if len(lst_a[0]) > 0:
        for i, pt in enumerate(lst_a[0]):
            cv2.polylines(img_tmp, [np.array(pt, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
            label = obj[lst_a[5][i]]
            label_position = (int(pt[0][0]), int(pt[0][1]) - 5)
            cv2.putText(img_tmp, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # 添加当前类别的提示信息
    info_text = f'current category: {current_category}'
    cv2.putText(img_tmp, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow(lst_a[2], img_tmp)

def initialize_parameters(last_time_num, source):
    source = str(source)
    if last_time_num is None:
        img_count = 0
    else:
        img_count = max(0, last_time_num)
    return source, img_count

def get_image_files(source_dir):
    files = sorted(os.listdir(Path(source_dir))) # 得到圖片列表
    img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(img_files)
    return img_files, total_images


def get_image_path_and_name(source, img_files, img_count, save_dir):
    img_path = os.path.join(source, img_files[img_count]) # 圖片路徑
    file_name, _ = os.path.splitext(img_files[img_count])
    sav_img = os.path.join(save_dir, file_name + '.txt')
    return img_path, file_name, sav_img

def run_obb_label(
    last_time_num = None,
    source = './images',
    store = './labels',
):
    
    global drawing, center_point, right_clicking_dragging, rotated_pts, prev_mouse_move, selected_edge, offset, key_actions, obj
    source, img_count = initialize_parameters(last_time_num, source)
    img_files, total_images = get_image_files(source)
    
    while img_count < total_images:
        img_path, file_name, save_img = get_image_path_and_name(source, img_files, img_count, store)
        print('Current image path:', img_path)
        
        img = cv2.imread(img_path)
        
        lst_obj = []
        lst_a = [[], img, file_name, [], 0, lst_obj] 
        # 第一個裝矩形座標，
        # 第二個裝圖片
        # 第三個裝圖片名稱
        # 第四個裝中心點
        # 第五個裝預設類別
        # 第六個裝真正的使用者所選類別
        cv2.imshow(file_name, img)
        cv2.setMouseCallback(file_name, show_xy_rotated_rect, lst_a)
        import os

        count = 0
        count2 = 0
        while True:
            key = cv2.waitKey(0)
            action = get_key_action(key)
            if action == 'exit':
                print('Program Terminated')
                quit()
            
            temp_image = lst_a[1].copy()
            
            edges = ['top', 'bottom', 'left', 'right']
            
                
            
            if action == 'switch_next_side':
                
                current_index = count % len(edges)
                selected_edge = edges[current_index]
                count += 1
                
                
            
            if action == 'switch_next':
                count2 += 1
                update_image_and_label(lst_a, obj, count2, action)
                
                
            if action == 'switch_prev':
                count2 -= 1
                update_image_and_label(lst_a, obj, count2, action)
                
                
            if action == 'delete':
                if lst_a[0]:
                    lst_a[0].pop()
                    lst_a[3].pop()
                    print('lst_a[0]:', lst_a[0])
                    print('lst_a[3]:', lst_a[3])
                    print('delete successfully')   
                    img_tmp = lst_a[1].copy()
                    if len(lst_a[0]) > 0:
                        for i in range(len(lst_a[0])):
                            cv2.polylines(img_tmp, [np.array(lst_a[0][i], dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
                    cv2.imshow(lst_a[2], img_tmp)
                    
                    
                else:
                    print('delete failed')
                    
            if action == 'save':
                
                with open(save_img, 'w+') as file:  # 打開文件用於寫入，如果文件不存在則創建
                    for i, coords in enumerate(lst_a[0]):
                        flat_coords = [str(float(int(item))) for sublist in coords for item in sublist]
                        category = obj[lst_a[5][i]]
                        line = ' '.join(flat_coords) + ' ' + category + ' 0'
                        file.write(line + '\n')
                print('save successfully')
                img_count += 1
                break
                  
                
                
            if action == 'pass':
                img_count += 1
                break
            
            if action == 'previous':
                img_count -= 1
                break
            
            if action not in key_actions.values():
                print('Input error. Please re-enter.')
                continue
            
    
                
            if not selected_edge or not lst_a[0]:
                continue
            
            if selected_edge == 'top':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 0, 1, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 0, 1, -offset)
                    
            elif selected_edge == 'bottom':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 2, 3, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 2, 3, -offset)
                    
            elif selected_edge == 'left':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 0, 3, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 0, 3, -offset)
                    
            elif selected_edge == 'right':
                if action == 'plus':
                    move_edge(lst_a[0][-1], 1, 2, offset)
                elif action == 'minus':
                    move_edge(lst_a[0][-1], 1, 2, -offset)
                    
            if selected_edge:
                temp_image = draw_selected_edge(lst_a, selected_edge)
                    
                    
            cv2.imshow(lst_a[2], temp_image)
            
                    
        cv2.destroyAllWindows()
            
        
    
    

if __name__ == '__main__':   

    rotated_pts = np.array([])
    rotated_pts = None
    drawing = False
    right_clicking_dragging = False
    selected_edge = None
    offset = 5
    prev_mouse_move = None


    config_manager = ConfigManager()
    conf = config_manager.get_config()
    key_actions = conf['key_actions']
    obj = conf['obj']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--last_time_num', type=int, default=None, help='The last time you labeled the image')
    parser.add_argument('--source', type=str, default='./images', help='The source of images')
    parser.add_argument('--store', type=str, default='./labels', help='The path to store labels')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.store):
        os.makedirs(args.store)
        
    if not os.path.exists(args.source):
        print('The source path does not exist.')
        exit()
        
    if not os.path.isdir(args.source):
        print('The source path is not a directory.')
        exit()
    
    if not os.path.isdir(args.store):
        print('The store path is not a directory.')
        exit()
        
    run_obb_label(args.last_time_num, args.source, args.store)
    
    

