'''
Combining YOLO with Visual Rhythm for Vehicle Counting  

Usage:
    python count.py [options]    
    --line: line position                           [defalt:  600]
    --interval: interval between VR images (frames) [default: 900]
    --save-VR: enable saving VR images              [default: False]
    --save-vehicle: enable saving vehicle images    [default: False]
    
    The video path is asked in the execution.
    
    The results are printed in the terminal 
    and saved detaily in 'infos.txt' and.
'''



import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

import argparse
import os


def counting(line, frame, l, r, time, model_vehicle, saveVehicle):
    labels = ['Bus', 'Car', 'Motorbike', 'Pickup', 'Truck', 'Van', '???'] 
    delta = 70
    count = np.array([0, 0, 0, 0, 0, 0, 0])
    infos = []
    height , width, _ = np.shape(frame)


    # detect vehicles
    vehicles = model_vehicle(frame, iou=0.5, conf=0.3, verbose=False, save=True, agnostic_nms=True)
    vehicles = vehicles[0].numpy()
    
    
    # extend the mark's x-coordinates
    # l = max(l - delta, 0) + aa
    # r = min(r + delta, width) + aa
    
    
    # find the vehicle associated with the mark
    dist_min = 600
    crop = frame.copy()
    classe = conf = -1
    x0 = y0 = x1 = y1 = 0
    for cls, xyxy, cnf in zip(vehicles.boxes.cls.astype(int), vehicles.boxes.xyxy, vehicles.boxes.conf):
        x0, y0, x1, y1 = np.floor(xyxy).astype(int)
        x0 = max(x0 - delta, 0) + aa
        x1 = min(x1 + delta, width) + aa

        mid_x = (r + l) // 2
        dist = abs(y1-line)
        print(y0,y1, abs(y1-line), x0, x1, mid_x, x0 < mid_x < x1)
        if y1 < line and x0 < mid_x < x1 and dist < dist_min:
            dist_min = dist
            classe = int(cls)
            conf = cnf
            # if saveVehicle:
            crop = frame[y0:y1, x0:x1]
                
    
    count[classe] += 1
    cv2.imshow('croppp', crop)
    infos.append([labels[classe], frame, conf, x0, y0, x1, y1])
    print(f'{labels[classe]} detected in frame {time}\n')

    if conf < 0:
        print(f"Couldn't find the vehicle in frame {time}!!!")
        print('VERIFY IT MANUALLY')


    if saveVehicle:
        name = f'vehicle-crops/{str(labels[classe])}{time}.jpg'
        print(f'Saving as {name}')
        cv2.imwrite(name, crop)
        
        
    print(f'{labels[classe]} detected with confidence {conf:.2f} \n')

    return count, infos
        
        
        
def count_clusters(vector):
    vector = vector.flatten()
    
    # indices que mudam de 0 <-> 1
    changes = np.where(np.diff(vector) != 0)[0] + 1
    
    # ultimo elemento do vetor há 0 <-> 1
    if vector[-1] == 1:
        changes = np.append(changes, len(vector))
    if vector[0] == 1:
        changes = np.append(0, changes)

    indices = changes.reshape(-1, 2)
    indices[:, 1] -= 1 

    return indices



def main(video_path, line, sec, saveVR, saveVehicle, midlane):
    infos = [['label', 'frame', 'conf', 'x0', 'y0', 'x1', 'y1']]  # details of each vehicle
    count = np.array([0, 0, 0, 0, 0, 0, 0]) # count of each class
    time = 0
    
    VR_ori = []
    VR_or = []
    VR = []
    

    # load models
    model_vehicle = YOLO('./YOLO/vehicle/weights/best.pt')
    bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=1)
    reader = easyocr.Reader(['en'])


     # creates folders to save images
    if saveVehicle and not os.path.exists('vehicle-crops/'):
        os.makedirs('vehicle-crops/')
    if saveVR and not os.path.exists('VR-images/'):
        os.makedirs('VR-images/')


    # opens the video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened() == True, 'Can not open the video'


    # move to frame 1000
    cap.set(cv2.CAP_PROP_POS_FRAMES,5700)
    

    linha_or = np.zeros(bb-aa+1)

    # begins video 
    while True: 
        ret, frameRGB = cap.read() 
        if not ret:
            print('Error reading video')
            break

        # line background subtraction
        frame_bg = frameRGB[line]
        frame_bg = bg.apply(frame_bg)
        frame_bg = cv2.blur(frame_bg, (5,5))
        _, frame_bg = cv2.threshold(frame_bg, 127, 255, cv2.THRESH_BINARY)
        diff = np.dot(frame_bg,[0.114, 0.587, 0.299])

                        
        linha_or = np.logical_or(linha_or, diff)
        counts = count_clusters(linha_or)
        
        if np.any(counts):
            for l, r in counts:
                linha_xor = np.logical_xor(diff[l:r+1], linha_or[l:r+1])
                xor_sum = np.sum(linha_xor)
                
                if r-l+1 < 100: 
                    linha_or[l:r+1] = False
                elif xor_sum == r-l+1:
                    cnt, inf = counting(line, frameRGB, l, r, time, model_vehicle, saveVehicle)
                    count += cnt
                    infos += inf
                    
                    linha_or[l:r+1] = False
                    cv2.waitKey(0)
                

        # vizualizacao apenas
        # VR_ori.append(frameRGB[line, aa:bb+1])
        # VR.append(diff.astype(int)*255)
        # VR_or.append(linha_or*255)


        cv2.line(frameRGB, (aa, line), (bb, line), (0, 255, 0), 2)
        cv2.imshow('frame', cv2.resize(frameRGB, (888, 500)))
        
        time += 1
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        

    cv2.imwrite('VR_original.jpg', np.array(VR_ori))
    cv2.imwrite('VR_or.jpg', np.array(VR_or))
    cv2.imwrite('VR.jpg', np.array(VR))
    

    # final results
    print('\n -------------------------- \n')
    print_count(count)
    print(sum(count), 'vehicles counted in total')
    

    # write infos
    # with open('infos.txt', 'w') as file:
    #     for item in infos:
    #         file.write(str(item) + '\n')


    # closes the video
    cap.release()
    cv2.destroyAllWindows()
    
    
      
      
def print_count(count):
    labels = ['Bus', 'Car', 'Motorbike', 'Pickup', 'Truck', 'Van', '???'] 
    max_length = max(len(label) for label in labels)

    for label, cnt in zip(labels, count):
        print(f'{label.ljust(max_length)} : {cnt}')
      
      
      
      
if __name__ == "__main__":    

    # parse arguments
    parser = argparse.ArgumentParser(description='Count the number of vehicles in a video')
    
    parser.add_argument('--line', type=int, default=450, help='Line position')
    parser.add_argument('--interval', type=int, default=900, help='Interval between VR images (frames)')
    parser.add_argument('--save-VR', type=bool, default=False, help='Enable saving VR images')
    parser.add_argument('--save-vehicle', type=bool, default=False, help='Enable saving vehicle images')
    args = parser.parse_args()


    # video_path = input('Enter the video path: ')
    video_path = 'Set05.mp4'
    
    if video_path == 'videoplayback1.mp4':
        args.line = 450
        midlane = 0
        aa = 370    
        bb = 1200
    elif video_path == 'videoplayback2.mp4':
        args.line = 330
        midlane = 450
        aa = 160 
        bb = 850
    elif video_path =='Set05.mp4':
        args.line = 800
        midlane = 0
        aa = 0
        bb = 1919
    elif video_path == 'videoplayback.mp4':
        args.line = 500
        midlane = 545
        aa = 0
        bb = 1279

        
    # 400 1200
    print('Counting vehicles in the video...\n')
    main(video_path, args.line, args.interval, args.save_VR, args.save_vehicle, midlane)