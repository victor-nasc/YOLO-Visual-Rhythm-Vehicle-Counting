
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

# video_path = 'videoplayback2.mp4'
# 1977 perde um carro na msm mancha que um caminhao



# TESTAR EM OUTROS VIDEOS + DEFINIR LINHA


import cv2
import numpy as np
from ultralytics import YOLO

import argparse
import os




def counting(line, frame, l, r, time, model_vehicle, saveVehicle, midlane):
    labels = ['Bus', 'Car', 'Motorbike', 'Pickup', 'Truck', 'Van', '???'] 
    delta = 15
    count = np.array([0, 0, 0, 0, 0, 0, 0])
    infos = []
    height , width, _ = np.shape(frame)


    # detect vehicles
    vehicles = model_vehicle(frame, iou=0.5, conf=0.3, verbose=False, save=True, agnostic_nms=True)
    vehicles = vehicles[0].numpy()
    
    
    # extend the mark's x-coordinates
    l = max(l - delta, 0) + aa
    r = min(r + delta, width) + aa
    
    
    # find the vehicle associated with the mark
    dist_min = height
    crop = frame
    classe = conf = -1
    for cls, xyxy, cnf in zip(vehicles.boxes.cls.astype(int), vehicles.boxes.xyxy, vehicles.boxes.conf):
        x0, y0, x1, y1 = np.floor(xyxy).astype(int)
        mid_x = (x0 + x1) // 2
        mid_lr = (l + r) // 2
        y = y0 if mid_x < midlane else y1
        print(y0, y1, y,'-', mid_x, x0, x1, l, r, '-', x0 > l, x1 < r, l < mid_x < r, abs(y-line) < 90, labels[cls])
        if y0+delta < line < y1-delta:
            continue
        
        if l <= mid_x <= r and abs(y-line) < 90:
            count[cls] += 1
            infos.append([labels[classe], frame, conf, x0, y0, x1, y1])
            # print(f'{labels[cls]} detected in frame {time, cnf}\n') 
            # dist = abs(y0 - line)
            # if dist < dist_min:
            #     dist_min = dist
            #     classe = int(cls)
            #     conf = cnf
            if saveVehicle:
                crop = frame[y0:y1 , x0:x1]
    

    if np.sum(count) == 0:
        print(f"Couldn't find the vehicle in frame {time}!!!")
        print('VERIFY IT MANUALLY')


    if saveVehicle:
        name = f'vehicle-crops/{str(labels[classe])}{time}.jpg'
        print(f'Saving as {name}')
        cv2.imwrite(name, crop)
        
        
    # update infos
    for i in range(len(count)):
        if count[i] > 0:
            print(f'{count[i]} {labels[i]} detected in frame {time}\n')

    # print(f'{labels[classe]} detected with confidence {conf:.2f} \n')

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
    atu = 0     # current VR initial frame
    double = [] # marks on the border of the previous VR
    infos = [['label', 'frame', 'conf', 'x0', 'y0', 'x1', 'y1']]  # details of each vehicle
    VR = []     # VR image
    count = np.array([0, 0, 0, 0, 0, 0, 0]) # count of each class
    time = 0
    
    VR_or = []
    VR = []
    VR_cor = []
    
    # load models
    model_mark = YOLO('./YOLO/marks/weights/best.pt')
    model_vehicle = YOLO('./YOLO/vehicle/weights/best.pt')


    # opens the video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened() == True, 'Can not open the video'


    # creates folders to save images
    if saveVehicle and not os.path.exists('vehicle-crops/'):
        os.makedirs('vehicle-crops/')
    if saveVR and not os.path.exists('VR-images/'):
        os.makedirs('VR-images/')
        
        
        
        
# 1400 carro na mask pb2
# 5400 glitch truck pb2
# 8800 carro mundando faixa pb2


    # 1000 van + carro colados pb2 perde o carro
    # 1800 caminha + carro colados pb2 perde o carro
    # 8500 carro na mask + colado com outro pb2 perdeu o carro
    # 9050 carro colado com caminhao pb2 perdeu o carro
    # 10000 carro mudando faixa pb2 perdeu o carro
    # 12600 carro mudando faixa pb2 perdeu o carro
    # 15350 carro colado com van pb2
    # 15500 carro colado com outro (contou ambos) pb2
    # 16000 contaagem dupla misteriosa pb2
    # 18500 busao gigante pb2
    # 20400 carro mudando de faixa pb2
    cap.set(cv2.CAP_PROP_POS_FRAMES,1800)
    
    
    
    
    ret , atu = cap.read()
    if not ret:
        print('Error reading video')
        return
    
    mask = np.ones(bb-aa+1, dtype=np.uint8)
    mask[290-aa:330-aa] = 0
    mask[430-aa:480-aa] = 0
    mask[580-aa:610-aa] = 0
    mask[710-aa:740-aa] = 0
    
    atu = cv2.cvtColor(atu, cv2.COLOR_BGR2GRAY)
    atu = atu[line, aa:bb+1]
    atu = atu * mask
    
    linha_or = np.zeros((bb-aa+1, 1))

    # begins video 
    while True: 
        ret, frameRGB = cap.read() 
        if not ret:
            print('Error reading video')
            break

        
        frame = cv2.cvtColor(frameRGB, cv2.COLOR_BGR2GRAY)
        prox = frame[line, aa:bb+1]  
        prox = prox * mask


        diff = cv2.absdiff(atu, prox)
        diff = cv2.blur(diff, (1,5))
        _, diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)
                
        
        linha_or = np.logical_or(linha_or, diff)
        counts = count_clusters(linha_or)
        
        if np.any(counts):
            for l, r in counts:
                linha_xor = np.logical_xor(diff[l:r+1], linha_or[l:r+1])
                xor_sum = np.sum(linha_xor)
                
                if r-l+1 < 40: 
                    linha_or[l:r+1] = False
                elif xor_sum == r-l+1:
                    print(l, r, xor_sum, r-l+1)
                    # crop = frameRGB[line-100:line, aa+l:aa+r+1]
                    # cv2.imshow('crop', crop)
                    cnt, inf = counting(line, frameRGB, l, r, time, model_vehicle, saveVehicle, midlane)
                    wait = cv2.waitKey(0)
                    
                    count += cnt
                    infos += inf
                    
                    linha_or[l:r+1] = False
                

        # vizualizacao apenas
        VR_cor.append(prox)
        VR.append(diff)
        VR_or.append(linha_or*255)


        cv2.line(frameRGB, (aa, line), (bb, line), (0, 255, 0), 2)
        cv2.imshow('frame', cv2.resize(frameRGB, (888, 500)))
        # cv2.imshow('frame', frameRGB)

        counts = []
        atu = prox
        
        time += 1
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        


    cv2.imwrite('VR_or.jpg', np.array(VR_or))
    cv2.imwrite('VR.jpg', np.array(VR))
    cv2.imwrite('VR_cor.jpg', np.array(VR_cor))
    
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
    video_path = 'videoplayback2.mp4'
    
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
    elif video_path =='4k.mp4':
        args.line = 680
        midlane = 900
        aa = 300
        bb = 1700
    elif video_path == 'videoplayback.mp4':
        args.line = 500
        midlane = 545
        aa = 0
        bb = 1279
    else:
        args.line = 1000
        midlane = 0
        aa = 0
        bb = 1919

        
    # 400 1200
    print('Counting vehicles in the video...\n')
    main(video_path, args.line, args.interval, args.save_VR, args.save_vehicle, midlane)
