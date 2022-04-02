import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from tool.torch_utils import *
import os

def findClosest(time, camera_time_list):
    val = min(camera_time_list, key=lambda x: abs(x - time))
    return camera_time_list.index(val)

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.8 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.2 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def custom_bbox(gt_coords, img, imgname):
    cbbox_coords = []
    for k in range(len(gt_coords)):
        if gt_coords[k][0] == imgname:
            box = [float(gt_coords[k][2]), float(gt_coords[k][3]), 50, 80]
            box = torch.tensor(box)
            bbox = box_center_to_corner(box)

            x1 = int(bbox[0].item())
            y1 = int(bbox[1].item())
            x2 = int(bbox[2].item())
            y2 = int(bbox[3].item())

            coords = [x1, y1, x2, y2]
            cbbox_coords.append(coords)
                
            # img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # print(cbbox_coords)         
    return img, cbbox_coords

def gen_images(width, height, savename, gt, file_name):
    cam1_det, cam2_det, cam3_det, cam4_det= 0, 0, 0, 0
    cam1_gt, cam2_gt, cam3_gt, cam4_gt = 0, 0, 0, 0

    patch = cv2.imread("/home/dissana8/Daedalus-physical/physical_examples/0.3 confidence__/adv_poster.png")
    print("patch read")
    resized_patch = cv2.resize(patch, (16, 16))


    # gt_actual=0
    #===== process the index files of camera 1 ======#
    with open('/home/dissana8/LAB/Visor/cam1/index.dmp') as f:
        content = f.readlines()
    cam_content = [x.strip() for x in content]
    c1_frames = []
    c1_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c1_frames.append(frame)
        c1_times.append(time)

    with open('/home/dissana8/LAB/Visor/cam2/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c2_frames = []
    c2_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c2_frames.append(frame)
        c2_times.append(time)
    

    # ===== process the index files of camera 3 ======#
    with open('/home/dissana8/LAB/Visor/cam3/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c3_frames = []
    c3_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c3_frames.append(frame)
        c3_times.append(time)
    

    # ===== process the index files of camera 4 ======#
    with open('/home/dissana8/LAB/Visor/cam4/index.dmp') as f:
        content = f.readlines()

    cam_content = [x.strip() for x in content]
    c4_frames = []
    c4_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1] + '.' + s[2])
        c4_frames.append(frame)
        c4_times.append(time)
      
    #===== process the GT annotations  =======#
    with open("/home/dissana8/LAB/"+file_name) as f:
        content = f.readlines()
        

    content = [x.strip() for x in content]
    counter = -1
    print('Extracting GT annotation ...')
    c1_frame_no, c2_frame_no, c3_frame_no, c4_frame_no = [], [], [], []
    for line in content:
        counter += 1
        # if counter % 1000 == 0:
        # print(counter)
        s = line.split(" ")
        
        time = float(s[0])
        frame_idx = findClosest(time, c1_times) # we have to map the time to frame number
        c1_frame_no.append(c1_frames[frame_idx])
        

        frame_idx = findClosest(time, c2_times)  # we have to map the time to frame number
        c2_frame_no.append(c2_frames[frame_idx])
        

        frame_idx = findClosest(time, c3_times)  # we have to map the time to frame number
        c3_frame_no.append(c3_frames[frame_idx])

        
        frame_idx = findClosest(time, c4_times)  # we have to map the time to frame number
        c4_frame_no.append(c4_frames[frame_idx])


    # print(c1_times)
    # view 01 success rate
    print("View 01 success rate")
    for ele in enumerate(c1_frame_no):
        im = "/home/dissana8/LAB/Visor/cam1/"+ele[1]
        
        img = cv2.imread(im)
        # sized = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        # print(directory)


        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)


        image, cbbox = custom_bbox(gt[0], img, imgname)
        replace = img.copy()
        if len(cbbox)>0:
            for i in range(len(cbbox)):
                x = int((cbbox[i][0]+cbbox[i][2])/2)
                y = int((cbbox[i][1]+cbbox[i][3])/2)
            #     print(x)
            #     print(y)

            #     print(replace[y-8: y +8, x-8 : x + 8].shape)
                if (y+8)>416 or (x+8)>416 or (x-8)<0 or (y-8)<0:
                    continue
                else:
                    replace[y-8: y +8, x-8 : x + 8] = resized_patch

                replace = cv2.cvtColor(replace, cv2.COLOR_RGB2BGR)
                try:
                    cv2.imwrite(sname, replace)
                except:
                    print("cannot save")
        else:
            cv2.imwrite(sname, img)
        # # break
   

#     # view 02 success rate
    print("View 02 success rate")
    for ele in enumerate(c1_frame_no):
        im = "/home/dissana8/LAB/Visor/cam2/"+ele[1]
        
        img = cv2.imread(im)
        # sized = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        # print(directory)


        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)


        image, cbbox = custom_bbox(gt[0], img, imgname)
        replace = img.copy()
        if len(cbbox)>0:
            for i in range(len(cbbox)):
                x = int((cbbox[i][0]+cbbox[i][2])/2)
                y = int((cbbox[i][1]+cbbox[i][3])/2)
            #     print(x)
            #     print(y)

            #     print(replace[y-8: y +8, x-8 : x + 8].shape)
                if (y+8)>416 or (x+8)>416 or (x-8)<0 or (y-8)<0:
                    continue
                else:
                    replace[y-8: y +8, x-8 : x + 8] = resized_patch

                replace = cv2.cvtColor(replace, cv2.COLOR_RGB2BGR)
                try:
                    cv2.imwrite(sname, replace)
                except:
                    print("cannot save")
        else:
            cv2.imwrite(sname, img)
        

# #     # view 03 success rate
    print("View 03 success rate")
    for ele in enumerate(c1_frame_no):
        im = "/home/dissana8/LAB/Visor/cam3/"+ele[1]
        
        img = cv2.imread(im)
        # sized = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        # print(directory)


        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)


        image, cbbox = custom_bbox(gt[0], img, imgname)
        replace = img.copy()
        if len(cbbox)>0:
            for i in range(len(cbbox)):
                x = int((cbbox[i][0]+cbbox[i][2])/2)
                y = int((cbbox[i][1]+cbbox[i][3])/2)
            #     print(x)
            #     print(y)

            #     print(replace[y-8: y +8, x-8 : x + 8].shape)
                if (y+8)>416 or (x+8)>416 or (x-8)<0 or (y-8)<0:
                    continue
                else:
                    replace[y-8: y +8, x-8 : x + 8] = resized_patch

                replace = cv2.cvtColor(replace, cv2.COLOR_RGB2BGR)
                try:
                    cv2.imwrite(sname, replace)
                except:
                    print("cannot save")
        else:
            cv2.imwrite(sname, img)
        
# #     # view 04 success rate
    print("View 04 success rate")
    for ele in enumerate(c1_frame_no):
        im = "/home/dissana8/LAB/Visor/cam4/"+ele[1]
        
        img = cv2.imread(im)
        # sized = cv2.resize(img, (width, height))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgfile = im.split('/')[6:]
        imgfile_ = im.split('/')[5:]

        imgname = '/'.join(imgfile)
        imgname_ = '/'.join(imgfile_)
        sname = savename + imgname_
        # imgname = '/'.join(sname)
        sname_ = sname.split('/')[:7]
        directory = '/'.join(sname_)
        # print(directory)


        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)


        image, cbbox = custom_bbox(gt[0], img, imgname)
        replace = img.copy()
        if len(cbbox)>0:
            for i in range(len(cbbox)):
                x = int((cbbox[i][0]+cbbox[i][2])/2)
                y = int((cbbox[i][1]+cbbox[i][3])/2)
            #     print(x)
            #     print(y)

            #     print(replace[y-8: y +8, x-8 : x + 8].shape)
                if (y+8)>416 or (x+8)>416 or (x-8)<0 or (y-8)<0:
                    continue
                else:
                    replace[y-8: y +8, x-8 : x + 8] = resized_patch

                replace = cv2.cvtColor(replace, cv2.COLOR_RGB2BGR)
                try:
                    cv2.imwrite(sname, replace)
                except:
                    print("cannot save")
        else:
            cv2.imwrite(sname, img)
#     tot_det = cam1_det+cam2_det+cam3_det+cam4_det
#     tot_gt = cam1_gt+cam2_gt+cam3_gt+cam4_gt

#     f = open("detections_adv.txt", "a")
#     f.write("total detections: " +str(tot_det)+"\n")
#     f.write("total gt : " +str(tot_gt)+"\n")
#     f.write("cam1 detections: " +str(cam1_det)+"\n")
#     f.write("cam1 gt: " +str(cam1_gt)+"\n")
#     f.write("cam2 detections: " +str(cam2_det)+"\n")
#     f.write("cam2 gt: " +str(cam2_gt)+"\n")
#     f.write("cam3 detections: " +str(cam3_det)+"\n")
#     f.write("cam3_gt: " +str(cam3_gt)+"\n")
#     f.write("cam4_detections: " +str(cam4_det)+"\n")
#     f.write("cam4_gt: " +str(cam4_gt)+"\n")
    
#     f.write("\n")
#     f.write("\n")
#     f.close()

#     return (tot_det/tot_gt)*100, (cam1_det/cam1_gt)*100, (cam2_det/cam2_gt)*100, (cam3_det/cam3_gt)*100, (cam4_det/cam4_gt)*100
#     # return 0, (cam1_det/cam1_gt)*100, 0, 0, 0   
# # #     # return 0, 0, 0, 0, 0


def single_image_det():
    patch = cv2.imread("/home/dissana8/Daedalus-physical/physical_examples/0.3 confidence__/adv_poster.png")
    resized_patch = cv2.resize(patch, (16, 16))
    im = "/home/dissana8/LAB/Visor/cam1/000005/005614.jpg"
    img = cv2.imread(im)
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    ## place the adversarial patch on a single image and check the detections made by yolo-v4
    ##Check the TOG attack to see how to place the patch on the image

    imgfile = im.split('/')[6:]
    imgname = '/'.join(imgfile)
    print(imgname)
    sname = savename + imgname

    # img, bbox = plot_boxes_cv2(img, boxes[0], sname, class_names)
    # print(bbox)

    image, cbbox = custom_bbox(gt[0], img, imgname)
#     print(cbbox)
#     img = cv2.rectangle(sized, (cbbox[0][0], cbbox[0][1]), (cbbox[0][2], cbbox[0][3]), (0, 0, 255), 2)
#     img = cv2.rectangle(img, (cbbox[1][0], cbbox[1][1]), (cbbox[1][2], cbbox[1][3]), (0, 0, 255), 2)
#     # print("resized patch ")
#     print(resized_patch.shape)
#     replace = sized.copy()
#     print("replace")
#     print(replace.shape)
#     for i in range(len(cbbox)):
#         x = int((cbbox[i][0]+cbbox[i][2])/2)
#         y = int((cbbox[i][1]+cbbox[i][3])/2)
#         print(x)
#         print(y)

#         if (y+8)>416 or (x+8)>416 or (x-8)<0 or (y-8)<0:
#             continue
#         else:
#             replace[y-8: y +8, x-8 : x + 8] = resized_patch

#     replace = cv2.cvtColor(replace, cv2.COLOR_RGB2BGR)
#     cv2.imwrite('boxed.png', img)
#     cv2.imwrite('replace.png', replace)


if __name__=='__main__':
    savename = '/home/dissana8/Daedalus-physical/Adv_Images/'
    file_name = 'LAB-GROUNDTRUTH.ref'

    gt = []
    gt.append(np.load('/home/dissana8/LAB/data3/LAB/cam1_coords.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data3/LAB/cam2_coords.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data3/LAB/cam3_coords.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data3/LAB/cam4_coords.npy', allow_pickle=True))

    height, width = 416, 416
    gen_images(height, width, savename, gt, file_name)
    # single_image_det()

