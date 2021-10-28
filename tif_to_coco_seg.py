from roifile import roiread
import os
import random
import numpy as np
from datetime import date
import json
import shutil
from PIL import Image
import cv2
from matplotlib.path import Path


def get_bbox(tif_file, tif_path, h, w):

    annot_info = {
        "1": [],
        "2": [],
        # "3": [],
        "P": [],
        "Cal": [],
        "M": [],
        "S": [],
        "Lym": []
    }

    for roi in roiread(f'{tif_path}/{tif_file}'):
        # if roi.name != "" and roi.name != "3" and roi.name != "P":
        if roi.name != "" and roi.name != "3":
            print("___________________________________________________")
            print(f'name = {roi.name} | number of points = {roi.n_coordinates}| type = {roi.roitype}')
            # print(roi)
            left = roi.left
            top = roi.top
            width = roi.right - roi.left
            height = roi.bottom - roi.top
            bbox = [left, top, width, height]
            area = width * height
            
            print(f'area = {area}, bbox = {bbox} type = {roi.roitype}')

            if roi.roitype == 0:
                mask = []
                seg = []
                vertex = []
                for pt in roi.integer_coordinates:
                    # print(f'{pt[0]+left}, {pt[1]+top}')
                    vertex.append((pt[0]+left, pt[1]+top))

                x,y = np.meshgrid(np.arange(w), np.arange(h))
                x,y = x.flatten(), y.flatten()
                points = np.vstack((x,y)).T
                p = Path(vertex) 
                grid = p.contains_points(points)
                mask = grid.reshape(h,w)
                # cv2.imwrite(f"mask_{roi.name}.png",mask*225)

                mask = np.array(np.where(mask==True)).T

                for pt in mask:
                    seg.append(pt[1].item())
                    seg.append(pt[0].item())
                test_mask = np.zeros((h,w))
                for i in mask:
                    test_mask[i[0]][i[1]]= 255
                # cv2.imwrite(f"mask_v2_{roi.name}.png",test_mask)

                seg_mask = grid.reshape(h,w).tolist()
            
            elif roi.roitype == 2:
                seg = []
                seg_mask = np.zeros((h,w))

                for y in range(roi.bottom-roi.top):
                    for x in range(roi.right-roi.left):
                        seg_mask[y+top][x+left]=1
                        seg.append(x+left)
                        seg.append(y+top)


                # cv2.imwrite(f"mask_v2_{roi.name}.png",seg_mask*255)

                seg_mask = seg_mask.tolist()



            if [area, bbox, seg, roi.name, seg_mask] not in annot_info[roi.name]:
                annot_info[roi.name].append([area, bbox, seg, roi.name, seg_mask])

    # if len(annot_info["Lym"]) >= 1 and len(annot_info["P"]) >= 1:
    #     for ann in annot_info["Lym"]:
    #         # print(ann)
    #         lym_mask = np.array(ann[4])
    #         # print(lym_mask)
    #         print(annot_info["P"][0][1])
    #         annot_info["P"][0][4] = (annot_info["P"][0][4]*1)-(lym_mask*1)
    #         # quit()
    #     seg = []
    #     mask = np.array(np.where(mask==1)).T

    #     for pt in mask:
    #         seg.append(pt[1].item())
    #         seg.append(pt[0].item())

    #     annot_info["P"][0][2] = seg

    return annot_info


def create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def split_dataset():
    path = os.path.dirname(os.path.realpath(__file__))
    tif_path = f'{path}/tif'
    img = []
    
    for tif in os.listdir(tif_path):
        if not os.path.exists(f'{tif_path}/{tif.split(".")[0]}_1.tif'):  
            img.append(tif)

    random.shuffle(img)

    #80 20
    train, test = np.split(np.array(img),[int(len(img)*(.80))])
    #60 20
    train, val = np.split(np.array(train),[int(len(train)*(.75))])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(len(train), len(val), len(test))

    to_coco_notation(train, "train")
    to_coco_notation(val, "val")
    to_coco_notation(test, "test")


def make_img(tif_file, coco_dir, used_for):
    path = os.path.dirname(os.path.realpath(__file__))
    tif_path = f'{path}/tif'
    png_name = tif_file.split(".")[0]

    img = Image.open(f'{tif_path}/{tif_file}')

    if img.mode == 'I;16B':
        img.mode = 'I'
    pixels = np.array(img.getdata()).reshape((img.size[1], img.size[0]))    
    dtype = {'F': np.float32, 'I': np.uint32}[img.mode]
    np_img = np.array(pixels, dtype=dtype)
 
    max_val = np_img.max()
    min_val = np_img.min()
    norm = ((pixels - min_val)*255 / (max_val-min_val))

    if img.mode == 'F':
        norm = norm.max() - norm

    print("saving image at", f'{coco_dir}/{used_for}/{png_name}.png')
    img = Image.fromarray(norm).convert('RGB')
    img.save(f'{coco_dir}/{used_for}/{png_name}.png')

    return norm.shape[0], norm.shape[1], norm


def iscrowd(info, w, h):
    mask = np.zeros((w,h))
    for annot in info:
        np_mask = np.array(annot[4])
        mask += np_mask

    mask[mask>0] = 1
    # cv2.imwrite("test.jpg",mask*255)
    pos = np.where(mask)
    xmin = np.min(pos[1]).item()
    xmax = np.max(pos[1]).item()
    ymin = np.min(pos[0]).item()
    ymax = np.max(pos[0]).item()
    width = xmax - xmin
    height = ymax - ymin
    
    bbox = [xmin, ymin, width, height]
    area = width * height
    print(bbox)
    return rle(mask), [xmin, ymin, width, height], width * height


#https://stackoverflow.com/questions/49494337/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def rle(mask):
    seg = {'counts': [], 'size': list(mask.shape)}
    counts = seg.get('counts')

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(mask.ravel(order='F')):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return seg


def to_coco_notation(img, used_for):
    coco_dataset = {
        "info": {
                "description": "COCO Style Breast X ray Dataset",
                "version": "1.0",
                "year": 2021,
                "contributor": "YRLee",
                "date_created": date.today().strftime('%Y/%m/%d'),
                "url": "N/A"
                },
        "licenses": [
            {
                "url": "N/A",
                "id": 0,
                "name": "N/A"
            }
                    ],
        "images": [],
        "annotations": [],
        "categories": [
            {"supercategory": "breast", "id": 1, "name": "MG"},
            {"supercategory": "breast", "id": 2, "name": "nipple"},
            # {"supercategory": "breast", "id": 3, "name": "pnl"},
            {"supercategory": "breast", "id": 4, "name": "P"},
            {"supercategory": "breast", "id": 5, "name": "Cal"},
            {"supercategory": "breast", "id": 6, "name": "M"},
            {"supercategory": "breast", "id": 7, "name": "S"},
            {"supercategory": "breast", "id": 8, "name": "Lym"}
                    ]
    }
    cat = {
        "1": 1,       
        "2": 2,       
        "3": 3,       
        "P": 4,     
        "Cal": 5,   
        "M": 6,     
        "S": 7,     
        "Lym": 8
        } 

    path = os.path.dirname(os.path.realpath(__file__))
    coco_dir = path+"/coco"

    create_dir(f'{coco_dir}/{used_for}')
    
    multi_mask = []
    img_id = 1
    annot_id = (len(img)*2)+1
    total = len(img)
    file_num = 1


    # img = ["36570_1004_Lcc", "97288_1909_Lmlo_1"]
    for tif_file in img:
        # tif_file = "186311_2011_Rcc_1.tif"
        filename = tif_file.split('/')[-1].split(".")[0]
        print(f'\n{file_num}/{total} of {used_for} working on: {filename}')
        file_num += 1
        w, h, img_arr = make_img(tif_file, coco_dir, used_for)
        
        image = {
            "license": 0,
            "file_name": f'{filename}.png',
            "id": img_id,
            "height": h,
            "width": w,
        }
        coco_dataset["images"].append(image)
 
        tif_path = f'{os.path.dirname(os.path.realpath(__file__))}/tif'
        
        annot_info = get_bbox(tif_file, tif_path, w, h)
    
        for key, info in annot_info.items():
            print("------------------------")
            print(key, len(info))
            if len(info) == 1:
                print(f'area = {info[0][0]}, bbox = {info[0][1]}, class = {cat[info[0][3]]}')
                # test_mask_v2 = np.zeros((w,h))
                # seg_arr = info[0][2]
                # for i in range(0,len(seg_arr),2):
                #     test_mask_v2[seg_arr[i+1]][seg_arr[i]]= 255
                # for i in range(0, info[0][1][3]):
                #     test_mask_v2[i+info[0][1][1]][info[0][1][0]] = 255
                #     test_mask_v2[i+info[0][1][1]][info[0][1][0]+info[0][1][2]] = 255
                # for i in range(0, info[0][1][2]):
                #     test_mask_v2[info[0][1][1]][i+info[0][1][0]] = 255
                #     test_mask_v2[info[0][1][1]+info[0][1][3]][i+info[0][1][0]] = 255
                # cv2.imwrite(f"seg_in_annot_{cat[info[0][3]]}.png",test_mask_v2)
                
                annotation = {
                    "segmentation": [info[0][2]],                    
                    "area": info[0][0],
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": info[0][1],
                    "category_id": cat[info[0][3]],   
                    "id": annot_id
                }

                coco_dataset["annotations"].append(annotation)
                annot_id+=1

            elif len(info) > 1:
                seg, bbox, area = iscrowd(info, w, h)
                annotation = {
                "segmentation": seg,
                "area": area,
                "iscrowd": 1,
                "image_id": img_id,
                "bbox": bbox,
                "category_id": cat[info[0][3]],   
                "id": annot_id
                }

                coco_dataset["annotations"].append(annotation)
                annot_id+=1
                
        img_id+=1

    with open(f'{coco_dir}/{used_for}/annotation_coco.json', 'w') as f:
        json.dump(coco_dataset, f, ensure_ascii=False, indent=4)
    # quit()


if __name__ == '__main__':
    split_dataset()