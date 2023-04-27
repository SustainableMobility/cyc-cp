data_path = '/home/mmvg/Desktop/mmdetection3d/data/Carla/'
data_path = r"D:\A_Monash_Cyc_Near_Miss_Detection\CARLA_Data\Carla_data_unziped"

scenes = os.listdir(data_path)
if '.DS_Store' in scenes:
    scenes.remove('.DS_Store')
if 'carla_train.json' in scenes:
    scenes.remove('carla_train.json')
total_anns = []
total_imgs = []
cate = set()
ind = 0
for scene in scenes:
    images = os.listdir(os.path.join(data_path,scene,'Images'))
    images = [os.path.join(scene,'Images',x) for x in images]
    with open(os.path.join(data_path,scene,'GT/gt.txt'),'r') as f:
        gts = f.readlines()
    for image in images:
        info = {}
        info['file_name'] = image
        image_path_split = os.path.normpath(image).split(os.path.sep)    # Split the path according to different OS
        info['id'] = '_'.join([image_path_split[0],image_path_split[2]])
        info['token'] = '_'.join([image_path_split[0],image_path_split[2]])
        info['cam2ego_rotation'] = [0.4998015430569128, -0.5030316162024876, 0.4997798114386805, -0.49737083824542755]
        info['cam2ego_translation'] = [1.70079118954, 0.0159456324149, 1.51095763913]
        info['ego2global_rotation'] = [0.5720320396729045, -0.0016977771610471074, 0.011798001930183783, -0.8201446642457809]
        info['ego2global_translation'] = [411.3039349319818, 1180.8903791765097, 0.0]
        info['cam_intrinsic'] = [[1266.417203046554, 0.0, 816.2670197447984],[0.0, 1266.417203046554, 491.50706579294757],[0.0, 0.0, 1.0]]
        info['width'] = 1600
        info['height'] = 900
        total_imgs.append(info)
    for gt in gts:
        gt_info = gt[:-1].split(',')
        image_name = "%05d.jpeg" % int(gt_info[0])
        image_name = scene+'/Images/'+image_name
        if math.sqrt(float(gt_info[13])**2+float(gt_info[14])**2) < 30:
            info={}
            info['file_name'] = image_name
            info['image_id'] = '_'.join([image_name.split('/')[0],image_name.split('/')[2]])
            info['area'] = (float(gt_info[8])-float(gt_info[6]))*(float(gt_info[9])-float(gt_info[7]))
            info['bbox'] = [float(gt_info[6]),float(gt_info[7]),float(gt_info[8])-float(gt_info[6]),float(gt_info[9])-float(gt_info[7])]
            if gt_info[2] == ' vehicle':
                info['category_name'] = 'car'
                info['category_id'] = 0
            elif gt_info[2] == ' pedestrian':
                info['category_name'] = 'pedestrian'
                info['category_id'] = 7
            info['iscrowd'] = 0
            if float(gt_info[16]) < -math.pi:
                angle = 2*math.pi+float(gt_info[16])
            elif float(gt_info[16]) > math.pi:
                angle = float(gt_info[16]) - 2*math.pi
            else:
                angle = float(gt_info[16])
            info['bbox_cam3d'] = [float(gt_info[14]),-float(gt_info[15]),float(gt_info[13]),float(gt_info[11]),float(gt_info[10]),float(gt_info[12]),angle]
            info['velo_cam3d'] = [0,0]
            info['center2d'] = [(float(gt_info[8])+float(gt_info[6]))/2, (float(gt_info[9])+float(gt_info[7]))/2, float(gt_info[13])]
            info['attribute_name'] = 'pedestrian.standing'
            info['attribute_id'] = 3
            info['segmentation'] = []
            info['id'] = ind
            ind+=1
            total_anns.append(info)
print(len(total_anns),len(total_imgs))