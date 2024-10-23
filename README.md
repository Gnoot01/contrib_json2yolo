This update to [Ultralytics' JSON2YOLO converter](https://github.com/ultralytics/JSON2YOLO), which initially supported only segmentation tasks, builds upon [Ryouchinsa's converter](https://github.com/ryouchinsa/Rectlabel-support/blob/master/general_json2yolo.py) and contributions from other developers.

While Ryouchinsa’s version enabled pose detection, it was limited to 17 keypoints. This update now allows for the conversion of COCO WholeBody .json files, which contain 133 keypoints, into YOLO format, for more comprehensive pose detection and further processing. It may be adjusted for variable keypoints as well

```
def set_coco_keypoints(use_keypoints, w, h, ann, box, keypoints, type="default"):
    if not use_keypoints:
        return
    if 'keypoints' not in ann:
        keypoints.append([])
        return
    if len(ann['keypoints']) == 0:
        keypoints.append([])
        return
    else:
        if type == "default":
            k = (np.array(ann['keypoints']).reshape(-1, 3) /
                 np.array([w, h, 1])).reshape(-1).tolist()
            k = box + k
            keypoints.append(k)
        elif type == "foot":
            k = (np.array(ann['foot_kpts']).reshape(-1, 3) /
                 np.array([w, h, 1])).reshape(-1).tolist()
            keypoints.append(k)
        elif type == "face":
            k = (np.array(ann['face_kpts']).reshape(-1, 3) /
                 np.array([w, h, 1])).reshape(-1).tolist()
            keypoints.append(k)
        elif type == "lefthand":
            k = (np.array(ann['lefthand_kpts']).reshape(-1, 3) /
                 np.array([w, h, 1])).reshape(-1).tolist()
            keypoints.append(k)
        elif type == "righthand":
            k = (np.array(ann['righthand_kpts']).reshape(-1, 3) /
                 np.array([w, h, 1])).reshape(-1).tolist()
            keypoints.append(k)
```

Example usage:
- [COCO .json file](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EQuxJ51ZSXVPv6EeGnLT65YBvkaVQLAMRYW6pnk6sobfPA?e=jjV2u4)

- ![image](https://github.com/user-attachments/assets/b7c07cb0-ae87-41f4-9fdc-f5e7db53de00)
- Converted YOLO .txt file

Each `.txt` annotation file in YOLO format should contains 404 values, structured as follows:

|    1     |     2     |    3     |     4     |     5     |                 Keypoints (133 x 3)                  |
|----------|-----------|----------|-----------|-----------|------------------------------------------------------|
| class-id | x_center  | y_center | width_bb  | height_bb | <x_p1> <y_p1> <v_p1> <x_p2> <y_p2> <v_p2> ... <x_p133> <y_p133> <v_p133> |

```
0 0.609594 0.513106 0.341719 0.815718 0.573438 0.190588 2 0.584375 0.171765 2 0.5625 0.176471 2 0.603125 0.183529 2 0.55625 0.190588 2 0.623437 0.254118 2 0.559375 0.303529 2 0.676562 0.334118 2 0.532813 0.374118 2 0.701562 0.388235 2 0.482812 0.418824 2 0.6625 0.477647 2 0.614062 0.503529 2 0.670312 0.691765 2 0.573438 0.642353 2 0.728125 0.851765 2 0.61875 0.802353 2 0.685937 0.889412 2 0.696875 0.894118 2 0.748437 0.870588 2 0.589063 0.844706 2 0.5875 0.842353 2 0.645312 0.830588 2 0.555973 0.177318 1 0.556803 0.186079 1 0.558194 0.194676 1 0.560184 0.202993 1 0.562728 0.21096 1 0.566041 0.218244 1 0.570319 0.224288 1 0.575761 0.227401 1 0.581548 0.226216 1 0.586944 0.222784 1 0.591564 0.217333 1 0.59547 0.210744 1 0.598286 0.202998 1 0.599838 0.194473 1 0.600117 0.18565 1 0.600156 0.176802 1 0.599848 0.167967 1 0.557944 0.171704 1 0.560316 0.168905 1 0.563156 0.167869 1 0.566214 0.167427 1 0.569312 0.167503 1 0.576517 0.16588 1 0.579602 0.164277 1 0.582814 0.163187 1 0.586156 0.16298 1 0.589622 0.164439 1 0.572766 0.174021 1 0.573022 0.179176 1 0.57325 0.18424 1 0.573422 0.189402 1 0.570475 0.194256 1 0.572356 0.194901 1 0.574445 0.194832 1 0.576831 0.193426 1 0.579208 0.192123 1 0.560477 0.177539 1 0.562959 0.17433 1 0.566213 0.173747 1 0.569222 0.175726 1 0.566473 0.177807 1 0.563456 0.178561 1 0.577767 0.173584 1 0.580327 0.170032 1 0.583773 0.169295 1 0.587066 0.171093 1 0.5842 0.173409 1 0.580992 0.174174 1 0.567475 0.204545 1 0.569981 0.201055 1 0.57347 0.199406 1 0.575075 0.198909 1 0.576692 0.19852 1 0.581536 0.197772 1 0.586067 0.199741 1 0.583886 0.204879 1 0.580694 0.20897 1 0.5766 0.211055 1 0.572922 0.210833 1 0.569877 0.208485 1 0.568106 0.204478 1 0.571442 0.201999 1 0.575288 0.200674 1 0.580381 0.199599 1 0.585386 0.200107 1 0.581509 0.205392 1 0.576266 0.207951 1 0.57182 0.207821 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.475162 0.42765 1 0.469847 0.43006 1 0.464531 0.432471 1 0.460469 0.438824 1 0.453281 0.441882 1 0.454531 0.415529 1 0.449219 0.414118 1 0.446094 0.419765 1 0.4475 0.429176 1 0.45125 0.422118 1 0.445312 0.425882 1 0.448906 0.437882 1 0.455937 0.445882 1 0.449531 0.429882 1 0.443438 0.433176 1 0.447656 0.444941 1 0.453125 0.451765 1 0.447969 0.436 1 0.441875 0.440941 1 0.445 0.450824 1 0.450625 0.457647 1

```
