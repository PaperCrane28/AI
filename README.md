# Face mask detection using YOLOv8

![image](https://github.com/user-attachments/assets/2066b3a2-d88d-4aa3-856e-d04ef714fbf6)

## Importing Libraries

```Jupyter Notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import os
import cv2
sns.set_theme('talk')
```

## Define paths for images and annotations
```Jupyter Notebook
images_path = r"/kaggle/input/face-mask-detection/images/"

annotations_path = r"/kaggle/input/face-mask-detection/annotations/"
```
```
image_names = os.listdir(images_path)

image_names[:5]
```
```
annotation_names = os.listdir(annotations_path)

annotation_names[:5]
```
```
len(annotation_names) == len(image_names)
```
```
images_data = pd.DataFrame(pd.Series([images_path + i for i in image_names], name='path'))

images_data.head()
```
```
images_data['path'].iloc[0]
```
```
images_data['id'] = images_data['path'].apply(lambda x: int(x.split('/')[-1].split('.')[0].removeprefix('maksssksksss')))
```

## DataFrame creation
```Jupyter Notebook
images_data = images_data[['id', 'path']]

images_data.head()
```
```
images_data.info()
```

## Parsing annotaions using ElementTree
```Jupyter Notebook
import xml.etree.ElementTree as ET
import pandas as pd

def get_xml_data(path):
    tree = ET.parse(path)
    root = tree.getroot()
    id = int(root[1].text.split('.')[0].removeprefix('maksssksksss')) # you can use regular expressions to extract the id though
    width = int(root[2][0].text)
    height = int(root[2][1].text)
    depth = int(root[2][2].text)
    segmented = int(root[3].text)

    records = []

    for i in root[4:]:
        record = {
            'file_id': id,
            'width': width,
            'height': height,
            'depth': depth,
            'segmented': segmented,
            'class': i[0].text,
            'pose': i[1].text,
            'truncated': int(i[2].text),
            'occluded': int(i[3].text),
            'difficult': int(i[4].text),
            'xmin': int(i[5][0].text),
            'ymin': int(i[5][1].text),
            'xmax': int(i[5][2].text),
            'ymax': int(i[5][3].text)
        }
        records.append(record)

    return records
```
```Jupyter Notebook
annotations_data = []
for i in annotation_names:
    annotations_data.extend(get_xml_data(annotations_path + i))

annotations_data = pd.DataFrame(annotations_data)
```
```Jupyter Notebook
annotations_data
```
## Reference
https://www.kaggle.com/code/abdelrhmankaram/face-mask-detection-using-yolov8/notebook
