# Face mask detection using YOLOv8

![image](https://github.com/user-attachments/assets/2066b3a2-d88d-4aa3-856e-d04ef714fbf6)

## Importing Libraries

```python
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
```python
images_path = r"/kaggle/input/face-mask-detection/images/"

annotations_path = r"/kaggle/input/face-mask-detection/annotations/"

image_names = os.listdir(images_path)

image_names[:5]

annotation_names = os.listdir(annotations_path)

annotation_names[:5]

len(annotation_names) == len(image_names)

images_data = pd.DataFrame(pd.Series([images_path + i for i in image_names], name='path'))

images_data.head()

images_data['path'].iloc[0]

images_data['id'] = images_data['path'].apply(lambda x: int(x.split('/')[-1].split('.')[0].removeprefix('maksssksksss')))
```

## DataFrame creation
```Jupyter Notebook
images_data = images_data[['id', 'path']]

images_data.head()

images_data.info()
```
## Reference
https://www.kaggle.com/code/abdelrhmankaram/face-mask-detection-using-yolov8/notebook
