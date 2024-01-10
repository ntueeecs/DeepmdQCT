# DeepmdQCT

## Environment configuration:
* Python3.6/3.7/3.8
* Python 1.10 or higher
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`)
* Ubuntu or Centos (Windows not recommended)
* It is best to use GPU training
* Detailed environmental configuration`requirements.txt`

## File Structure：
```
  ├── model: DeepmdQCT network structure
  ├── my_dataset_coco.py: Custom dataset for reading datasets
  ├── person_keypoints.json: Information related to key points in the dataset
  ├── predict.py: Simple prediction script, using trained weights for prediction
  └── transforms.py: Data augmentation related
```
weight:https://1drv.ms/u/s!Auy8I-BCHDaJgRu5EgZjYlRf4PLY?e=46cLuw

data
```
├── data: Dataset Root Directory
     ├── images
     ├── images_val
     ├── labels 
     ├── labels_segement
     ├── labels_val
     ├── train.json
     └── val.json
```
class.xlsx: Category, Clinical Data, Domain Category






