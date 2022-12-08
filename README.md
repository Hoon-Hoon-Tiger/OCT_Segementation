# 💡 Segmentation layer of retina and the choroid in OCT images

## __Purpose__   
Background, 망막, 맥락막을 포함하고 있는 __11classes Segmentation__

## Requirements
- Python 3.6.13
- PyTorch 1.10.2

## Data Curation

  ### Dataset Distribution
  
  - __11 classes__   
     - 망막의 경우 __총 12개의 layer로 구성__ 
     - 막 구조의 __ELM(External Limiting Membrane) 제외__   
     - MZ(Myoid Zone), IZ(Interdigitation Zone)의 경우 층이 매우 얇기 때문에 각각 주변층과 하나의 Class로 통합   
     - Choroid layer추가   
  
  - __Dataset__  
     
     - 각 Volume에 대해서 총 21 Slices이 존재
          - Volume : 각 환자의 OD(right eye), OS(left eye)에 대한 OCT Scan Data List 
          - 황반의 중심을 지나는 11 slice를 기준으로 +-3mm 영역을 관찰하기 위해 5 slice ~ 17 Slice로 총 13개의 B-scan Images이 저장
     
     - 현재 633 Volumes이 존재 >> Total 8229(633*13) Images 존재

  ### __Image Processing & Cropping__
  
  1. Labeling 과정에서 맥락막과 배경 사이에 새로운 Layer가 잘 못 발생하여 수정
     ![image](https://user-images.githubusercontent.com/97836929/206370158-cc26dd5a-b946-41e8-b87e-029d1a078917.png)
  
  2. Mask 이미지에 다른 Class로 Labeling 되어있는 부분과 검은색 Hole들 제거
  ![image](https://user-images.githubusercontent.com/97836929/206370652-0e22e9c2-fe63-4655-b476-2f8438bbe392.png)
  
  3. 각 Mask Image에서 Start, Finish index를 찾은 후 Image와 Mask를 Cropping
  ![image](https://user-images.githubusercontent.com/97836929/206371145-ef51673f-64ad-4c84-b65e-8ab946d527d5.png)

## __DataSplit__
- 각 중증도 별로 Patient를 기준으로 8:1:1 비율로 Split

## __Study Design__

  ### __Pilot Study__
  - __Model : Unet (pretrained model)__
  - __Augmentation(Image Augmentation Library, albumnetation 이용)__
     - Resize, RandomScale, RandomHorizontalFlip, CLAHE
  - Epoch : 50
  - Optimizer : SGD
  - Loss Functioin : CE(Cross Entropy)
  - Batch Size : 8
  - Learning Rate : 0.001
  
