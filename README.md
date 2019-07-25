### 1. 환경 만들기
# https://pytorch.org/
- conda create -n upbit_auto_trade python=3.7
- conda activate upbit_auto_trade
- cd ~/git/upbit_auto_trade/
- pip install --ignore-installed pip
- pip install -r requirements.txt
- conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

### 2. requirements.txt 구성 

- pip freeze > requirements.txt

### 3. gitignore 적용

- git rm -r --cached .
- git add .
- git commit -m "Apply .gitignore"

### 4. Output 설명 

- BTC_3_0.2564_92.8571_42_0.0476.pt/png
  - 3: saved_epoch
  - 0.2564: validation_loss_min
  - 92.8571: validation_accuracy
  - 42: validation data size
  - 0.0476:rate of one in validation data

