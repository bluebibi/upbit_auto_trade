### 1. 환경 만들기

- conda create -n upbit_auto_trade python=3.7
- conda activate upbit_auto_trade
- cd ~/git/upbit_auto_trade/
- pip install --ignore-installed pip
- pip install -r requirements.txt

### 2. requirements.txt 구성 

- pip freeze > requirements.txt

### 3. gitignore 적용

- git rm -r --cached .
- git add .
- git commit -m "Apply .gitignore"

### 4. Output 설명 

- BTC_0.0944_451_36_0.0000.pt/png
  - 0.9444: validation_loss
  - 451: saved epoch
  - 36: validation data size
  - 0.0000: rate of one in validation data

