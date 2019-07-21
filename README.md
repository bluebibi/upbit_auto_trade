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

