### 1. 환경 만들기

- conda create -n upbit_auto_trade python=3.7
- conda activate upbit_auto_trade
- cd ~/git/upbit_auto_trade/
- pip install --ignore-installed pip
- pip freeze > requirements.txt
- pip install -r requirements.txt

### 6. gitignore 적용

- git rm -r --cached .
- git add .
- git commit -m "Apply .gitignore"