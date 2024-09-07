# Transformo-docs
## Deployment Notes
```
apt update
```
```
apt upgrade
```
```
apt install python3.10-virtualenv
```
```
apt install cmake build-essential python3-dev
```
```
git clone https://github.com/Harshroxnox/Transformo-docs.git
cd Transformo-docs
```
```
python3 -m venv venv
```
```
pip install -r requirements.txt
```
Create a .env file with your credentials
```
nano .env
```
```
python3 login.py
```
Download the model
```
./download.sh
```
Run the rag system
```
cd rag && python3 rag.py
```











