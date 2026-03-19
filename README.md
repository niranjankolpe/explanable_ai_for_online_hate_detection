# Explanable AI for Online Hate Detection

This project utilizes Explainable NLP Framework for Cyberbullying and Online Harassment Detection Across Social Media Platforms.

### Pre-installation System Requirements:
(Assuming Windows 11 OS)
1. PC/Laptop with 8 GB or more RAM
2. Python version: 3.14.2
3. Git version (for windows): 2.52.0.windows.1
4. DVC version: 3.65.0

### Project Installation
1. Git pull the project to your local repository.
2. Open CMD in root folder and create a virtual environment. Run "pip install -r requirements.txt --force-reinstall --no-cache-dir" to install required dependencies inside an activated virtual environment.
3. In Git Bash run "dvc remote add -d storage C:/dvc-storage --local".
4. Next, run "dvc repro" if installing on new machine with no dvc local remote storage configured earlier OR run "dvc pull" if dvc local remote was configured earlier on same machine.

### Project Run/ Execution
1. Open CMD in root folder with virtual environment activated and run "uvicorn src.api:app --reload". Visit "http://127.0.0.1:8000/docs".
2. In the POST/predict, click "Try it out" and enter the sample input for the text.
- "you are idiot" (shall return OFF classification)
- "you are nice"  (shall return NOT classification)
- "nice work bro" (shall return NOT classification)

### Instructions for Docker
The requirements-docker.txt file is a copy of requirements.txt file with OS based dependenceis(like pywin32 for windows) removed. Remove any such OS based dependency that may cause issue during container creation. Utilize requirements-docker.txt for docker containers and requirements.txt for all other tasks.