import subprocess

for lettre in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    subprocess.run(["python", "scraper.py", lettre], check=True)
    
subprocess.run(["python", "cleaner.py"], check=True)
