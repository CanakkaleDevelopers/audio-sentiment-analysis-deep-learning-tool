import urllib.request
import platform
import os
import art


def check_internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)  # Python 3.x
        return True
    except:
        return False

def return_os_info():
    os = platform.system()
    rel = platform.release()
    ver = platform.version()

    return os, rel, ver

def check_directories():
    print("Checking directories..")
    files = ['TEMP','Downloads', 'Datasets']
    for file in files:
        if not os.path.exists(file):
            pass
            #print('Creating {} folder'.format(file))
           # os.makedirs(file)

def welcome():
    print("                        "+"Welcome to " + art.art('hello'))
    art.tprint('DEPSEMO V1')
    print("                      "+art.art('line brack'))

    print("Check")
    print("--Checking Internet Connection --")
    if check_internet_connection():
        print("Internet is working.. good..")
    else:
        print("Can you check your internet connection?")
        print("You cant download dataset. Offline")

    os,rel,ver = return_os_info()
    print("--System--")
    print(os,rel,ver)
    print('-- Checking directories --')
    check_directories()
    art.tprint('System check complete!',font='minion')



welcome()