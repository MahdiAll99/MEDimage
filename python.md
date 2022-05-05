# How to install python 3.8 or more

This document provides instructions on how to install Python on  different operating systems.

## Table of Contents
- [Windows users](#windows-users)
- [Linux users](#linux-users)
- [MAC OS users](#mac-os-users)
- [References](#references)

## Windows users

First, go the Python official [download website](https://www.python.org/downloads/).

![alt](https://csharpcorner-mindcrackerinc.netdna-ssl.com/article/how-to-install-python-3-8-in-windows/Images/How%20To%20Install%20Python%20on%20Windows02.png)

Click downloads and click on the latest version then your download will start and you should have something like this  

![alt](https://csharpcorner-mindcrackerinc.netdna-ssl.com/article/how-to-install-python-3-8-in-windows/Images/How%20To%20Install%20Python%20on%20Windows03.png)

Next, open the download file and enable *Add Python to Path* option and click *Install now* as shown below

![alt](https://csharpcorner-mindcrackerinc.netdna-ssl.com/article/how-to-install-python-3-8-in-windows/Images/How%20To%20Install%20Python%20on%20Windows05.png)

Wait a few minutes until it displays *setup was successful* and click the close button

![alt](https://csharpcorner-mindcrackerinc.netdna-ssl.com/article/how-to-install-python-3-8-in-windows/Images/How%20To%20Install%20Python%20on%20Windows06.png)

Verify that the installation was successful by typing
```
python3.8 --version
```
```
Output

Python 3.8.0
```

## Linux users

To installing Python on Linux we will only need *apt*command, so make sure to run all the commands as root or as user with sudo access.

First, we update the packages list and install the prerequisites using these commands
```
sudo apt update
```
```
sudo apt install software-properties-common
```

Second, add the [deadsnakes](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) Personal Package Archives (PPA) to your system’s sources list
```
sudo add-apt-repository ppa:deadsnakes/ppa
```
When prompted (see below) press `Enter` to continue
```
Output

Press [ENTER] to continue or Ctrl-c to cancel adding it.
```

Once the PPA is added, install Python 3.8 with
```
sudo apt install python3.8
```
**PS**: You can install later versions of python. All you need is change the version on the last command and make sure it's available on the [deadsnakes packages](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa/+packages).

Verify that the installation was successful by typing
```
python3.8 --version
```
```
Output

Python 3.8.0
```
## MAC OS users

We will use the official installer from the [Python website](https://www.python.org/) which is the most common way for installing Python on MAC.

First, from the [Python downloads website](https://www.python.org/downloads/) download the latest version of Python installer on your Mac by clicking on the download link

![alt](https://www.dataquest.io/wp-content/uploads/2022/01/installing-python-on-mac-screenshot-s-1024x578.webp)

Once the download is complete, double-click the package to start installing Python using the default settings

![alt](https://www.dataquest.io/wp-content/uploads/2022/01/installing-python-on-mac-screenshot-r-1024x778.webp)

When the installation completes, it will open up the Python folder

![alt](https://www.dataquest.io/wp-content/uploads/2022/01/installing-python-on-mac-screenshot-q-1024x561.webp)

Let’s verify that Python and IDLE installed correctly. To do that, double-click IDLE, which is the integrated development environment shipped with Python. If everything works correctly, IDLE shows the Python shell as follows

![alt](https://www.dataquest.io/wp-content/uploads/2022/01/installing-python-on-mac-screenshot-p-1024x728.webp)

## References

All the pictures used in the this document are extracted from the following links:

- [How To Install Python 3.8 On Windows](https://www.c-sharpcorner.com/article/how-to-install-python-3-8-in-windows/).
- [Tutorial: Installing Python on Mac](https://www.dataquest.io/blog/installing-python-on-mac/).
