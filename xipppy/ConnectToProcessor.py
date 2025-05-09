#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo of connecting to processor either through UDP or TCP

@author: richard
"""

import xipppy as xp
from time import sleep

def connectToProcessor():
    
    # Connect to processor (Try UDP then TCP) and then print the processor time.
    try:
        with xp.xipppy_open():
            print(xp.time())
    except:
        try:
            with xp.xipppy_open(use_tcp=True):
                print(xp.time())
        except:
            print("Failed to connect to processor.")
        else:
            print("Connected over TCP")
    else:
        print("Connected over UDP")
    
    sleep(0.001)
    
if __name__ == '__main__':
    connectToProcessor()