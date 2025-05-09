#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example script for sending a train of stimulation pulses on a Micro FE with a certain frequency, amplitude, pulse width, and duration.

As an example, the stimulation waveform is defined as a biphasic pulse, 
with properties that can be set by the user. However, the waveform can
be modified as the user sees fit.

Created on Wed Aug 07 2024

@author: richard
"""

import xipppy as xp
from time import sleep
import numpy as np

# Function that converts milliseconds to Ripple processor clock cycle units (33.33 us)
def ms_to_clk(ms):
    clk=np.ceil(ms*30)
    return int(clk)

# Function that sets stimulation segments and sequence for given channel, polarity,
# pulse width (in milliseconds), amplitude (in mA), frequency (in Hz), train duration (in milliseconds)
def setStimSeq(channel, pol, pw, amp, freq, train_dur):

    xp.stim_enable_set(False)
    sleep(0.001)
    xp.stim_set_res(channel, 3)   

    # Enable stimulation on processor          
    xp.stim_enable_set(True)

    # Conversion to # of steps. Assume step size is 10 uA (res = 3 for Micro)
    setAmp = int(amp*100)

    first_seg = []
    if amp > 0:
        first_seg = xp.StimSegment(ms_to_clk(pw/2), setAmp, pol)
    else:
        first_seg = xp.StimSegment(ms_to_clk(pw/2), 0, pol, enable=False)

    second_seg = []
    if amp > 0:
        second_seg = xp.StimSegment(ms_to_clk(pw/2), setAmp, -pol)
    else:
        second_seg = xp.StimSegment(ms_to_clk(pw/2), 0, pol, enable=False)

    chan_seq = xp.StimSeq(channel, ms_to_clk(1000/freq), int(freq*(train_dur/1000)), first_seg, second_seg, action=1)
    return chan_seq

def sendStim(channel):
    
    seq = []
    # Define Stim Train Parameters - 5 ms pulse width, 1 mA amplitude, 50 Hz, 1000 ms train duration
    seq.append(setStimSeq(channel=channel, pol=-1, pw=5, amp=1, freq=50, train_dur=1000))

    xp.StimSeq.send_stim_seqs(seq)

    sleep(1)

if __name__ == '__main__':
    try:
        with xp.xipppy_open():
            sendStim(0)
    except:
        try:
            with xp.xipppy_open(use_tcp=True):
                sendStim(0)
        except:
            print("Failed to connect to processor.")
        else:
            print("Connected over TCP")
    else:
        print("Connected over UDP")