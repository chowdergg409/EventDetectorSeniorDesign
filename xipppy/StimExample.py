#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example script for sending a train of stimulation pulses.

As an example, the stimulation waveform is defined as a biphasic pulse, 
with properties that can be set by the user. However, the waveform can
be modified as the user sees fit.

Created on Wed Nov 11 2022

Edited Aug 2024 by richard

@author: kyleloizos
"""

import xipppy as xp
from time import sleep

def stimWaveform(stim_channel, pulse_width, stim_mag_steps, stim_res):
    # stim_channel = channel # to send stimulus to
    # pulse_width = width of each phase (units of clock cycles)
    # magnitude = magnitude of stim pulse (units of steps)
    # stim_res = index of resolution of stim desired 
    #    (e.g.for nano, 1=1uA/step, 2=2uA/step, 3=5uA/step, 4=10uA/step, 5=20uA/step)

    xp.stim_enable_set(False)
    sleep(0.001)
    xp.stim_set_res(stim_channel,stim_res)   

    # Enable stimulation on NIP           
    xp.stim_enable_set(True)
    
    # Design stimulation waveform
    # Biphasic cathodic-first (single pulse)
    pseg = xp.StimSegment(pulse_width,stim_mag_steps,-1)
    ipi = xp.StimSegment(round(pulse_width/2),0,1, enable=False) 
    nseg = xp.StimSegment(pulse_width,stim_mag_steps,1) 
    seq0 = xp.StimSeq(stim_channel, 1000,1,pseg,ipi,nseg)
        
    return seq0

def sendStim(max_stim_count):
    
    # Define stimulation waveform
    stim_waveform = stimWaveform(stim_channel=1,pulse_width=200,stim_mag_steps=50,stim_res=4)
    stim_count = 0
    
    # Stimulate every second until maximum stim count is reached
    while stim_count < max_stim_count:
        sleep(0.1)
        xp.StimSeq.send(stim_waveform)
        stim_count = stim_count + 1
        print("Spike count: ", stim_count, "out of ", max_stim_count)
        sleep(1)
    
if __name__ == '__main__':
    try:
        with xp.xipppy_open():
            sendStim(20)
    except:
        try:
            with xp.xipppy_open(use_tcp=True):
                sendStim(20)
        except:
            print("Failed to connect to processor.")
        else:
            print("Connected over TCP")
    else:
        print("Connected over UDP")