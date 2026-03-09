#!/usr/bin/env python3
from __future__ import division, print_function
import bilby
from gwpy.timeseries import TimeSeries
import math
from bilby.gw.conversion import component_masses_to_chirp_mass, component_masses_to_symmetric_mass_ratio, luminosity_distance_to_redshift, chirp_mass_and_mass_ratio_to_component_masses
from bilby.core import utils
import numpy as np
from gwosc import datasets
from astropy import constants as const
import sys
import argparse
import configparser
import os
import errno
import single_posterior_plot
import json
import datetime
import scipy.signal.windows as windows
import cmath

import priors_condition

"""get current time"""
start = datetime.datetime.now()

"""constants"""
Mo = const.M_sun.value #solar mass [kg]
G = const.G.value #Newton constant [m^3 kg^-1 s^2]
c = const.c.value #light speed [m s^-1]
pc = const.pc.value #1pc [m]

"""recieve the argments from command line"""
parser = argparse.ArgumentParser()
parser.add_argument('-1', '--config_path', help='path of config file')
args = parser.parse_args()

config_ini = configparser.ConfigParser()
config_ini.read(args.config_path, encoding='utf-8')
if not os.path.exists(args.config_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.config_path)
config_setting = config_ini['setting']
config_injection = config_ini['injection_parameters']
config_wf_args = config_ini['waveform_arguments']
config_others = config_ini['other_parameters']

"""set the event to be analyzed"""
event_name = config_setting['event_name']
outdir_path = config_setting['outdir_path']
outdir = outdir_path + 'outdir_' + event_name
label = event_name
logger = bilby.core.utils.logger
bilby.core.utils.setup_logger(outdir=outdir, label=label)

#bilby.core.utils.logger.info("!!! KOKO GA ATARASHII CODE DESU !!!")
"""output to .log file"""
comment = config_setting['comment']
bilby.core.utils.logger.info(comment.replace('_', ' '))
bilby.core.utils.logger.info('event_name : '+event_name)

"""parameterization"""
parameterization = config_setting['parameterization']
bilby.core.utils.logger.info('parameterization : '+parameterization)

"""mode number"""
one_mode=False
mode_number = config_setting['mode_number']
if mode_number=='one_mode':
    one_mode=True
bilby.core.utils.logger.info('one_mode : '+str(one_mode))

analysis_one_mode = False
analysis_mode_number = config_setting['analysis_mode_number']
if analysis_mode_number=='analysis_one_mode':
    analysis_one_mode = True
bilby.core.utils.logger.info('analysis_one_mode : '+str(analysis_one_mode))

"""output waveform args"""
for key in config_wf_args.keys():
    bilby.core.utils.logger.info('{0} : {1}'.format(key, config_wf_args[key]))

"""window duration for signal roll on"""
signal_roll_on_duration = 0.0  #default:0s
if 'signal_roll_on_duration' in config_setting.keys():
    signal_roll_on_duration = float(config_setting['signal_roll_on_duration'])
bilby.core.utils.logger.info('signal_roll_on_duration : {}'.format(signal_roll_on_duration))

"""waveform"""
def toy_model_of_two_QNMs_window(time, A, alpha, f1, f2, tau1, tau2, phi1, phi2, geocent_time):
    
    waveform1 = np.zeros(len(time), dtype=complex)
    waveform2 = np.zeros(len(time), dtype=complex)

    A = A * 1e-17

    w1 = (2*np.pi*f1 + 1j / tau1)
    w2 = (2*np.pi*f2 + 1j / tau2)
    delta_w = w2 - w1
    
    tidx = time >= geocent_time
    
    waveform1[tidx] = A / delta_w * np.exp(1j * (w1 * (time[tidx] - geocent_time)) + 1j * phi1)
    waveform2[tidx] = -A / delta_w * (1 + delta_w * alpha) * np.exp(1j * (w2 * (time[tidx] - geocent_time)) + 1j * phi2)

    total_waveform = waveform1 + waveform2

    if one_mode:
        waveform2 = np.zeros(len(time), dtype=complex)
    
    """roll on windowing"""
    dt = time[1] - time[0]
    roll_on_samples = int(signal_roll_on_duration / dt)
    start_index = np.where(tidx)[0][0]
    taper_end_index = min(start_index + roll_on_samples, len(time))
    taper_len = taper_end_index - start_index
    
    hann_taper = windows.hann(2 * taper_len)[:taper_len]
    total_waveform[start_index:taper_end_index] *= hann_taper

    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = total_waveform.real * tukey_window
    cross = total_waveform.imag * tukey_window
    
    return {"plus": plus, "cross": cross}

def toy_model_of_two_QNMs_Heaviside(time, A, alpha, f1, f2, tau1, tau2, phi1, phi2, geocent_time):
    
    waveform1 = np.zeros(len(time), dtype=complex)
    waveform2 = np.zeros(len(time), dtype=complex)

    A = A * 1e-17

    w1 = (2*np.pi*f1 + 1j / tau1)
    w2 = (2*np.pi*f2 + 1j / tau2)
    delta_w = w2 - w1
    
    tidx = time >= geocent_time
    
    waveform1[tidx] = A / delta_w * np.exp(1j * (w1 * (time[tidx] - geocent_time)) + 1j * phi1)
    waveform2[tidx] = -A / delta_w * (1 + delta_w * alpha) * np.exp(1j * (w2 * (time[tidx] - geocent_time)) + 1j * phi2)

    total_waveform = waveform1 + waveform2

    if one_mode:
        waveform2 = np.zeros(len(time), dtype=complex)
    
    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = total_waveform.real * tukey_window
    cross = total_waveform.imag * tukey_window
    
    return {"plus": plus, "cross": cross}

def toy_model_of_two_QNMs_Mirror(time, A, alpha, f1, f2, tau1, tau2, phi1, phi2, geocent_time):
    
    waveform1 = np.zeros(len(time), dtype=complex)
    waveform2 = np.zeros(len(time), dtype=complex)

    A = A * 1e-17 / np.sqrt(2)

    w1 = (2*np.pi*f1 + 1j / tau1)
    w2 = (2*np.pi*f2 + 1j / tau2)
    delta_w = w2 - w1
        
    waveform1 = A / delta_w * np.exp(1j * (2*np.pi*f1 * (time - geocent_time)) + 1j * phi1) * np.exp(- np.abs(time - geocent_time) / tau1)
    waveform2 = -A / delta_w * (1 + delta_w * alpha) * np.exp(1j * (2*np.pi*f2 * (time - geocent_time)) + 1j * phi2) * np.exp(- np.abs(time - geocent_time) / tau2)

    total_waveform = waveform1 + waveform2

    if one_mode:
        waveform2 = np.zeros(len(time), dtype=complex)
    
    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = total_waveform.real * tukey_window
    cross = total_waveform.imag * tukey_window
    
    return {"plus": plus, "cross": cross}

def toy_model_of_two_QNMs_real_amp(time, A, alpha, f1, f2, tau1, tau2, phi1, phi2, geocent_time):

    waveform1 = np.zeros(len(time), dtype=complex)
    waveform2 = np.zeros(len(time), dtype=complex)

    A = A * 1e-17

    w1 = (2*np.pi*f1 + 1j / tau1)
    w2 = (2*np.pi*f2 + 1j / tau2)
    delta_w = w2 - w1
    
    tidx = time >= geocent_time

    real_amp_1 = np.abs(A / delta_w)
    real_amp_2 = np.abs(-A / delta_w * (1 + delta_w * alpha))

    waveform1[tidx] = real_amp_1 * np.exp(1j * (w1 * (time[tidx] - geocent_time)) + 1j * phi1)
    waveform2[tidx] = real_amp_2 * np.exp(1j * (w2 * (time[tidx] - geocent_time)) + 1j * phi2)

    total_waveform = waveform1 + waveform2

    if one_mode:
        waveform2 = np.zeros(len(time), dtype=complex)

    """roll on windowing"""
    dt = time[1] - time[0]
    roll_on_samples = int(signal_roll_on_duration / dt)
    start_index = np.where(tidx)[0][0]
    taper_end_index = min(start_index + roll_on_samples, len(time))
    taper_len = taper_end_index - start_index
    
    hann_taper = windows.hann(2 * taper_len)[:taper_len]
    total_waveform[start_index:taper_end_index] *= hann_taper

    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = total_waveform.real * tukey_window
    cross = total_waveform.imag * tukey_window
    
    return {"plus": plus, "cross": cross}

def damped_sinusoid(time, A1, A2, f1, f2, tau1, tau2, phi1, phi2, geocent_time):
    waveform1 = np.zeros(len(time), dtype=complex)
    waveform2 = np.zeros(len(time), dtype=complex)

    A1 = A1 * 1e-19
    A2 = A2 * 1e-19

    w1 = (2*np.pi*f1 + 1j / tau1)
    w2 = (2*np.pi*f2 + 1j / tau2)
    delta_w = w2 - w1
    
    tidx = time >= geocent_time

    waveform1[tidx] = A1 * np.exp(1j * (w1 * (time[tidx] - geocent_time)) + 1j * phi1)
    waveform2[tidx] = A2 * np.exp(1j * (w2 * (time[tidx] - geocent_time)) + 1j * phi2)

    if one_mode:
        waveform2 = np.zeros(len(time), dtype=complex)
    
    total_waveform = waveform1 + waveform2

    """roll on windowing"""
    dt = time[1] - time[0]
    roll_on_samples = int(signal_roll_on_duration / dt)
    start_index = np.where(tidx)[0][0]
    taper_end_index = min(start_index + roll_on_samples, len(time))
    taper_len = taper_end_index - start_index

    hann_taper = windows.hann(2 * taper_len)[:taper_len]
    total_waveform[start_index:taper_end_index] *= hann_taper

    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = total_waveform.real * tukey_window
    cross = total_waveform.imag * tukey_window

    return {"plus": plus, "cross": cross}

def damped_sinusoid_fdomain_Heaviside(frequency, A1, A2, f1, f2, tau1, tau2, phi1, phi2, geocent_time):
        A1 = A1 * 1e-19
        A2 = A2 * 1e-19

        d_positive_1 = (-1j / tau1) + (2 * (frequency + f1) * np.pi)
        d_negative_1 = (-1j / tau1) + (2 * (frequency - f1) * np.pi)
    
        d_positive_2 = (-1j / tau2) + (2 * (frequency + f2) * np.pi)
        d_negative_2 = (-1j / tau2) + (2 * (frequency - f2) * np.pi)
        
        plus = A1 * ((-1/tau1 - 1j*2*np.pi*frequency) * np.cos(phi1) + 2*np.pi*f1*np.sin(phi1)) / (d_positive_1 * d_negative_1) + A2 * ((-1/tau2 - 1j*2*np.pi*frequency) * np.cos(phi2) + 2*np.pi*f2*np.sin(phi2)) / (d_positive_2 * d_negative_2)
        cross = -A1 * (2*np.pi*f1*np.cos(phi1) + (1/tau1 + 1j*2*np.pi*frequency) * np.sin(phi1)) / (d_positive_1 * d_negative_1) - A2 * (2*np.pi*f2*np.cos(phi2) + (1/tau2 + 1j*2*np.pi*frequency) * np.sin(phi2)) / (d_positive_2 * d_negative_2)

        return {'plus': plus, 'cross': cross}

def damped_sinusoid_fdomain_mirror(frequency, A1, A2, f1, f2, tau1, tau2, phi1, phi2, geocent_time):
        A1 = A1 * 1e-19
        A2 = A2 * 1e-19

        d_positive_1 = (1 / tau1)**2 + (2 * (frequency + f1) * np.pi)**2
        d_negative_1 = (1 / tau1)**2 + (2 * (frequency - f1) * np.pi)**2
    
        d_positive_2 = (1 / tau2)**2 + (2 * (frequency + f2) * np.pi)**2
        d_negative_2 = (1 / tau2)**2 + (2 * (frequency - f2) * np.pi)**2
        
        plus = 1 / 2**0.5 * (A1 / tau1 * (np.exp(-1j * phi1) / d_positive_1 + np.exp(1j * phi1) / d_negative_1) + A2 / tau2 * (np.exp(-1j * phi2) / d_positive_2 + np.exp(1j * phi2) / d_negative_2))
        cross = 1 / 2**0.5 * (-1j) * (A1 / tau1 * (- np.exp(-1j * phi1) / d_positive_1 + np.exp(1j * phi1) / d_negative_1) + A2 / tau2 * (- np.exp(-1j * phi2) / d_positive_2 + np.exp(1j * phi2) / d_negative_2))

        return {'plus': plus, 'cross': cross}

def damped_sinusoid_one_mode_fdomain_Heaviside(frequency, A1, f1, tau1, phi1, geocent_time):
        A1 = A1 * 1e-19

        d_positive_1 = (-1j / tau1) + (2 * (frequency + f1) * np.pi)
        d_negative_1 = (-1j / tau1) + (2 * (frequency - f1) * np.pi)
    
        plus = A1 * ((-1/tau1 - 1j*2*np.pi*frequency) * np.cos(phi1) + 2*np.pi*f1*np.sin(phi1)) / (d_positive_1 * d_negative_1)
        cross = -A1 * (2*np.pi*f1*np.cos(phi1) + (1/tau1 + 1j*2*np.pi*frequency) * np.sin(phi1)) / (d_positive_1 * d_negative_1)

        return {'plus': plus, 'cross': -cross}

def damped_sinusoid_one_mode_fdomain_mirror(frequency, A1, f1, tau1, phi1, geocent_time):
        A1 = A1 * 1e-19

        d_positive_1 = (1 / tau1)**2 + (2 * (frequency + f1) * np.pi)**2
        d_negative_1 = (1 / tau1)**2 + (2 * (frequency - f1) * np.pi)**2
    
        plus = 1 / 2**0.5 * (A1 / tau1 * (np.exp(-1j * phi1) / d_positive_1 + np.exp(1j * phi1) / d_negative_1))
        cross = 1 / 2**0.5 * (-1j) * (A1 / tau1 * (- np.exp(-1j * phi1) / d_positive_1 + np.exp(1j * phi1) / d_negative_1))

        return {'plus': plus, 'cross': cross}

def damped_sinusoid_one_mode_tdomain_mirror(time, A1, f1, tau1, phi1, geocent_time):
    """
    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.
    """
    waveform = np.zeros(len(time), dtype=complex)

    A1 = A1 * 1e-19 / np.sqrt(2)
    w1 = (2*np.pi*f1 + 1j / tau1)
    
    waveform = A1 * np.exp(1j * 2 * np.pi * f1 * (time - geocent_time) + 1j * phi1) * np.exp(- np.abs(time - geocent_time) / tau1)
    
    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = waveform.real * tukey_window
    cross = waveform.imag * tukey_window
    
    return {"plus": plus, "cross": cross}

def damped_sinusoid_one_mode(time, A1, f1, tau1, phi1, geocent_time):
    """
    Returns
    -------
    dict:
        A dictionary containing "plus" and "cross" entries.
    """
    waveform = np.zeros(len(time), dtype=complex)

    A1 = A1 * 1e-19
    w1 = (2*np.pi*f1 + 1j / tau1)
    
    tidx = time >= geocent_time

    waveform[tidx] = A1 * np.exp(1j * (w1 * (time[tidx] - geocent_time)) + 1j * phi1)
    
    """roll on windowing"""
    dt = time[1] - time[0]
    roll_on_samples = int(signal_roll_on_duration / dt)
    start_index = np.where(tidx)[0][0]
    taper_end_index = min(start_index + roll_on_samples, len(time))
    taper_len = taper_end_index - start_index

    hann_taper = windows.hann(2 * taper_len)[:taper_len]
    waveform[start_index:taper_end_index] *= hann_taper

    """segment windowing"""
    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus = waveform.real * tukey_window
    cross = waveform.imag * tukey_window
    
    return {"plus": plus, "cross": cross}

def toy_model_of_two_QNMs_real_amp_fdomain_Mirror(frequency, A, alpha, f1, tau1, f2, tau2, phi1, phi2, geocent_time):
        A = A * 1e-17
        w1 = (2*np.pi*f1 + 1j / tau1)
        w2 = (2*np.pi*f2 + 1j / tau2)
        delta_w = w2 - w1

        real_amp_1 = np.abs(A / delta_w) / 2**0.5
        real_amp_2 = np.abs(-A / delta_w * (1 + delta_w * alpha)) / 2**0.5

        d_positive_1 = (1 / tau1)**2 + (2 * (frequency + f1) * np.pi)**2
        d_negative_1 = (1 / tau1)**2 + (2 * (frequency - f1) * np.pi)**2
        d_positive_2 = (1 / tau2)**2 + (2 * (frequency + f2) * np.pi)**2
        d_negative_2 = (1 / tau2)**2 + (2 * (frequency - f2) * np.pi)**2

        plus = real_amp_1 / tau1  * (np.exp(-1j * phi1) / d_positive_1 + np.exp(1j * phi1) / d_negative_1) + real_amp_2 / tau2 * (np.exp(-1j * phi2) / d_positive_2 + np.exp(1j * phi2) / d_negative_2)
        cross = 1j * (real_amp_1 / tau1 * (np.exp(-1j * phi1) / d_positive_1 - np.exp(1j * phi1) / d_negative_1) + real_amp_2 / tau2 * (np.exp(-1j * phi2) / d_positive_2 - np.exp(1j * phi2) / d_negative_2))
        
        return {'plus': plus, 'cross': cross}
    
def toy_model_of_two_QNMs_real_amp_fdomain_Heaviside(frequency, A, alpha, f1, tau1, f2, tau2, phi1, phi2, geocent_time):
    A = A * 1e-17
    w1 = (2*np.pi*f1 + 1j / tau1)
    w2 = (2*np.pi*f2 + 1j / tau2)
    delta_w = w2 - w1

    real_amp_1 = np.abs(A / delta_w)
    real_amp_2 = np.abs(-A / delta_w * (1 + delta_w * alpha))

    d_positive_1 = (-1j / tau1) + (2 * (frequency + f1) * np.pi)
    d_negative_1 = (-1j / tau1) + (2 * (frequency - f1) * np.pi)
    
    d_positive_2 = (-1j / tau2) + (2 * (frequency + f2) * np.pi)
    d_negative_2 = (-1j / tau2) + (2 * (frequency - f2) * np.pi)
        
    plus = real_amp_1 * ((-1/tau1 - 1j*2*np.pi*frequency) * np.cos(phi1) + 2*np.pi*f1*np.sin(phi1)) / (d_positive_1 * d_negative_1) + real_amp_2 * ((-1/tau2 - 1j*2*np.pi*frequency) * np.cos(phi2) + 2*np.pi*f2*np.sin(phi2)) / (d_positive_2 * d_negative_2)
    cross = -real_amp_1 * (2*np.pi*f1*np.cos(phi1) + (1/tau1 + 1j*2*np.pi*frequency) * np.sin(phi1)) / (d_positive_1 * d_negative_1) - real_amp_2 * (2*np.pi*f2*np.cos(phi2) + (1/tau2 + 1j*2*np.pi*frequency) * np.sin(phi2)) / (d_positive_2 * d_negative_2)

    return {'plus': plus, 'cross': cross}

def EP_waveform_tdomain_Mirror(time, A, alpha, f, tau, geocent_time):
    waveform = np.zeros(len(time), dtype=complex)

    A = A * 1e-17
    w_ep = (2 * np.pi * f + 1j / tau)

    waveform = - (1/np.sqrt(2)) * (A*alpha + 1j*A*np.abs(time - geocent_time)) * np.exp(1j * (2 * np.pi * f * (time - geocent_time))) * np.exp(- np.abs(time - geocent_time) / tau)

    plus = waveform.real
    cross = waveform.imag

    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus *= tukey_window
    cross *= tukey_window

    return {"plus": plus, "cross": cross}

def EP_waveform_tdomain_Heaviside(time, A, alpha, f, tau, geocent_time):
    waveform = np.zeros(len(time), dtype=complex)

    A = A * 1e-17
    w_ep = (2 * np.pi * f + 1j / tau)

    tidix = time >= geocent_time
    waveform[tidix] = - (A*alpha + 1j*A*(time[tidix] - geocent_time)) * np.exp(1j * (2 * np.pi * f * (time[tidix] - geocent_time))) * np.exp(- np.abs(time[tidix] - geocent_time) / tau)
    
    plus = waveform.real
    cross = waveform.imag

    tukey_window = windows.tukey(len(time), 0.1)  # Tukey window with alpha=0.1
    plus *= tukey_window
    cross *= tukey_window

    return {"plus": plus, "cross": cross}

"""analysis code"""
if __name__ == "__main__" :
    """set parameters"""
    np.random.seed(1)
    duration = float(config_others['duration'])
    sampling_frequency = float(config_others['sampling_frequency'])
    trigger_time = float(config_injection['geocent_time'])
    post_trigger_duration = float(config_others['post_trigger_duration'])
    start_time = trigger_time - duration + post_trigger_duration

    bilby.core.utils.logger.info('duration : {}'.format(duration))
    bilby.core.utils.logger.info('sampling frequency : {}'.format(sampling_frequency))
    bilby.core.utils.logger.info('trigger time : {}'.format(trigger_time))
    bilby.core.utils.logger.info('start time : {}'.format(start_time))

    config_injection['tau1'] = str(np.abs(float(config_injection['tau1']))) # make tau1 positive to be consistent with FT convention
    config_injection['tau2'] = str(np.abs(float(config_injection['tau2'])))
    """set injection parameters and waveform generator"""
    analyze_EP = False
    if parameterization == 'EP_Mirror' or parameterization == 'EP_Heaviside':
        analyze_EP = True
        if parameterization == 'EP_Heaviside':
            parameterization = 'OT_Heaviside'
            waveform_generator_EP = bilby.gw.waveform_generator.WaveformGenerator(
                                        duration = duration,
                                        sampling_frequency= sampling_frequency,
                                        time_domain_source_model= EP_waveform_tdomain_Heaviside,
                                        start_time = start_time,
                                        )
        elif parameterization == 'EP_Mirror':
            parameterization = 'OT_Mirror'
            waveform_generator_EP = bilby.gw.waveform_generator.WaveformGenerator(
                                            duration = duration,
                                            sampling_frequency= sampling_frequency,
                                            time_domain_source_model= EP_waveform_tdomain_Mirror,
                                            start_time = start_time,
                                            )

    if parameterization == 'DS_window' or parameterization == 'DS_Mirror' or parameterization == 'DS_Heaviside':
        """calculate A1, A2, phi1, phi2 from A, alpha, f1, tau1, f2, tau2"""
        A = float(config_injection['A']) * 1e-20  # Convert to strain unit
        alpha = float(config_injection['alpha'])
        w1 = (2 * np.pi * float(config_injection['f1']) + 1j / float(config_injection['tau1']))
        w2 = (2 * np.pi * float(config_injection['f2']) + 1j / float(config_injection['tau2']))
        delta_w = w2 - w1
        A1_complex = A / delta_w
        A1 = np.abs(A1_complex) / 1e-20
        phi1_from_amp = np.angle(A1_complex)
        phi1 = float(config_injection['phi1']) + phi1_from_amp
        A2_complex = - A * (1 + alpha * delta_w) / delta_w
        A2 = np.abs(A2_complex) / 1e-20
        phi2_from_amp = np.angle(A2_complex)
        phi2 = float(config_injection['phi2']) + phi2_from_amp

        """print parameters"""
        bilby.core.utils.logger.info('')
        bilby.core.utils.logger.info('--------------------------------')
        bilby.core.utils.logger.info('A1 : {}'.format(A1))
        bilby.core.utils.logger.info('A2 : {}'.format(A2))
        bilby.core.utils.logger.info('alpha : {}'.format(alpha))
        bilby.core.utils.logger.info('phi1 : {}'.format(phi1))
        bilby.core.utils.logger.info('phi2 : {}'.format(phi2))
        bilby.core.utils.logger.info('w1 : {}'.format(w1))
        bilby.core.utils.logger.info('w2 : {}'.format(w2))
        bilby.core.utils.logger.info('delta_w = w2-w1 : {}'.format(delta_w))
        bilby.core.utils.logger.info('delta_w * alpha : {}'.format(delta_w * alpha))
        bilby.core.utils.logger.info('Re[delta_w] / Re[w1] = {}'.format(delta_w.real / w1.real))
        bilby.core.utils.logger.info('Im[delta_w] / Im[w1] = {}'.format(delta_w.imag / w1.imag))
        bilby.core.utils.logger.info('|delta_w| = {}'.format(np.abs(delta_w)))
        bilby.core.utils.logger.info('|delta_w| / |w1| = {}'.format(np.abs(delta_w) / np.abs(w1)))
        bilby.core.utils.logger.info('A / |delta_w| = {}'.format(float(config_injection['A']) / np.abs(delta_w)))
        bilby.core.utils.logger.info('--------------------------------')
        bilby.core.utils.logger.info('')
        """"""

        # A1 = float(config_injection['A1'])
        # A2 = float(config_injection['A2'])
        # phi1 = float(config_injection['phi1'])
        # phi2 = float(config_injection['phi2'])
        injection_parameters = dict(
                                    A1 = A1 * 1e-1,
                                    A2 = A2 * 1e-1,
                                    f1 = float(config_injection['f1']),
                                    f2 = float(config_injection['f2']),
                                    # delta_f = float(config_injection['f2']) - float(config_injection['f1']),
                                    tau1 = float(config_injection['tau1']),
                                    tau2 = float(config_injection['tau2']),
                                    # delta_tau = float(config_injection['tau2']) - float(config_injection['tau1']),
                                    phi1 = phi1,
                                    phi2 = phi2,
                                    ra = float(config_injection['ra']),
                                    dec = float(config_injection['dec']),
                                    psi = float(config_injection['psi']),
                                    geocent_time = float(config_injection['geocent_time']),
                                    )

        if not one_mode:
            if parameterization == 'DS_window':
                waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                    duration = duration,
                                    sampling_frequency = sampling_frequency,
                                    time_domain_source_model = damped_sinusoid,
                                    start_time = start_time,
                                    parameter_conversion=None,
                                    )
            elif parameterization == 'DS_Mirror':
                waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                    duration = duration,
                                    sampling_frequency = sampling_frequency,
                                    frequency_domain_source_model = damped_sinusoid_fdomain_mirror,
                                    start_time = start_time,
                                    parameter_conversion=None,
                                    )
            elif parameterization == 'DS_Heaviside':
                waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                    duration = duration,
                                    sampling_frequency = sampling_frequency,
                                    frequency_domain_source_model = damped_sinusoid_fdomain_Heaviside,
                                    start_time = start_time,
                                    parameter_conversion=None,
                                    )
        else:
            remove_keys = ['A2', 'f2', 'tau2', 'phi2']
            for key in remove_keys:
                injection_parameters.pop(key, None)
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                            duration = duration,
                            sampling_frequency = sampling_frequency,
                            # frequency_domain_source_model = damped_sinusoid_one_mode_fdomain_mirror,
                            frequency_domain_source_model = damped_sinusoid_one_mode_fdomain_Heaviside,
                            # time_domain_source_model = damped_sinusoid_one_mode_tdomain_mirror,
                            start_time = start_time,
                            )

    if analysis_one_mode:
        waveform_generator_one_mode_analysis = bilby.gw.waveform_generator.WaveformGenerator(
                        duration = duration,
                        sampling_frequency = sampling_frequency,
                        # frequency_domain_source_model = damped_sinusoid_one_mode_fdomain_mirror,
                        frequency_domain_source_model = damped_sinusoid_one_mode_fdomain_Heaviside,
                        # time_domain_source_model = damped_sinusoid_one_mode_tdomain_mirror,
                        start_time = start_time,
                        )

    if parameterization == 'OT_Mirror' or parameterization == 'OT_Heaviside' or parameterization == 'OT_window':
        injection_parameters = dict(
                                    A = float(config_injection['A']) * 1e-3,
                                    alpha = float(config_injection['alpha']),
                                    f1 = float(config_injection['f1']),
                                    f2 = float(config_injection['f2']),
                                    # delta_f = float(config_injection['f2']) - float(config_injection['f1']),
                                    tau1 = float(config_injection['tau1']),
                                    tau2 = float(config_injection['tau2']),
                                    # delta_tau = float(config_injection['tau2']) - float(config_injection['tau1']),
                                    phi1 = float(config_injection['phi1']),
                                    phi2 = float(config_injection['phi2']),
                                    ra = float(config_injection['ra']),
                                    dec = float(config_injection['dec']),
                                    psi = float(config_injection['psi']),
                                    geocent_time = float(config_injection['geocent_time']),
                                    )

        if parameterization == 'OT_Mirror':
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                 duration = duration,
                                 sampling_frequency = sampling_frequency,
                                 time_domain_source_model = toy_model_of_two_QNMs_Mirror,
                                 start_time = start_time,
                                 )
        elif parameterization == 'OT_Heaviside':
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                 duration = duration,
                                 sampling_frequency = sampling_frequency,
                                 time_domain_source_model = toy_model_of_two_QNMs_Heaviside,
                                 start_time = start_time,
                                 )
        elif parameterization == 'OT_window':
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                 duration = duration,
                                 sampling_frequency = sampling_frequency,
                                 time_domain_source_model = toy_model_of_two_QNMs_window,
                                 start_time = start_time,
                                 )
    
    if parameterization == 'OT_real_amp_Mirror' or parameterization == 'OT_real_amp_Heaviside':
        delta_w = (2 * np.pi * float(config_injection['f2']) + 1j / float(config_injection['tau2'])) - (2 * np.pi * float(config_injection['f1']) + 1j / float(config_injection['tau1']))
        amp1 = float(config_injection['A']) / delta_w
        amp2 = - float(config_injection['A']) * (1 + delta_w * float(config_injection['alpha'])) / delta_w
        phi1_from_amp = np.angle(amp1)
        phi2_from_amp = np.angle(amp2)
        phi1 = float(config_injection['phi1']) + phi1_from_amp
        phi2 = float(config_injection['phi2']) + phi2_from_amp
        injection_parameters = dict(
                                    A = float(config_injection['A']) * 1e-3,
                                    alpha = float(config_injection['alpha']),
                                    f1 = float(config_injection['f1']),
                                    f2 = float(config_injection['f2']),
                                    # delta_f = float(config_injection['f2']) - float(config_injection['f1']),
                                    tau1 = float(config_injection['tau1']),
                                    tau2 = float(config_injection['tau2']),
                                    # delta_tau = float(config_injection['tau2']) - float(config_injection['tau1']),
                                    phi1 = phi1,
                                    phi2 = phi2,
                                    ra = float(config_injection['ra']),
                                    dec = float(config_injection['dec']),
                                    psi = float(config_injection['psi']),
                                    geocent_time = float(config_injection['geocent_time']),
                                    )

        if parameterization == 'OT_real_amp_Mirror':
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                 duration = duration,
                                 sampling_frequency = sampling_frequency,
                                 frequency_domain_source_model = toy_model_of_two_QNMs_real_amp_fdomain_Mirror,
                                 start_time = start_time,
                                 )
        elif parameterization == 'OT_real_amp_Heaviside':
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
                                 duration = duration,
                                 sampling_frequency = sampling_frequency,
                                 frequency_domain_source_model = toy_model_of_two_QNMs_real_amp_fdomain_Heaviside,
                                 start_time = start_time,
                                 )
    
    """set interferometers"""
    ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

    # ifos.set_strain_data_from_zero_noise(
    #                                sampling_frequency = sampling_frequency,
    #                                duration = duration,
    #                                start_time = start_time
    #                                )
    
    ifos.set_strain_data_from_power_spectral_densities(
                                    sampling_frequency = sampling_frequency,
                                    duration = duration,
                                    start_time = start_time
                                    )

    for interferometer in ifos:
        interferometer.minimum_frequency = float(config_wf_args['minimum_frequency'])
        interferometer.maximum_frequency = float(config_wf_args['maximum_frequency'])

    """inject signal"""
    ifos.inject_signal(
        waveform_generator=waveform_generator,
        # waveform_generator=waveform_generator_analysis,
        parameters=injection_parameters,
        raise_error=False
    )

    """calculate SNR"""
    SNR = []
    for num, ifo in enumerate(ifos):
            hf_det = ifo.get_detector_response(waveform_generator.frequency_domain_strain(injection_parameters), injection_parameters)
            SNR.append(np.sqrt(np.real(ifo.optimal_snr_squared(hf_det))))
    network_SNR = np.sqrt(np.sum(np.array(SNR)**2.))
    bilby.core.utils.logger.info('')
    bilby.core.utils.logger.info('--------------------------------')
    bilby.core.utils.logger.info('SNR: bilby {}'.format(SNR))
    bilby.core.utils.logger.info('network SNR: {}'.format(network_SNR))
    bilby.core.utils.logger.info('--------------------------------')
    bilby.core.utils.logger.info('')

    """set prior"""
    if analysis_one_mode:
        priors = {}
        priors['A1'] = bilby.core.prior.Uniform(0, 100, r"$A_1$")
        priors['f1'] = bilby.core.prior.Uniform(20, 500, r"$f_1$ [Hz]")
        priors['tau1'] = bilby.core.prior.Uniform(0.0005, 0.05, r"$\tau_1$ [ms]")
        priors['phi1'] = bilby.core.prior.Uniform(-np.pi, np.pi, r"$\phi_1$ [rad]", boundary='periodic')
    elif analyze_EP:
        priors = {}
        priors['A'] = bilby.core.prior.Uniform(0, 100, r"$A$")
        priors['alpha'] = bilby.core.prior.Uniform(0, 1, r"$\alpha$")
        priors['f'] = bilby.core.prior.Uniform(20, 500, name='f', latex_label=r"$f$ [Hz]")
        priors['tau'] = bilby.core.prior.Uniform(0.0005, 0.05, name='tau', latex_label=r"$\tau$ [ms]")
    else:
        # priors = injection_parameters.copy()
        priors = bilby.core.prior.ConditionalPriorDict(injection_parameters.copy())

        if parameterization == 'DS_window' or parameterization == 'DS_Mirror' or parameterization == 'DS_Heaviside':
            priors['A1'] = bilby.core.prior.Uniform(0, 100, r"$A_1$")
            priors['A2'] = bilby.core.prior.Uniform(0, 100, r"$A_2$")
            
            if injection_parameters['f1'] == injection_parameters['f2']: # f1==f2 --> hierarchical by tau; tau1>tau2
                priors['f1'] = bilby.core.prior.Uniform(20, 500, name='f1', latex_label=r"$f_1$ [Hz]")
                priors['f2'] = bilby.core.prior.Uniform(20, 500, name='f2', latex_label=r"$f_2$ [Hz]")
                priors['tau1'] = bilby.core.prior.Triangular(mode=0.05, minimum=0.0005, maximum=0.05, name='tau1', latex_label=r"$\tau_1$ [ms]")
                priors['tau2'] = bilby.core.prior.conditional.ConditionalUniform(condition_func=priors_condition.condition_for_tau2, name='tau2', latex_label=r'$\tau_2$ [ms]', minimum=0.0005, maximum=0.05)
            else: # f1!=f2 --> hierarchical by f; f1>f2
                priors['f1'] = bilby.core.prior.Triangular(mode=500, minimum=20, maximum=500, name='f1', latex_label=r"$f_1$ [Hz]")
                priors['f2'] = bilby.core.prior.conditional.ConditionalUniform(condition_func=priors_condition.condition_for_f2, name='f2', latex_label=r'$f_2$ [Hz]', minimum=20, maximum=500)
                priors['tau1'] = bilby.core.prior.Uniform(0.0005, 0.05, name='tau1', latex_label=r"$\tau_1$ [ms]")
                priors['tau2'] = bilby.core.prior.Uniform(0.0005, 0.05, name='tau2', latex_label=r"$\tau_2$ [ms]")

        if parameterization == 'OT' or parameterization == 'OT_real_amp_Mirror' or parameterization == 'OT_real_amp_Heaviside':
            priors['A'] = bilby.core.prior.Uniform(0, 1, r"$A$")
            priors['alpha'] = bilby.core.prior.Uniform(0, 1, r"$\alpha$")

            if injection_parameters['f1'] == injection_parameters['f2']: # f1==f2 --> hierarchical by tau; tau1>tau2
                priors['f1'] = bilby.core.prior.Uniform(20, 500, name='f1', latex_label=r"$f_1$ [Hz]")
                priors['f2'] = bilby.core.prior.Uniform(20, 500, name='f2', latex_label=r"$f_2$ [Hz]")
                priors['tau1'] = bilby.core.prior.Triangular(mode=0.05, minimum=0.0005, maximum=0.05, name='tau1', latex_label=r"$\tau_1$ [ms]")
                priors['tau2'] = bilby.core.prior.conditional.ConditionalUniform(condition_func=priors_condition.condition_for_tau2, name='tau2', latex_label=r'$\tau_2$ [ms]', minimum=0.0005, maximum=0.05)
            else: # f1!=f2 --> hierarchical by f; f1>f2
                priors['f1'] = bilby.core.prior.Triangular(mode=500, minimum=20, maximum=500, name='f1', latex_label=r"$f_1$ [Hz]")
                priors['f2'] = bilby.core.prior.conditional.ConditionalUniform(condition_func=priors_condition.condition_for_f2, name='f2', latex_label=r'$f_2$ [Hz]', minimum=20, maximum=500)
                priors['tau1'] = bilby.core.prior.Uniform(0.0005, 0.05, name='tau1', latex_label=r"$\tau_1$ [ms]")
                priors['tau2'] = bilby.core.prior.Uniform(0.0005, 0.05, name='tau2', latex_label=r"$\tau_2$ [ms]")
        
        priors['phi1'] = bilby.core.prior.Uniform(-np.pi, np.pi, r"$\phi_1$ [rad]", boundary='periodic')
        priors['phi2'] = bilby.core.prior.Uniform(-np.pi, np.pi, r"$\phi_2$ [rad]", boundary='periodic')

        """set prior of fixed parameters"""
        if mode_number == 'one_mode':
            priors['A2'] = 0
            # priors['alpha'] = injection_parameters['alpha']
            priors['f2'] = injection_parameters['f2']
            priors['tau2'] = injection_parameters['tau2']
            priors['phi2'] = injection_parameters['phi2']

    priors['ra'] = bilby.core.prior.Uniform(0, 2*np.pi, name='ra', latex_label='$\\alpha$', unit='rad', boundary='periodic')
    priors['dec'] = bilby.core.prior.Cosine(name='dec', latex_label='$\\delta$', unit='rad', boundary='reflective')
    priors['psi'] = bilby.core.prior.Uniform(0, np.pi, name='psi', latex_label='$\\psi$', unit='rad', boundary='periodic')
    priors['geocent_time'] = bilby.core.prior.Uniform(minimum = trigger_time - duration/2, maximum = trigger_time + duration/2)
    
    if 'fix_parameters' in config_ini.keys():
        fix_list =  json.loads(config_ini['fix_parameters']['fix_list']) #str to list
        for key in fix_list:
            if key == 'A2':
                if one_mode:
                    priors[key] = 0
                else:
                    priors[key] = injection_parameters[key]
            else:
                priors[key] = injection_parameters[key]

    """set likelihood"""
    if analysis_one_mode:
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
                                                            interferometers=ifos,
                                                            priors=priors,
                                                            waveform_generator=waveform_generator_one_mode_analysis,
                                                            distance_marginalization=False,
                                                            phase_marginalization=False
                                                            )
    elif analyze_EP:
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
                                                            interferometers=ifos,
                                                            priors=priors,
                                                            waveform_generator=waveform_generator_EP,
                                                            distance_marginalization=False,
                                                            phase_marginalization=False
                                                            )
    else:
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
                                                                interferometers=ifos,
                                                                priors=priors,
                                                                waveform_generator=waveform_generator,
                                                                distance_marginalization=False,
                                                                phase_marginalization=False
                                                                )
    
    """RUN SAMPLER"""
    result = bilby.run_sampler(
        likelihood,
        priors,
        outdir=outdir,
        label=label,
        sampler="dynesty",
        injection_parameters=injection_parameters, #plot injection parameter
        nlive=3000, #default:1000
        sample="acceptance-walk", #default:'act-walk'
        npool=64,
        naccept=60, #default:60
        nact=20, #default:2
        walks=200, #default:100
        maxmcmc=15000, #default:5000
        queue_size=64,
        dlogz=0.1, #default:0.1
        conversion_function=None,
        result_class=bilby.gw.result.CBCResult, #default:bilby.core.result.Result
        resume=False,
        )
    
    """get time taken to analyze"""
    end = datetime.datetime.now()
    time_diff = end - start
    bilby.core.utils.logger.info('time taken to analyze : {}'.format(time_diff))

    """Plot result"""
    """write down fixed parameter to json file"""
    if 'fix_parameters' in config_ini.keys():
        path_json = outdir+'/'+label+'_result.json'
        with open(path_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            fix_list =  json.loads(config_ini['fix_parameters']['fix_list']) #str to list
            data["injection_parameters"]["fix_list"] = fix_list
        with open(path_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    """corner plot from our code"""
    single_posterior_plot.plot_posterior(
                                        path_json = outdir+'/'+label+'_result.json',
                                        path_outdir = outdir,
                                        default_plot_parameters = True
                                        )

    # result.plot_skymap(maxpts=5000) #maxpts:Maximum number of samples to use, if None all samples are used

    # result = bilby.result.read_in_result(filename=outdir+'/GW200105_result.json')
    # result.plot_corner()
    #                  parameters=plot_parameter_keys, #we can set only one from parameters and truths
    #                  quantiles=[0.05,0.95], #default quantiles:(0.16, 0.84), used caluculate errors bars
    #                  #truths=plot_injection #if set, plot injection parameters
    #                  )