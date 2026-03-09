import configparser
import argparse
import single_posterior_plot
import os

"""set up"""
#recieve the argments from command line
parser = argparse.ArgumentParser()
parser.add_argument('-1', '--config_path', help='path of config file')
args = parser.parse_args()

config_ini = configparser.ConfigParser()
config_ini.read(args.config_path, encoding='utf-8')
if not os.path.exists(args.config_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), args.config_path)
config_setting = config_ini['setting']

"""set the event to be analyzed"""
event_name = config_setting['event_name']
#index = config_setting['index']
#index = 'v1'
index = None
if index is not None:
    outdir = 'outdirs/outdir_' + event_name + '_' + index
    label = event_name + '_' + index
else:
    outdir = 'outdirs/outdir_' + event_name
    label = event_name

single_posterior_plot.plot_posterior(
                                    path_json = outdir+'/'+label+'_result.json',
                                    path_outdir = outdir,
                                    default_plot_parameters = True
                                    )
