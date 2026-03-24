import corner
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib
from gwpy.timeseries import TimeSeries
import math
import bilby
from bilby.gw.conversion import generate_all_bbh_parameters
from bilby.core import utils
import numpy as np
from gwosc import datasets
import h5py
import pesummary
from pesummary.io import read
from pesummary.core.plots.plot import _make_comparison_corner_plot
import seaborn as sns
import pandas as pd
import itertools
import argparse
import warnings
import os
import json
import configparser
import errno

def plot_corner_pyring(path_posterior_sample, path_config, path_outdir, show_fig=False):
    """ignore warnings"""
    warnings.filterwarnings("ignore", category=FutureWarning, 
                            module="seaborn._oldcore", lineno=1119)
    warnings.filterwarnings("ignore", category=FutureWarning, 
                            module="seaborn._oldcore", lineno=1498)
    
    """plot setting"""
    plt.style.use('~/research/my_plot_style.style')
    plt.style.use('seaborn-v0_8-colorblind')
    plt.style.use('seaborn-v0_8-whitegrid')
    first_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
    second_color = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    linestyles = ['--', '-', ':', '-.']
    
    """set labels"""
    latex_labels = {
        'mass_1_source': r'$m_{1,\mathrm{source}}\ [M_{\odot}]$',    
        'mass_2_source': r'$m_{2,\mathrm{source}}\ [M_{\odot}]$',    
        'mass_1': r'$m_{1}\ [M_{\odot}]$',    
        'mass_2': r'$m_{2}\ [M_{\odot}]$',    
        'chirp_mass_source': r'$\mathcal{M}_{\mathrm{source}}\ [M_{\odot}]$',
        'chirp_mass': r'$\mathcal{M}\ [M_{\odot}]$',
        'mass_ratio': r'$q$',
        'total_mass_source': r'$m_{\mathrm{total,source}}\ [M_{\odot}]$',
        'symmetric_mass_ratio': r'$\eta$',
        'chi_1': r'$\chi_{\rm 1}$',
        'chi_2': r'$\chi_{\rm 2}$',
        'chi_eff': r'$\chi_{\rm eff}$',
        'luminosity_distance': r'$d_L\ [{\rm Mpc}]$',
        'theta_jn': r'$\iota\ [{\rm rad}]$',
        'cos(iota)': r'$\cos \iota$',
        'sin(iota)': r'$\sin \iota$',
        'sin^2(iota)': r'$\sin^{2} \iota$',
        'geocent_time': r'$t_{c}$',
        'psi': r'$\psi$',
        'phase': r'$\phi$',
        'A_b1': r'$A_{b1}$',
        'A_b2': r'$A_{b2}$',
        'tilde_Ab1': r'$\tilde{A}_{b1}$',
        'tilde_Ab2': r'$\tilde{A}_{b2}$',
        'A1_real': r"$\mathrm{Re}[A_1] \times 10^{-20}$",
        'A1_imag': r"$\mathrm{Im}[A_1] \times 10^{-20}$",
        'alpha_real': r"$\mathrm{Re}[\alpha]$",
        'alpha_imag': r"$\mathrm{Im}[\alpha]$",
        'w1_real': r"$\mathrm{Re}[w_1]$",
        'w1_imag': r"$\mathrm{Im}[w_1]$",
        'w2_real': r"$\mathrm{Re}[w_2]$",
        'w2_imag': r"$\mathrm{Im}[w_2]$",
        'A': r'$A \times 10^{-17}$',
        'A1': r'$A_1 \times 10^{-19}$',
        'A2': r'$A_2 \times 10^{-19}$',
        'alpha': r'$\alpha$',
        'f': r'$f \ [\mathrm{Hz}]$',
        'f1': r'$f_1 \ [\mathrm{Hz}]$',
        'f2': r'$f_2 \ [\mathrm{Hz}]$',
        'tau': r'$\tau \ [\mathrm{s}]$',
        'tau1': r'$\tau_1 \ [\mathrm{s}]$',
        'tau2': r'$\tau_2 \ [\mathrm{s}]$',
        'phi1': r'$\phi_1 \ [\mathrm{rad}]$',
        'phi2': r'$\phi_2 \ [\mathrm{rad}]$',
        """for pyring"""
        'logA': r'$\log A \times 10^{-17}$',
        'logA_t_0': r'$\log A_1$',
        'logA_t_1': r'$\log A_2$',
        'A_t_0': r'$A_1 \times 10^{-19}$',
        'A_t_1': r'$A_2 \times 10^{-19}$',
        'f_t_0': r'$f_1 \ [\mathrm{Hz}]$',
        'f_t_1': r'$f_2 \ [\mathrm{Hz}]$',
        'tau_t_0': r'$\tau_1 \ [\mathrm{s}]$',
        'tau_t_1': r'$\tau_2 \ [\mathrm{s}]$',
        'phi_t_0': r'$\phi_1 \ [\mathrm{rad}]$',
        'phi_t_1': r'$\phi_2 \ [\mathrm{rad}]$',
        'C' : r'$C$',
        'D' : r'$D$',
    }

    """set function of conversion"""
    def conversion_samples(sample_pandas):
        # INPUT
        # sample_dict: posteriors
        new_sample_list = []
        for _, sample in sample_pandas.iterrows():
            new_sample = generate_all_bbh_parameters(sample)
            new_sample_list.append(new_sample)
    
        new_sample_pandas = pd.DataFrame(new_sample_list)
        return new_sample_pandas
    
    label = 'injection'

    """read posterior file"""
    file_path_ver1 = path_posterior_sample
    df_ver1 = pd.read_csv(file_path_ver1, sep='\s+')
    if '#' in df_ver1.columns:
        if df_ver1.iloc[:, -1].isna().all():
            real_columns = df_ver1.columns[1:].tolist()
            df_ver1 = df_ver1.iloc[:, :-1]
            df_ver1.columns = real_columns
            # print("Detected header misalignment due to '#'. Fixing columns...")
    # print(df_ver1)

    """set injection value of version1"""
    config_ini = configparser.ConfigParser()
    config_ini.optionxform = str
    config_ini.read(path_config, encoding='utf-8')
    if not os.path.exists(path_config):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_config)
    config_injection = config_ini['Injection']
    injection_parameters_dict = {}
    for key, val in config_injection.items():
        if 'A_' in key:
            injection_parameters_dict[key] = float(val) * 1e19
        elif key == 'A':
            injection_parameters_dict[key] = float(val) * 1e17
        else:
            injection_parameters_dict[key] = float(val)
    print(' injection_parameters : {}'.format(injection_parameters_dict))

    """set fixed parameters"""
    config_prior = config_ini['Priors']
    prior_parameters_dict = {key: value for key, value in config_prior.items()}
    config_input = config_ini['input']
    mode = json.loads(config_input.get('inject-n-ds-modes', '{}'))
    fix_parameters_list = []
    for key in prior_parameters_dict.keys():
        if 'fix-' in key:
            param_name = key.replace('fix-', '')
            if param_name == 't':
                fix_parameters_list.append('t')
                for num in range(mode['t']):
                    param_name = f't_t_{num}'
                    fix_parameters_list.append(param_name)
            else:
                fix_parameters_list.append(param_name)
    
    """print keys to plot"""
    keys_to_plot = list(injection_parameters_dict.keys())
    for key in keys_to_plot:
        print('plot key : {}'.format(key))

    """fixed parameters"""
    if fix_parameters_list != []:
        for key in fix_parameters_list:
            if key in keys_to_plot:
                keys_to_plot.remove(key)
                print('not plot key : {}'.format(key))

    """change theta_jn to cos(theta_jn)"""
    if 'theta_jn' in keys_to_plot:
        change = 'cos'
        #change = 'sin'
        #change = 'sin^2'
        if change=='cos':
            df_ver1['theta_jn'] = np.cos(df_ver1['theta_jn'])
            df_ver1 = df_ver1.rename(columns={'theta_jn':'cos(iota)'})
            injection_parameters_dict['cos(iota)'] = np.cos(injection_parameters_dict['theta_jn'])
        elif change=='sin':
            df_ver1['theta_jn'] = np.sin(df_ver1['theta_jn'])
            df_ver1 = df_ver1.rename(columns={'theta_jn':'sin(iota)'})
            injection_parameters_dict['sin(iota)'] = np.sin(injection_parameters_dict['theta_jn'])
        elif change=='sin^2':
            df_ver1['theta_jn'] = np.sin(df_ver1['theta_jn'])**2.
            df_ver1 = df_ver1.rename(columns={'theta_jn':'sin^2(iota)'})
            injection_parameters_dict['sin^2(iota)'] = np.sin(injection_parameters_dict['theta_jn'])**2.

    """change delta_f to f2, delta_tau to tau2"""
    if 'delta_f' in keys_to_plot:
        df_ver1['delta_f'] = df_ver1['delta_f'] + df_ver1['f1']
        df_ver1 = df_ver1.rename(columns={'delta_f':'f2'})
        injection_parameters_dict['f2'] = injection_parameters_dict['delta_f'] + injection_parameters_dict['f1']
    if 'delta_tau' in keys_to_plot:
        df_ver1['delta_tau'] = df_ver1['delta_tau'] + df_ver1['tau1']
        df_ver1 = df_ver1.rename(columns={'delta_tau':'tau2'})
        injection_parameters_dict['tau2'] = injection_parameters_dict['delta_tau'] + injection_parameters_dict['tau1']
    if 'f' in keys_to_plot and 'f1' in injection_parameters_dict.keys():
        injection_parameters_dict['f'] = (injection_parameters_dict['f1'] + injection_parameters_dict['f2']) / 2
    if 'tau' in keys_to_plot and 'tau1' in injection_parameters_dict.keys():
        injection_parameters_dict['tau'] = (injection_parameters_dict['tau1'] + injection_parameters_dict['tau2']) / 2
    
    """change log_A to A"""
    if 'logA' in df_ver1.columns:
        df_ver1['logA'] = 10**(df_ver1['logA']) * 1e17
        df_ver1 = df_ver1.rename(columns={'logA':'A'})
    if 'logA_t_0' in df_ver1.columns:
        df_ver1['logA_t_0'] = 10**(df_ver1['logA_t_0']) * 1e19
        df_ver1 = df_ver1.rename(columns={'logA_t_0':'A_t_0'})
    if 'logA_t_1' in df_ver1.columns:
        df_ver1['logA_t_1'] = 10**(df_ver1['logA_t_1']) * 1e19
        df_ver1 = df_ver1.rename(columns={'logA_t_1':'A_t_1'})

    """set plot data"""
    df_ver1['source'] = 'ver1' # Add 'source' line
    use_data = df_ver1[keys_to_plot + ['source']] # Select only keys to plot and 'source'

    """set latex format for error bar"""
    def format_error_latex_scaled(median, dif_lower, dif_upper, significant_digits=3):
        max_error = max(abs(dif_lower), abs(dif_upper))
        if max_error == 0:
            return f"${median}_{{-{dif_lower}}}^{{+{dif_upper}}}$"

        exponent = int(np.floor(np.log10(max_error)))

        if exponent <= -2:
            adjusted_exponent = int(np.floor(np.log10(abs(median))))
            scale = 10 ** adjusted_exponent

            median_scaled = median / scale
            dif_lower_scaled = abs(dif_lower) / scale
            dif_upper_scaled = abs(dif_upper) / scale

            # 有効数字に合わせて小数桁数を決定
            decimal_places = significant_digits - 1 - int(np.floor(np.log10(dif_upper_scaled)))
            decimal_places = max(decimal_places, 0)

            fmt_str = "{:." + str(decimal_places) + "f}"

            median_str = fmt_str.format(median_scaled)
            lower_str = fmt_str.format(dif_lower_scaled)
            upper_str = fmt_str.format(dif_upper_scaled)

            # 表示 (指数表記)
            error = rf"${median_str}_{{-{lower_str}}}^{{+{upper_str}}} \times 10^{{{adjusted_exponent}}}$"

        else:
            # exponent が -1 以上なら普通に表示
            decimal_places = significant_digits - 1 - int(np.floor(np.log10(abs(dif_upper))))
            decimal_places = max(decimal_places, 0)

            fmt_str = "{:." + str(decimal_places) + "f}"

            median_str = fmt_str.format(median)
            lower_str = fmt_str.format(abs(dif_lower))
            upper_str = fmt_str.format(abs(dif_upper))

            error = rf"${median_str}_{{-{lower_str}}}^{{+{upper_str}}}$"
        return error
        
    def format_custum_f_g(median, dif_lower, dif_upper):
        max_error = max(abs(dif_lower), abs(dif_upper))
        exponent = int(np.floor(np.log10(max_error)))
        if exponent <= -3:
            fmt = '.2g'
            fmt = "{{0:{0}}}".format(fmt).format
            string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            error = string_template.format(fmt(median), fmt(dif_lower), fmt(dif_upper))
        else:
            fmt = '.3f'
            fmt = "{{0:{0}}}".format(fmt).format
            string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            error = string_template.format(fmt(median), fmt(abs(dif_lower)), fmt(abs(dif_upper)))
        return error
    
    def format_value(num, precision=3):
        if abs(num) < 0.1 and num != 0:
            return f"{num:.{precision + 2}f}"
        else:
            return f"{num:.{precision}f}"
    
    """Plot kdeplot for lower triangle"""
    def kde_contour_plot(x, y, **kwargs):
        ax = plt.gca() #get current axes object
        if np.isnan(x).all() or np.isnan(y).all(): #nan value do not plotted
          return
        #sns.kdeplot(x=x, y=y, ax=ax, levels=[0.5, 1.0], fill=True, cut=0, alpha=0.9, **kwargs) #credible level 50
        #sns.kdeplot(x=x, y=y, ax=ax, levels=[0.1, 1.0], fill=True, cut=0, alpha=0.5, **kwargs) #credible level 90
        #sns.kdeplot(x=x, y=y, ax=ax, levels=[0.1, 0.5], cut=0, **kwargs)
    
        #The contour levels to draw. see https://corner.readthedocs.io/en/latest/api/#corner.hist2d(be careful the difference of "levels" in corner.py and seaborn)
        sns.kdeplot(x=x, y=y, ax=ax, levels=[np.exp(-9./2.), np.exp(-4./2.)], fill=True, cut=3, alpha=0.3, **kwargs) #3-sigma to 2-sigma
        sns.kdeplot(x=x, y=y, ax=ax, levels=[np.exp(-4./2.), np.exp(-1./2.)], fill=True, cut=3, alpha=0.6, **kwargs) #2-sigma to 1-sigma
        sns.kdeplot(x=x, y=y, ax=ax, levels=[np.exp(-1./2.), 1.0], fill=True, cut=2, alpha=0.9, **kwargs)
        sns.kdeplot(x=x, y=y, ax=ax, levels=[np.exp(-9. / 2.), np.exp(-4. / 2.), np.exp(-1. / 2.)], cut=2, **kwargs)
    
    def add_center_text(ax, text, hight, font_color):
        """add text to top of figure"""
        ax = ax
        x = 0.5
        y = hight
        ax.text(x, y, text, ha='center', va='center', fontsize=16, color=font_color, transform=ax.transAxes)
    
    def custom_kdeplot(*args, **kwargs):
        ax = plt.gca()
        data = args[0]
    
        color = kwargs.pop("color") #recieve color index
        
        sns.kdeplot(data,
                    ax=ax, #set axes
                    fill=False, #fill color
                    color=color, #color of plot
                    cut=3, #cut off probability
                    warn_singular=False, 
                    **kwargs)
    
        """get quantiles"""
        quantiles = (0.05, 0.95)
        quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
        quants = np.percentile(data, quants_to_compute * 100)
        lower_limit = quants[0]
        median = quants[1]  
        upper_limit = quants[2]
        dif_upper = upper_limit - median 
        dif_lower = median - lower_limit
    
        """add quantiles line"""
        ax.axvline(lower_limit, color=color, linestyle='--', lw=1.5)
        ax.axvline(upper_limit, color=color, linestyle='--', lw=1.5)
        
        """set text of error bar"""
        # fmt = '.2f'
        # fmt = "{{0:{0}}}".format(fmt).format
        # string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        # error = string_template.format(fmt(median), fmt(dif_lower), fmt(dif_upper))
        #ax.set_title(error)

        # error = format_error_latex_scaled(median, dif_lower, dif_upper)

        error = format_custum_f_g(median, dif_lower, dif_upper)
        
        add_center_text(ax, error, 1.2, 'black')
        
        #y_min, y_max = ax.get_ylim()
        #ax.set_ylim(0, y_max*1.3)
    
    
    def add_reference_lines_lower(x, y, **kwargs):
        ax = plt.gca()
        ref_values = kwargs.pop('ref_values') #get injection value
        x_ref, y_ref = ref_values.get(x.name, None), ref_values.get(y.name, None)
        """write injection value"""
        if x_ref is not None:
            ax.axvline(x=x_ref, color='red', linestyle=':', lw=2.3)
        if y_ref is not None:
            ax.axhline(y=y_ref, color='red', linestyle=':', lw=2.3)
    
    def add_reference_lines_diag(x, **kwargs):
        ax = plt.gca()
        ref_value = kwargs.pop('ref_value')
        x_ref = ref_value.get(x.name, None)
        print(x.name, x_ref)
        if ref_value is not None:
            ax.axvline(x=x_ref, color='red', linestyle=':', lw=2.3)
            
            # value = r'${}$'.format(np.round(x_ref, 3))
            value = r'${}$'.format(format_value(x_ref, precision=3))
            
            add_center_text(ax, value, 1.07, 'red')
    
    """Make PairGrid"""
    sns.set_style("whitegrid")
    sns.set_context("paper")
    palette = sns.color_palette("colorblind", n_colors=2)
    plot = sns.PairGrid(use_data, #data set
                        diag_sharey=False, #diagnal plot do not use different range
                        corner=True, #only oneside plot
                        hue='source', #use different color for each data
                        palette=palette, #color  
                        aspect=1.0)
    
    """add some plots"""
    plot = plot.map_lower(kde_contour_plot)
    plot.map_diag(custom_kdeplot,  lw=2, alpha=0.7)
    plot.map_lower(add_reference_lines_lower, ref_values=injection_parameters_dict)
    plot.map_diag(add_reference_lines_diag, ref_value=injection_parameters_dict) 
    
    """Add legend"""
    #handles = [Line2D([0], [0], color=palette[0], lw=4),
    #           Line2D([0], [0], color=palette[1], lw=4)]
    #labels = [ "ver1", "ver2" ]
    #plot.fig.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.7, 0.98), fontsize=20)
    
    """Adjust style""" 
    plot.fig.subplots_adjust(top=0.95)
    #plot.fig.suptitle('Your Title Here', fontsize=16)
    for ax in plot.axes.flatten(): #get latex label
        if ax is not None:
          label_x=ax.get_xlabel()
          label_y=ax.get_ylabel()
          ax.set_xlabel(latex_labels.get(label_x, label_x), fontsize=20)
          ax.set_ylabel(latex_labels.get(label_y, label_y), fontsize=20)
    
    """set file name of save fig"""
    def get_filename_without_extension(path):
        filename = os.path.basename(path)
        return os.path.splitext(filename)[0]
    
    if show_fig:
        plt.show()
        print('create posterior plot from: {}'.format(path_posterior_sample))

    else:
        filename = get_filename_without_extension(path_config) + '_posterior_corner_plot'
        save_filename = '{}'.format(filename) + '.pdf'
        plt.savefig('./' + path_outdir + '/' + '{}'.format(save_filename), bbox_inches='tight', pad_inches=0.2, dpi=200)
        plt.close()
        print('create posterior plot {}'.format(save_filename))
        bilby.core.utils.logger.info('create posterior plot {}'.format(save_filename))

if __name__ == "__main__":
    """recieve the argments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-1', '--config_path', help='path of config file')
    args = parser.parse_args()

    """get label and make paths"""
    filename = os.path.basename(args.config_path)
    label = os.path.splitext(filename)[0]
    path_posterior = f'./outdirs/outdir_{label}/Nested_sampler/posterior.dat'
    config_file_path = f'./outdirs/outdir_{label}/{label}.ini'
    path_outdir = f'outdirs/outdir_{label}/'
    plot_corner_pyring(path_posterior, config_file_path, path_outdir, show_fig=False)

    # label_list = ['pyring_shiftIm_to_220_dw0.1w1_snr100_DSparam',
    #               'pyring_shiftIm_to_220_dw0.01w1_snr100_DSparam',
    #               'pyring_shiftIm_to_220_dw0.001w1_snr100_DSparam',
    #               'pyring_shiftRe_to_220_dw0.1w1_snr100_DSparam',
    #               'pyring_shiftRe_to_220_dw0.01w1_snr100_DSparam',
    #               'pyring_shiftRe_to_220_dw0.001w1_snr100_DSparam',
    #               ]
    # for label in label_list:
    #     path_posterior = f'./outdirs/outdir_{label}/Nested_sampler/posterior.dat'
    #     config_file_path = f'./outdirs/outdir_{label}/{label}.ini'
    #     path_outdir = f'outdirs/outdir_{label}/'
    #     plot_corner_pyring(path_posterior, config_file_path, path_outdir, show_fig=False)