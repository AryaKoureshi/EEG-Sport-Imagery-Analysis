import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import pandas as pd

# Load the EEGLAB .set file
path_file = r""
sub_name = ""
task_name = ""

subfolder_path = os.path.join(path_file, sub_name)

task_folder_path = os.path.join(subfolder_path, task_name)

if not os.path.exists(subfolder_path):
    os.makedirs(subfolder_path)

if not os.path.exists(task_folder_path):
    os.makedirs(task_folder_path)
    
eeglab_raw = mne.io.read_raw_eeglab(path_file + rf'\{sub_name}_{task_name}_cd.set', preload=True)

events, event_id = mne.events_from_annotations(eeglab_raw)
print("Event ID mapping:", event_id)

# Get integer IDs for labels '1', '2', '3', '4', '5', '6'
id_1 = event_id['1']
id_2 = event_id['2']
id_3 = event_id['3']
id_4 = event_id['4']
id_5 = event_id['5']
id_6 = event_id['6']

def extract_data_between_events(raw, events, start_id, end_id):
    start_indices = np.where(events[:, 2] == start_id)[0]
    end_indices = np.where(events[:, 2] == end_id)[0]

    data_segments = []

    for start_idx in start_indices:
        start_sample = events[start_idx, 0]

        next_end_indices = end_indices[end_indices > start_idx]
        if len(next_end_indices) == 0:
            continue
        end_idx = next_end_indices[0]
        end_sample = events[end_idx, 0]

        data, times = raw[:, int(start_sample):int(end_sample)]
        data_segments.append((data, times))

    return data_segments

rest_segments = extract_data_between_events(eeglab_raw, events, id_1, id_2)
task_segments = extract_data_between_events(eeglab_raw, events, id_3, id_4)
baseline_segments = extract_data_between_events(eeglab_raw, events, id_5, id_6)

def compute_and_plot_psd(data_segments, sfreq, condition_name):
    for idx, (data, times) in enumerate(data_segments):
        n_channels, n_times = data.shape

        psd_list = []
        freqs = None
        for ch in range(n_channels):
            f, Pxx = welch(data[ch, :], fs=sfreq, nperseg=512, return_onesided=True)
            psd_list.append(Pxx)
            if freqs is None:
                freqs = f

        psd_array = np.array(psd_list)

        plt.figure(figsize=(10, 6))
        for ch in range(n_channels):
            plt.semilogy(freqs, psd_array[ch, :], label=f'Channel {ch+1}')

        plt.title(f'Power Spectral Density - {condition_name.capitalize()} Trial {idx+1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (V^2/Hz)')
        plt.tight_layout()
        plt.xlim([1, 40])
        if condition_name=='baseline':
            filename=f'{condition_name}.png'
        else:
            filename = f'{condition_name}_trial{idx+1}.png'
        plt.savefig(path_file + rf"\{sub_name}\{task_name}\{filename}")
        plt.close()

        print(f'Saved PSD plot for {condition_name} trial {idx+1} as {filename}')

sfreq = eeglab_raw.info['sfreq']

compute_and_plot_psd(rest_segments, sfreq, 'rest')
compute_and_plot_psd(task_segments, sfreq, 'task')
compute_and_plot_psd(baseline_segments, sfreq, 'baseline')

eeglab_raw = mne.io.read_raw_eeglab(path_file + rf'\{sub_name}_{task_name}_cd.set', preload=True)

eeglab_raw.rename_channels({
    'Afz': 'AFz',
    'Af3': 'AF3',
    'Af4': 'AF4'
})

eeglab_raw.set_montage('standard_1020')

events, event_id = mne.events_from_annotations(eeglab_raw)
print("Event ID mapping:", event_id)

def extract_data_between_events(raw, events, start_id, end_id):
    start_indices = np.where(events[:, 2] == start_id)[0]
    end_indices = np.where(events[:, 2] == end_id)[0]

    data_segments = []

    for start_idx in start_indices:
        start_sample = events[start_idx, 0]

        # Find the next end event after this start event
        next_end_indices = end_indices[end_indices > start_idx]
        if len(next_end_indices) == 0:
            continue
        end_idx = next_end_indices[0]
        end_sample = events[end_idx, 0]

        data, times = raw[:, int(start_sample):int(end_sample)]
        data_segments.append((data, times))

    return data_segments

rest_segments = extract_data_between_events(eeglab_raw, events, id_1, id_2)
task_segments = extract_data_between_events(eeglab_raw, events, id_3, id_4)
baseline_segments = extract_data_between_events(eeglab_raw, events, id_5, id_6)

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'low_alpha': (8, 10),
    'high_alpha': (10, 12),
    'alpha': (8, 12),
    'SMR': (12, 15),
    'low_beta': (15, 23),
    'high_beta': (23, 30),
    'beta': (15, 30),
    'gamma': (30, 40)
}

band_powers = {'rest': {}, 'task': {}, 'baseline': {}}
for condition in ['rest', 'task', 'baseline']:
    for band in bands.keys():
        band_powers[condition][band] = []

def compute_and_collect_band_powers(data_segments, sfreq, condition_name):
    for idx, (data, times) in enumerate(data_segments):
        n_channels, n_times = data.shape

        psd_list = []
        freqs = None
        for ch in range(n_channels):
            f, Pxx = welch(data[ch, :], fs=sfreq, nperseg=512)
            psd_list.append(Pxx)
            if freqs is None:
                freqs = f

        psd_array = np.array(psd_list)

        for band_name, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.zeros(n_channels)
            for ch in range(n_channels):
                band_power[ch] = np.trapz(psd_array[ch, idx_band], freqs[idx_band])
            band_powers[condition_name][band_name].append(band_power)

sfreq = eeglab_raw.info['sfreq']
compute_and_collect_band_powers(rest_segments, sfreq, 'rest')
compute_and_collect_band_powers(task_segments, sfreq, 'task')
compute_and_collect_band_powers(baseline_segments, sfreq, 'baseline')

min_max_band_powers = {}
for band_name in bands.keys():
    all_values = []
    for condition in ['rest', 'task', 'baseline']:
        for trial_band_power in band_powers[condition][band_name]:
            all_values.extend(trial_band_power)
    min_value = np.min(all_values)
    max_value = np.max(all_values)
    min_max_band_powers[band_name] = (min_value, max_value)

def plot_topomaps(band_powers, min_max_band_powers, condition_name):
    info = eeglab_raw.info
    n_trials = len(band_powers[condition_name][list(bands.keys())[0]])
    for idx in range(n_trials):
        for band_name in bands.keys():
            band_power = band_powers[condition_name][band_name][idx]
            vmin, vmax = min_max_band_powers[band_name]
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 2, width_ratios=[10, 1])
            ax = fig.add_subplot(gs[0])
            cbar_ax = fig.add_subplot(gs[1])
            im, cn = mne.viz.plot_topomap(
                band_power, info, axes=ax, vlim=(vmin, vmax), show=False,
                names=eeglab_raw.ch_names, cmap='RdBu_r', extrapolate='head')
            fig.colorbar(im, cax=cbar_ax)
            ax.set_title(f'{condition_name.capitalize()} Trial {idx+1} - {band_name} band')
            if condition_name!='baseline':
                filename = f'{sub_name}_{condition_name}_trial{idx+1}_{band_name}_topomap.png'
            else:
                filename = f'{sub_name}_{condition_name}_{band_name}_topomap.png'
            save_dir = os.path.join(path_file, sub_name, f'{task_name}', 'topomaps')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, filename))
            plt.close()
            print(f'Saved topomap for {condition_name} trial {idx+1} band {band_name} as {filename}')

plot_topomaps(band_powers, min_max_band_powers, 'rest')
plot_topomaps(band_powers, min_max_band_powers, 'task')
plot_topomaps(band_powers, min_max_band_powers, 'baseline')

def save_band_powers_to_excel(band_powers, filename):
    band_dfs = {}

    for band_name in band_powers['baseline'].keys():
        data_rest = []
        data_task = []
        data_baseline = []

        data_baseline = band_powers['baseline'][band_name]

        # Rest condition has 5 trials
        data_rest = band_powers['rest'][band_name]  # Shape: (5 trials, 32 channels)

        # Task condition has 5 trials
        data_task = band_powers['task'][band_name]  # Shape: (5 trials, 32 channels)

        data_rest = np.array(data_rest).T  # Shape: (32 channels, 5 trials)
        data_task = np.array(data_task).T  # Shape: (32 channels, 5 trials)
        data_baseline = np.array(data_baseline).T  # Shape: (32 channels, 1 trial)

        columns_rest = [f"rest{i+1}" for i in range(data_rest.shape[1])]
        columns_task = [f"task{i+1}" for i in range(data_task.shape[1])]
        columns_baseline = ["baseline"]

        df_rest = pd.DataFrame(data_rest, index=eeglab_raw.ch_names, columns=columns_rest)
        df_task = pd.DataFrame(data_task, index=eeglab_raw.ch_names, columns=columns_task)
        df_baseline = pd.DataFrame(data_baseline, index=eeglab_raw.ch_names, columns=columns_baseline)

        band_df = pd.concat([df_baseline, df_rest, df_task], axis=1)

        band_dfs[band_name] = band_df

    with pd.ExcelWriter(filename) as writer:
        for band_name, band_df in band_dfs.items():
            band_df.to_excel(writer, sheet_name=band_name)

    print(f"Band powers saved to {filename}")

output_file = os.path.join(path_file, sub_name, f'{task_name}', f"{sub_name}_band_powers.xlsx")
save_band_powers_to_excel(band_powers, output_file)

