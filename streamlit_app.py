import streamlit as st
import mne
import moabb
from moabb.datasets import utils
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(page_title="MOABB EEG Dataset Visualizer", layout="wide")

def get_standard_channel_list():
    """Return list of standard 10-20 system channels"""
    return [
        'Fp1', 'Fpz', 'Fp2',
        'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T8',
        'T5', 'P7', 'P3', 'Pz', 'P4', 'P8', 'T6',
        'O1', 'Oz', 'O2'
    ]

def display_topo_snapshot(raw, time_point):
    """Display topographic map at a single time point"""
    sample_idx = int(time_point * raw.info['sfreq'])
    data = raw.get_data(picks='eeg')[:, sample_idx]
    
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    im = mne.viz.plot_topomap(data, raw.info, axes=ax, show=False,
                             cmap='RdBu_r', sensors=True, contours=8,
                             outlines='head', res=128)
    plt.colorbar(im[0], ax=ax)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def standardize_channels(raw):
    """Standardize channels to only include standard 10-20 EEG channels"""
    standard_channels = get_standard_channel_list()
    current_channels = raw.ch_names
    channels_to_keep = []
    
    for ch in current_channels:
        if ch in standard_channels:
            channels_to_keep.append(ch)
        elif ch.upper().startswith(('STIM', 'STI', 'STATUS', 'MARKER')):
            channels_to_keep.append(ch)
    
    if channels_to_keep:
        raw = raw.pick_channels(channels_to_keep)
        logger.info(f"Kept channels: {channels_to_keep}")
    
    return raw

def find_interesting_segment(raw, window_size=1000):
    """Find a segment of data with more variation"""
    standard_channels = get_standard_channel_list()
    channels_to_analyze = [ch for ch in raw.ch_names if ch in standard_channels]
    if not channels_to_analyze:
        return 0
        
    data = raw.get_data(picks=channels_to_analyze)
    var_list = []
    for i in range(0, min(10000, data.shape[1] - window_size), window_size):
        var_list.append(np.var(data[:, i:i+window_size]))
    return int(np.argmax(var_list) * window_size)

def get_available_datasets() -> list:
    """Returns list of available MOABB dataset names"""
    return sorted([cls.__name__ for cls in utils.dataset_list])

def examine_raw_array(raw):
    """Extract key information from a RawArray object - analyze only standard channels"""
    standard_channels = get_standard_channel_list()
    eeg_channels = [ch for ch in raw.ch_names if ch in standard_channels]
    
    if not eeg_channels:
        return None
        
    data = raw.get_data(picks=eeg_channels)
    return {
        'channels': len(eeg_channels),
        'channel_names': eeg_channels,
        'duration_sec': raw.times[-1],
        'sampling_rate': raw.info['sfreq'],
        'n_samples': len(raw.times),
        'data_range': (np.percentile(data, 5), np.percentile(data, 95)),
        'std': np.std(data)
    }

def load_dataset(dataset_name: str):
    """Loads raw data from dataset without processing"""
    try:
        dataset_class = next(cls for cls in utils.dataset_list if cls.__name__ == dataset_name)
        dataset = dataset_class()
        
        st.write("### Dataset Information")
        st.markdown(f"""
        - **Dataset name:** {dataset.code}
        - **Number of subjects:** {len(dataset.subject_list)}
        - **Number of sessions:** {dataset.n_sessions}
        - **Time interval:** {dataset.interval if hasattr(dataset, 'interval') else 'Not specified'}
        - **Event IDs:** {dataset.event_id if hasattr(dataset, 'event_id') else 'Not specified'}
        """)
        
        first_subject = dataset.subject_list[0]
        st.write(f"\n### Loading data for subject {first_subject}")
        
        raw_data = dataset.get_data([first_subject])
        
        if isinstance(raw_data, dict):
            subject_data = raw_data[first_subject]
            if isinstance(subject_data, dict):
                first_session = list(subject_data.values())[0]
                if isinstance(first_session, dict):
                    first_session = list(first_session.values())[0]
                elif isinstance(first_session, list):
                    first_session = first_session[0]
            elif isinstance(subject_data, list):
                first_session = subject_data[0]
        elif isinstance(raw_data, list):
            first_session = raw_data[0]
            
        first_session = standardize_channels(first_session)
        return {'raw': raw_data, 'first_session': first_session}, dataset, None
        
    except Exception as e:
        logger.exception("Error loading dataset")
        return None, None, str(e)

def format_value(x):
    """Format values for display - scale to microvolts"""
    scaled = x * 1e6
    if abs(scaled) < 0.01:
        return f"{scaled:.2e}"
    else:
        return f"{scaled:.3f}"

def main():
    st.title("MOABB EEG Dataset Visualizer")
    
    selected_dataset = st.selectbox(
        "Choose a dataset",
        options=[""] + get_available_datasets(),
        format_func=lambda x: "Choose a dataset" if x == "" else x
    )
    
    if selected_dataset:
        with st.spinner(f"Loading {selected_dataset}..."):
            data_dict, dataset, error = load_dataset(selected_dataset)
            
        if error:
            st.error(f"Error loading dataset: {error}")
        elif data_dict is not None:
            st.success("Dataset loaded successfully!")
            
            first_session = data_dict['first_session']
            info = examine_raw_array(first_session)
            
            if info is None:
                st.warning("No standard 10-20 channels found in this dataset")
                return
                
            st.write("### Data Organization")
            st.markdown("""
            The data is organized in a hierarchical structure:
            1. Subject level (Dictionary with subject IDs as keys)
            2. Run/Session level (Dictionary with session IDs)
            3. Raw EEG data level (MNE Raw Array objects)
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Signal Information")
                st.markdown(f"""
                - Sampling rate: **{info['sampling_rate']} Hz**
                - Recording duration: **{info['duration_sec']:.2f} seconds**
                - Total samples: **{info['n_samples']:,}**
                - 5-95th percentile range: **{format_value(info['data_range'][0])} to {format_value(info['data_range'][1])} µV**
                - Standard deviation: **{format_value(info['std'])} µV**
                """)
            
            with col2:
                st.write("#### Channel Information")
                st.markdown(f"""
                - Number of standard channels: **{info['channels']}**
                - Channel names:
                ```
                {', '.join(info['channel_names'])}
                ```
                """)
            
            start_idx = find_interesting_segment(first_session)
            
            st.write("### Raw Data Preview")
            st.write(f"Showing data from {start_idx/info['sampling_rate']:.3f}s @ {info['sampling_rate']} Hz:")
            
            standard_channels = get_standard_channel_list()
            preview_channels = [ch for ch in first_session.ch_names if ch in standard_channels]
            preview_data = first_session.get_data(picks=preview_channels)[:, start_idx:start_idx+10]
            preview_data_uv = preview_data * 1e6
            
            time_points = [f"{t:.3f}s" for t in (np.arange(10) + start_idx) / info['sampling_rate']]
            preview_df = pd.DataFrame(
                preview_data_uv,
                columns=time_points,
                index=preview_channels
            )
            
            st.dataframe(
                preview_df.style.format("{:.3f}")
                         .background_gradient(cmap='RdYlBu', axis=1),
                use_container_width=True
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Value (µV)", f"{preview_data_uv.min():.3f}")
            with col2:
                st.metric("Max Value (µV)", f"{preview_data_uv.max():.3f}")
            with col3:
                st.metric("Mean Value (µV)", f"{preview_data_uv.mean():.3f}")
            
            st.info("""
            💡 Values are shown in microvolts (µV). 
            Color gradient shows relative voltage changes across time points.
            Only standard 10-20 system channels are shown.
            """)
            
            st.write("### EEG Topography")
            selected_time = st.slider(
                "Select time point for visualization",
                0.0,
                info['duration_sec'],
                0.0,
                step=0.1
            )

            topo_data = display_topo_snapshot(first_session, selected_time)
            st.markdown(f'<img src="data:image/png;base64,{topo_data}" alt="EEG Topography">', 
                      unsafe_allow_html=True)

if __name__ == "__main__":
    main()