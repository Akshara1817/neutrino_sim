import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np
import awkward as ak
from pathlib import Path
import json

# Import your simulation logic
from ibd_simulation import load_spectrum, Spectrum
from fermi_event import generate_events as gen_fermi
from comparator import compute_statistical_delta, Event_Fermi, Event_Spectral

# -------------------------------------------------------------------------
# 1. Headless Data Processing (Awkward-Array Backend)
# -------------------------------------------------------------------------

def events_to_awkward(events: list) -> ak.Array:
    """Converts a list of dataclasses to a columnar awkward array."""
    return ak.Array([ev.__dict__ for ev in events])

def compute_spatial_vertices(ak_events: ak.Array, detector_radius_cm: float = 200.0) -> dict:
    """Vectorized calculation of prompt (e+) and delayed (n) detector flashes."""
    n_events = len(ak_events)
    if n_events == 0:
        return {}

    # Random spatial generation for prompt vertex
    r = detector_radius_cm * (np.random.uniform(0, 1, n_events) ** (1/3))
    costheta = np.random.uniform(-1, 1, n_events)
    phi = np.random.uniform(0, 2 * np.pi, n_events)
    sintheta = np.sqrt(1 - costheta**2)
    
    prompt_x = r * sintheta * np.cos(phi)
    prompt_y = r * sintheta * np.sin(phi)
    prompt_z = r * costheta

    # Delayed vertex calculation (Neutron drift)
    # Mean free path coeff: ~5.0 cm per MeV of kinetic energy
    drift_distance = ak_events.neutron_kinetic_energy_mev * 5.0 
    
    p_mag = np.sqrt(ak_events.neutron_px_mev_c**2 + ak_events.neutron_py_mev_c**2 + ak_events.neutron_pz_mev_c**2)
    
    # Avoid division by zero
    safe_p_mag = np.where(p_mag == 0, 1e-10, p_mag)
    dir_x = ak_events.neutron_px_mev_c / safe_p_mag
    dir_y = ak_events.neutron_py_mev_c / safe_p_mag
    dir_z = ak_events.neutron_pz_mev_c / safe_p_mag

    delayed_x = prompt_x + (dir_x * drift_distance)
    delayed_y = prompt_y + (dir_y * drift_distance)
    delayed_z = prompt_z + (dir_z * drift_distance)

    return {
        "prompt_x": prompt_x, "prompt_y": prompt_y, "prompt_z": prompt_z,
        "delayed_x": delayed_x, "delayed_y": delayed_y, "delayed_z": delayed_z,
        "delta_r": drift_distance
    }

# --- Initialization: Generate initial data ---
print("Initializing Background Physics Engine...")
spectrum_path = Path("tests/data/u235_spectrum.csv")
if spectrum_path.exists():
    spec = load_spectrum(spectrum_path)
else:
    energies = np.linspace(1.9, 9.0, 50)
    spec = Spectrum(energy_mev=energies, relative_flux=np.exp(-energies / 2.0))

TOTAL_EVENTS = 2000
raw_fermi = gen_fermi(spec, TOTAL_EVENTS, k_Fermi=225.0, E_binding=25.0, seed=42)
ak_fermi_base = events_to_awkward(raw_fermi)


# -------------------------------------------------------------------------
# 2. Dash UI Application
# -------------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'display': 'flex', 'height': '100vh'}, children=[
    
    # Left Sidebar: Controls & Stats
    html.Div(style={'width': '350px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRight': '1px solid #ddd'}, children=[
        html.H3("IBD Kinematic Cuts"),
        html.Hr(),
        
        # UI Control: Positron Kinetic Energy
        html.Label("Positron Kinetic Energy (MeV):"),
        dcc.RangeSlider(
            id='ke-slider',
            min=0.0, max=10.0, step=0.1,
            value=[1.5, 8.0],
            marks={0: '0', 2: '2', 4: '4', 6: '6', 8: '8', 10: '10'}
        ),
        html.Br(),

        # UI Control: Delta R (Fiducial Cut)
        html.Label("Max Correlation Distance ΔR (cm):"),
        dcc.Slider(
            id='delta-r-slider',
            min=10, max=300, step=10,
            value=150,
            marks={10: '10', 100: '100', 200: '200', 300: '300'}
        ),
        html.Br(),
        
        # Statistics Panel
        html.Div(id='stats-panel', style={'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'}),
        html.Br(),

        # Export Button
        html.Button("Export Sliced Data to Parquet", id="export-btn", n_clicks=0, 
                    style={'width': '100%', 'padding': '10px', 'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'cursor': 'pointer'}),
        html.Div(id='export-status', style={'marginTop': '10px', 'color': 'green', 'fontSize': '14px'})
    ]),

    # Right Main Area: 3D Visualization
    html.Div(style={'flexGrow': 1, 'padding': '0px'}, children=[
        dcc.Graph(id='3d-vertex-plot', style={'height': '100vh'})
    ])
])


# -------------------------------------------------------------------------
# 3. Callbacks: Tying UI to Awkward Array Logic
# -------------------------------------------------------------------------

@app.callback(
    [Output('3d-vertex-plot', 'figure'),
     Output('stats-panel', 'children')],
    [Input('ke-slider', 'value'),
     Input('delta-r-slider', 'value')]
)
def update_dashboard(ke_range, max_delta_r):
    # 1. Apply primary kinematic cuts via boolean Awkward masking
    ke_mask = (ak_fermi_base.positron_kinetic_energy_mev >= ke_range[0]) & \
              (ak_fermi_base.positron_kinetic_energy_mev <= ke_range[1])
    ak_sliced = ak_fermi_base[ke_mask]

    # 2. Compute physical vertices for the surviving events
    spatial_data = compute_spatial_vertices(ak_sliced)
    
    if not spatial_data:
        return go.Figure(), html.P("No events survived cuts.")

    # 3. Apply secondary spatial cut (ΔR threshold)
    dr_mask = spatial_data["delta_r"] <= max_delta_r
    final_prompt_x = spatial_data["prompt_x"][dr_mask]
    final_prompt_y = spatial_data["prompt_y"][dr_mask]
    final_prompt_z = spatial_data["prompt_z"][dr_mask]
    
    final_delayed_x = spatial_data["delayed_x"][dr_mask]
    final_delayed_y = spatial_data["delayed_y"][dr_mask]
    final_delayed_z = spatial_data["delayed_z"][dr_mask]

    # 4. Build 3D Plotly Figure
    fig = go.Figure()
    
    # Interleave coordinates for vectorized line drawing (avoids plotting thousands of individual traces)
    link_x = np.empty(len(final_prompt_x) * 3)
    link_x[0::3], link_x[1::3], link_x[2::3] = final_prompt_x, final_delayed_x, None
    link_y = np.empty(len(final_prompt_y) * 3)
    link_y[0::3], link_y[1::3], link_y[2::3] = final_prompt_y, final_delayed_y, None
    link_z = np.empty(len(final_prompt_z) * 3)
    link_z[0::3], link_z[1::3], link_z[2::3] = final_prompt_z, final_delayed_z, None

    fig.add_trace(go.Scatter3d(x=link_x, y=link_y, z=link_z, mode='lines', 
                               line=dict(color='gray', width=1, dash='dot'), name='Correlation (ΔR)'))
    fig.add_trace(go.Scatter3d(x=final_prompt_x, y=final_prompt_y, z=final_prompt_z, mode='markers', 
                               marker=dict(size=4, color='blue'), name='Prompt Flash (e+)'))
    fig.add_trace(go.Scatter3d(x=final_delayed_x, y=final_delayed_y, z=final_delayed_z, mode='markers', 
                               marker=dict(size=4, color='red'), name='Delayed Flash (n)'))

    fig.update_layout(
        title="IBD Dual-Vertex Spatial Distribution",
        scene=dict(xaxis_title='X (cm)', yaxis_title='Y (cm)', zaxis_title='Z (cm)', aspectmode='cube'),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # 5. Calculate live statistics for the panel
    retained = len(final_prompt_x)
    efficiency = (retained / TOTAL_EVENTS) * 100.0
    stats_html = [
        html.H5("Live Diagnostics"),
        html.P(f"Total Generated: {TOTAL_EVENTS}"),
        html.P(f"Retained Events: {retained}"),
        html.P(f"Cut Efficiency: {efficiency:.2f}%", style={'fontWeight': 'bold'})
    ]

    return fig, stats_html


@app.callback(
    Output('export-status', 'children'),
    Input('export-btn', 'n_clicks'),
    [State('ke-slider', 'value')]
)
def export_to_parquet(n_clicks, ke_range):
    if n_clicks > 0:
        # Re-apply the mask to get the exact data subset
        ke_mask = (ak_fermi_base.positron_kinetic_energy_mev >= ke_range[0]) & \
                  (ak_fermi_base.positron_kinetic_energy_mev <= ke_range[1])
        ak_sliced = ak_fermi_base[ke_mask]
        
        filename = "filtered_fermi_events.parquet"
        ak.to_parquet(ak_sliced, filename)
        return f"Exported {len(ak_sliced)} events to {filename}"
    return ""


if __name__ == '__main__':
    app.run(debug=True)