import plotly.graph_objects as go
from neutrino_sim.data_ingestionold import good_pseudojets, missing_p4


event_idx = 0


vis_px = good_pseudojets.px[event_idx]
vis_py = good_pseudojets.py[event_idx]
vis_pz = good_pseudojets.pz[event_idx]


miss_px = missing_p4.px[event_idx]
miss_py = missing_p4.py[event_idx]
miss_pz = missing_p4.pz[event_idx]

fig = go.Figure()


for i in range(len(vis_px)):
    fig.add_trace(go.Scatter3d(
        x=[0, vis_px[i]], y=[0, vis_py[i]], z=[0, vis_pz[i]],
        mode='lines+markers',
        line=dict(color='blue', width=4),
        marker=dict(size=4, symbol='diamond'),
        name=f'Visible Track {i}'
    ))


fig.add_trace(go.Scatter3d(
        x=[0, miss_px], y=[0, miss_py], z=[0, miss_pz],
        mode='lines+markers',
        line=dict(color='red', width=6, dash='dash'),
        marker=dict(size=6, symbol='cross'),
        name='Missing Momentum (MET)'
))


fig.update_layout(
    title=f"protoype fermi (Event {event_idx})",
    scene=dict(
        xaxis_title="Px (GeV)",
        yaxis_title="Py (GeV)",
        zaxis_title="Pz (GeV)",
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()