import torch.utils.data
import dash
import plotly.express as plt

from src.data.dataset import WaveDataset

if __name__ == "__main__":
    ds = WaveDataset("dataset/ISO Wr")
    # dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

    x, y = ds[0]

    app = dash.Dash()
    app.layout = [
        dash.dcc.Graph(figure=plt.line(y=x[:, 1])),
        dash.dcc.Graph(figure=plt.line(y=x[:, 5])),
        dash.dcc.Graph(figure=plt.line(y=x[:, 9])),
        dash.dcc.Graph(figure=plt.line(y=x[:, 13])),
    ]
    app.run(debug=True)
