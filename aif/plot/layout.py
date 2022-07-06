"""
AIF - Artificial Intelligence for Finance
Copyright (C) 2022 Christian Liguda

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def update_layout(fig, title: str, y_axis_prefix: str = ''):
    """Get default layout for plotting"""
    layout = {
        'paper_bgcolor': '#faf7ec',
        'plot_bgcolor': '#faf7ec',
        'title': title,
        'title_font_size': 20,
        'title_x': 0.5,
        'margin': {'t': 70, 'b': 20, 'r': 20, 'l': 10},
        'showlegend': False,
        'xaxis': {
            'showgrid': False,
            'rangeslider': {
                'visible': False
            },
        },
        'yaxis': {
            'showgrid': False,
            'tickprefix': y_axis_prefix
        }
    }

    fig.update_layout(layout)

    return None
