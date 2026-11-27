"""A small learned model as an alternative to the hand-written tree.

It sees exactly what the tree sees, the answers of Apertus and the backdrop measurement, and learns from labeled
images how to combine them. One logistic regression per dimension.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import DIMENSIONS
from .data import is_scored


def inputs(answers, backdrops):
    """One 0/1 column per possible answer, plus the backdrop measurement."""
    columns = {}
    for key in answers.columns:
        values = sorted(answers[key].dropna().unique())
        if set(values) <= {"yes", "no"}:
            columns[key] = (answers[key] == "yes").astype(float)
        else:
            for value in values:
                columns[f"{key}={value}"] = (answers[key] == value).astype(float)
    table = pd.DataFrame(columns, index=answers.index)
    table["white"] = backdrops.loc[answers.index, "white"].astype(float)
    corners = np.array(list(backdrops.loc[answers.index, "corners"]), dtype=float) / 255
    for position in range(4):
        table[f"corner_{position}"] = corners[:, position]
    return table


class Combiner:
    def fit(self, answers, backdrops, rows):
        """rows: the labeled train rows of the images in answers. Only scored cells are used as examples."""
        table = inputs(answers, backdrops)
        labels = rows.set_index("image_id").loc[table.index]
        self.columns, self.mean, self.spread = table.columns, table.mean(), table.std() + 1e-9
        self.models = {}
        for dimension in DIMENSIONS:
            scored = is_scored(labels, dimension).to_numpy()
            model = LogisticRegression(C=0.5, max_iter=3000)
            self.models[dimension] = model.fit(((table - self.mean) / self.spread)[scored], labels[dimension][scored])
        return self

    def predict(self, answers, backdrops):
        table = inputs(answers, backdrops).reindex(columns=self.columns, fill_value=0.0)
        scaled = (table - self.mean) / self.spread
        predictions = pd.DataFrame({"image_id": table.index})
        for dimension in DIMENSIONS:
            predictions[dimension] = self.models[dimension].predict(scaled)
        return predictions
