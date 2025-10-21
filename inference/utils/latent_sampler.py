from scipy.spatial import ConvexHull
import numpy as np

def sampled_2d_latent_space(normalised_features_train, features_to_embed, num_samples=200):

    embeddings_train     = normalised_features_train[features_to_embed].to_numpy()

    hull = ConvexHull(embeddings_train)
    eqs = hull.equations

    hull_pts = embeddings_train[hull.vertices]
    hull_pts = np.vstack([hull_pts, hull_pts[0]])
    xmin, ymin = hull_pts.min(axis=0)
    xmax, ymax = hull_pts.max(axis=0)

    def inside_hull(x, y):
        return np.all(eqs[:,0]*x + eqs[:,1]*y + eqs[:,2] <= 0)

    samples = []
    while len(samples) < num_samples:
        x, y = np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)
        if inside_hull(x, y):
            samples.append((x, y))
    samples = np.array(samples)

    return samples