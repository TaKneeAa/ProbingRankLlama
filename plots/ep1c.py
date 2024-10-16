import matplotlib.pyplot as plt

# Data arrays
layers = list(range(32))

tfidf_cosine_scores = [0.01, 0.02, 0.06, 0.10, 0.13, 0.17, 0.17, 0.19, 0.19, 0.18, 0.17, 0.16, 0.15, 0.20, 0.17, 0.15, 0.15, 0.16, 0.13, 0.16, 0.18, 0.16, 0.17, 0.16, 0.18, 0.16, 0.17, 0.18, 0.16, 0.14, 0.10, 0.03]
euclidean_scores = [0.06, 0.16, 0.18, 0.29, 0.39, 0.46, 0.51, 0.53, 0.53, 0.53, 0.55, 0.58, 0.57, 0.58, 0.58, 0.58, 0.57, 0.59, 0.58, 0.59, 0.57, 0.59, 0.59, 0.59, 0.60, 0.61, 0.60, 0.61, 0.59, 0.60, 0.58, 0.53]
manhattan_scores = [0.06, 0.14, 0.17, 0.27, 0.37, 0.44, 0.47, 0.50, 0.50, 0.50, 0.52, 0.55, 0.54, 0.55, 0.55, 0.54, 0.54, 0.56, 0.55, 0.56, 0.54, 0.56, 0.56, 0.56, 0.57, 0.57, 0.57, 0.57, 0.55, 0.56, 0.55, 0.48]
kl_divergence_scores = [0.03, 0.05, 0.13, 0.20, 0.28, 0.33, 0.38, 0.39, 0.39, 0.40, 0.39, 0.41, 0.44, 0.44, 0.42, 0.44, 0.42, 0.44, 0.42, 0.45, 0.44, 0.45, 0.46, 0.45, 0.45, 0.47, 0.47, 0.45, 0.44, 0.46, 0.44, 0.40]
js_divergence_scores = [0.05, 0.09, 0.15, 0.22, 0.31, 0.34, 0.37, 0.39, 0.38, 0.39, 0.39, 0.38, 0.41, 0.43, 0.41, 0.43, 0.40, 0.42, 0.42, 0.41, 0.43, 0.43, 0.44, 0.44, 0.42, 0.44, 0.44, 0.44, 0.43, 0.43, 0.40, 0.36]

# Plotting
plt.figure(figsize=(15, 4))  # width equal to page width, height 0.3 of page
plt.plot(layers, tfidf_cosine_scores, label='tfidf_cosine_scores', color='blue')
plt.plot(layers, euclidean_scores, label='euclidean_scores', color='green')
plt.plot(layers, manhattan_scores, label='manhattan_scores', color='teal')
plt.plot(layers, kl_divergence_scores, label='kl_divergence_scores', color='purple')
plt.plot(layers, js_divergence_scores, label='js_divergence_scores', color='orange')

plt.xlabel('Layers')
plt.ylabel('R^2 Scores')
plt.title('R^2 Scores Across RankLlama 7b Layers for Statistical similarity metrics')
plt.legend()
plt.grid(True)

plt.savefig('ep1c.png')
