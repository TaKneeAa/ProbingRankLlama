import matplotlib.pyplot as plt
import numpy as np

# Data for 7b model
tfidf_cosine_scores_7b = np.array([0.01, 0.02, 0.06, 0.10, 0.13, 0.17, 0.17, 0.19, 0.19, 0.18, 0.17, 0.16, 0.16, 0.20, 0.17, 0.15, 0.15, 0.16, 0.13, 0.16, 0.18, 0.16, 0.17, 0.16, 0.18, 0.16, 0.16, 0.18, 0.16, 0.14, 0.10, 0.03])
bert_cosine_scores_7b = np.array([0.08, 0.21, 0.23, 0.36, 0.48, 0.55, 0.60, 0.63, 0.63, 0.64, 0.66, 0.68, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.68, 0.69, 0.70, 0.71, 0.71, 0.71, 0.71, 0.71, 0.70, 0.70, 0.66, 0.61])
t5_cosine_scores_7b = np.array([0.09, 0.27, 0.27, 0.39, 0.51, 0.56, 0.60, 0.63, 0.63, 0.64, 0.64, 0.65, 0.67, 0.66, 0.65, 0.66, 0.66, 0.67, 0.67, 0.67, 0.67, 0.68, 0.67, 0.68, 0.69, 0.68, 0.68, 0.68, 0.66, 0.66, 0.66, 0.63])

# Data for 13b model
tfidf_cosine_scores_13b = np.array([0.04, 0.10, 0.14, 0.24, 0.32, 0.38, 0.43, 0.51, 0.51, 0.55, 0.58, 0.61, 0.60, 0.62, 0.60, 0.62, 0.59, 0.61, 0.61, 0.58, 0.60, 0.60, 0.57, 0.58, 0.55, 0.59, 0.52, 0.56, 0.55, 0.54, 0.56, 0.55, 0.54, 0.55, 0.56, 0.59, 0.60, 0.58, 0.56])
bert_cosine_scores_13b = np.array([0.12, 0.21, 0.31, 0.48, 0.56, 0.67, 0.77, 0.80, 0.80, 0.82, 0.84, 0.83, 0.82, 0.85, 0.83, 0.86, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.84, 0.84, 0.84, 0.84, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.83, 0.84])
t5_cosine_scores_13b = np.array([0.14, 0.27, 0.40, 0.57, 0.64, 0.72, 0.80, 0.83, 0.83, 0.85, 0.87, 0.87, 0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.87, 0.86, 0.87, 0.87, 0.87, 0.87, 0.87, 0.85, 0.85, 0.85, 0.85, 0.86, 0.86, 0.85, 0.86, 0.86, 0.86, 0.86, 0.85, 0.85])

# Set up the plot
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

# Plot 7b model
axs[0].plot(tfidf_cosine_scores_7b, label='tfidf_cosine_scores', color='purple', marker='o')
axs[0].plot(bert_cosine_scores_7b, label='bert_cosine_scores', color='green', marker='o')
axs[0].plot(t5_cosine_scores_7b, label='t5_cosine_scores', color='cyan', marker='o')
axs[0].axhline(y=1, color='grey', linestyle='--')
axs[0].set_title('Probing Rankllama 7b Model')
axs[0].set_xlabel('Rankllama Layer')
axs[0].set_ylabel('R^2 Score')
axs[0].legend()

# Plot 13b model
axs[1].plot(tfidf_cosine_scores_13b, label='tfidf_cosine_scores', color='purple')
axs[1].plot(bert_cosine_scores_13b, label='bert_cosine_scores', color='green')
axs[1].plot(t5_cosine_scores_13b, label='t5_cosine_scores', color='cyan')
axs[1].axhline(y=1, color='grey', linestyle='--')
axs[1].set_title('Probing Rankllama 13b Model')
axs[1].set_xlabel('Rankllama Layer')
axs[1].set_ylabel('R^2 Score')
axs[1].legend()

plt.tight_layout()
plt.savefig('ep1b.png')
