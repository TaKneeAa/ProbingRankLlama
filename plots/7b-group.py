# Updating the plot to correctly label the second graph as RankLlama 13b instead of 14b
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(16, 6))

# RankLlama 7b
plt.subplot(2, 1, 1)
plt.plot(layers, abc_scores, marker='o', label='(a+b+c)', color='blue')
plt.plot(layers, abc_squared_scores, marker='o', label='(a+b+c)^2', color='green')
plt.plot(layers, abc_cubed_scores, marker='o', label='(a+b+c)^3', color='cyan')
plt.plot(layers, BM25_scores, marker='o', label='BM25', color='orange')
plt.title('R² Scores Across Layers for RankLlama 7b', fontsize=16)
plt.xlabel('Layer Number', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True)

# RankLlama 13b (previously labeled as 14b)
plt.subplot(2, 1, 2)
plt.plot(layers_14b, abc_scores_14b, marker='o', label='(a+b+c)', color='blue')
plt.plot(layers_14b, abc_squared_scores_14b, marker='o', label='(a+b+c)^2', color='green')
plt.plot(layers_14b, abc_cubed_scores_14b, marker='o', label='(a+b+c)^3', color='cyan')
plt.plot(layers_14b, BM25_scores_14b, marker='o', label='BM25', color='orange')
plt.title('R² Scores Across Layers for RankLlama 13b', fontsize=16)
plt.xlabel('Layer Number', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True)

# Adjust layout to reduce height and increase width
plt.tight_layout()
plt.savefig()