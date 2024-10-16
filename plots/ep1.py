import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import seaborn as sns

plt.clf()
plt.figure(figsize=(16, 10), dpi=192)

covered_query_term_number = [0.15, 0.38, 0.33, 0.49, 0.67, 0.77, 0.82, 0.85, 0.87, 0.89, 0.90, 0.91, 0.92, 0.93, 0.93, 0.93, 0.94, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.96, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98]
covered_query_term_ratio = [0.15, 0.41, 0.42, 0.57, 0.74, 0.81, 0.85, 0.88, 0.90, 0.91, 0.91, 0.93, 0.93, 0.94, 0.94, 0.94, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98]
stream_length = [0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00, 0.01, 0.01, 0.01, 0.00, -0.02, 0.00, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, -0.02, -0.01, -0.04, -0.10, -0.10]
sum_of_term_frequency = [0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00, 0.01, 0.01, 0.01, 0.00, -0.02, 0.00, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, -0.02, -0.01, -0.04, -0.10, -0.10]
bm25 = [0.05, 0.17, 0.18, 0.27, 0.39, 0.42, 0.45, 0.47, 0.47, 0.47, 0.47, 0.49, 0.47, 0.49, 0.50, 0.49, 0.50, 0.50, 0.50, 0.50, 0.50, 0.51, 0.51, 0.51, 0.52, 0.53, 0.53, 0.51, 0.52, 0.50, 0.47, 0.47]
min_of_term_frequency = [1.00] * 32
max_of_term_frequency = [0.02, 0.03, 0.04, 0.06, 0.07, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05, 0.05, 0.07, 0.05, 0.05, 0.05, 0.03, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.04, 0.03, 0.01, -0.04, -0.04]
mean_of_term_frequency = [0.06, 0.08, 0.13, 0.16, 0.18, 0.18, 0.18, 0.19, 0.19, 0.19, 0.18, 0.19, 0.18, 0.17, 0.18, 0.18, 0.18, 0.18, 0.18, 0.17, 0.18, 0.19, 0.19, 0.18, 0.19, 0.19, 0.18, 0.16, 0.17, 0.15, 0.10, 0.11]
variance_of_term_frequency = [0.03, 0.04, 0.07, 0.08, 0.09, 0.10, 0.09, 0.09, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.06, 0.08, 0.08, 0.09, 0.08, 0.09, 0.09, 0.09, 0.06, 0.07, 0.05, 0.00, 0.00]
sum_of_stream_length_normalized_term_frequency = [0.00] * 32
min_of_stream_length_normalized_term_frequency = [1.00] * 32
max_of_stream_length_normalized_term_frequency = [0.02, 0.03, 0.05, 0.07, 0.09, 0.09, 0.10, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.02, -0.07, -0.07]
mean_of_stream_length_normalized_term_frequency = [0.20, 0.33, 0.52, 0.65, 0.78, 0.83, 0.86, 0.88, 0.89, 0.89, 0.89, 0.90, 0.92, 0.92, 0.92, 0.93, 0.95, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98]
variance_of_stream_length_normalized_term_frequency = [0.05, 0.07, 0.14, 0.18, 0.22, 0.24, 0.26, 0.25, 0.25, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.22, 0.22, 0.22, 0.22, 0.21, 0.22, 0.22, 0.22, 0.23, 0.23, 0.23, 0.21, 0.22, 0.19, 0.12, 0.12]
sum_of_tf_idf = [0.02, 0.02, 0.03, 0.03, 0.04, 0.03, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, 0.00, -0.01, -0.00, -0.01, -0.02, -0.02, -0.02, -0.03, -0.01, -0.01, -0.01, -0.02, -0.01, -0.01, -0.02, -0.05, -0.03, -0.06, -0.13]
min_of_tf_idf = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
max_of_tf_idf = [0.02, 0.02, 0.03, 0.03, 0.04, 0.03, 0.04, 0.02, 0.02, 0.02, 0.01, 0.00, 0.00, -0.00, -0.00, -0.01, -0.01, -0.02, -0.03, -0.02, -0.02, -0.01, 0.00, -0.01, -0.01, -0.01, -0.01, -0.02, -0.03, -0.02, -0.06, -0.11]
mean_of_tf_idf = [0.12, 0.20, 0.29, 0.36, 0.42, 0.44, 0.45, 0.46, 0.46, 0.46, 0.46, 0.46, 0.47, 0.47, 0.47, 0.47, 0.48, 0.48, 0.47, 0.48, 0.47, 0.48, 0.48, 0.48, 0.48, 0.48, 0.48, 0.47, 0.47, 0.47, 0.47, 0.43]
variance_of_tf_idf = [0.20, 0.32, 0.52,0.64, 0.77, 0.83, 0.86, 0.88, 0.89, 0.89, 0.89, 0.90, 0.92, 0.92, 0.92, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98]
layers_7b = np.arange(1, 33)
plt.plot(layers_7b, covered_query_term_number, marker='o', label='covered_query_term_number')
plt.plot(layers_7b, covered_query_term_ratio, marker='o', label='covered_query_term_ratio')
plt.plot(layers_7b, stream_length, marker='o', label='stream_length')
plt.plot(layers_7b, sum_of_term_frequency, marker='o', label='sum_of_term_frequency')
plt.plot(layers_7b, bm25, marker='o', label='bm25')
plt.plot(layers_7b, min_of_term_frequency, marker='o', label='min_of_term_frequency')
plt.plot(layers_7b, max_of_term_frequency, marker='o', label='max_of_term_frequency')
plt.plot(layers_7b, mean_of_term_frequency, marker='o', label='mean_of_term_frequency')
plt.plot(layers_7b, variance_of_term_frequency, marker='o', label='variance_of_term_frequency')
plt.plot(layers_7b, sum_of_stream_length_normalized_term_frequency, marker='o', label='sum_of_stream_length_normalized_tf')
plt.plot(layers_7b, min_of_stream_length_normalized_term_frequency, marker='o', label='min_of_stream_length_normalized_tf')
plt.plot(layers_7b, max_of_stream_length_normalized_term_frequency, marker='o', label='max_of_stream_length_normalized_tf')
plt.plot(layers_7b, mean_of_stream_length_normalized_term_frequency, marker='o', label='mean_of_stream_length_normalized_tf')
plt.plot(layers_7b, variance_of_stream_length_normalized_term_frequency, marker='o', label='variance_of_stream_length_normalized_tf')
plt.plot(layers_7b, sum_of_tf_idf, marker='o', label='sum_of_tf_idf')
plt.plot(layers_7b, min_of_tf_idf, marker='o', label='min_of_tf_idf')
plt.plot(layers_7b, max_of_tf_idf, marker='o', label='max_of_tf_idf')
plt.plot(layers_7b, mean_of_tf_idf, marker='o', label='mean_of_tf_idf')
plt.plot(layers_7b, variance_of_tf_idf, marker='o', label='variance_of_tf_idf')


import matplotlib.colors as mcolors

# colors = sns.color_palette('husl', 20)
# colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS.keys())
cmap = cm.get_cmap('tab20', 20)  # 'tab20' is a colormap with 20 distinct colors
colors = [cmap(i) for i in range(20)]


sorted_lines = sorted(plt.gca().get_lines(), key=lambda line: line.get_ydata()[-1], reverse=True)
sorted_names = [line.get_label() for line in sorted_lines]
for i, (line, color) in enumerate(zip(sorted_lines, colors)):
    y_line_data = line.get_ydata()
    x_line_data = line.get_xdata()
    x_pos = x_line_data[-1]  # Position near the end of the line
    y_pos = y_line_data[-1]  # Position near the end of the line
    text_x_pos = x_pos + 0.25
    text_y_pos = y_pos
    if i in [0, 1, 2]:
        text_x_pos += 0.3 * i
    if i in [3, 4, 5, 6]:
        text_x_pos += 0.3 * (i - 3)
    if i in [10]:
        text_y_pos -= 0.02
    if i in [11, 12]:
        text_x_pos += 0.6 * (i - 11)
    if i in [15, 16]:
        text_x_pos += 0.6 * (i - 15)
    if i in [17, 18]:
        text_y_pos -= 0.02
    plt.text(text_x_pos, text_y_pos, f"{i+1}", fontsize=12, verticalalignment='center',
             horizontalalignment='left')  # You can change 'left' to 'center' or 'right'
    line.set_color(color)
sorted_names = [f"{i+1}: {name}" for i, name in enumerate(sorted_names)]
# plt.legend(sorted_lines, sorted_names, loc=(0.21, 0.57), fontsize=12, ncol=2)
plt.legend(sorted_lines, sorted_names, loc='best', fontsize=12)



plt.title('R² Scores when Probing for MSLR features in RankLlama 7b', fontsize=16)
plt.xlabel('Layer Number', fontsize=14)
plt.ylabel('R² Score', fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('ep1.png')