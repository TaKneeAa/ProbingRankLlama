# Plotting both LLaMA 7b and 13b top and bottom with shared x-axis
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
features = [
    "mean of term frequency", "variance of term frequency", "sum of stream length normalized term frequency",
    "min of stream length normalized term frequency", "max of stream length normalized term frequency",
    "mean of stream length normalized term frequency", "variance of stream length normalized term frequency",
    "sum of tf*idf", "min of tf*idf", "max of tf*idf", "mean of tf*idf", "variance of tf*idf", "BM25",
    "covered query term number", "covered query term ratio", "stream length", "sum of term frequency",
    "min of term frequency", "max of term frequency"]
    
scores_1 = [
    0.9814398692000113, 0.9827225102472532, -0.20267658022433643, -0.20267658022433643, 1.0,
    -0.08722717180620854, -0.07350124036966488, -0.13169483838810803, -0.5128205128205128, 1.0,
    -0.01654535703003357, 0.9767365709364091, 0.0615382321924155, -0.2581343641413578, 1.0,
    -0.24852501454069498, 0.24769941681865515, 0.9664357052590657, 0.2679892733263882
]

scores_2 = [
    0.9577654672114781, 0.9600422393440776, -0.19714606358405073, -0.19714606358405073, 1.0,
    -0.11592044620872777, -0.05237441374883156, -0.12346389619840847, -4.153846153846154, 1.0,
    -0.03047579633756281, 0.9639751615468145, 0.09662501577022764, -0.19415897621458655, 1.0,
    -0.20226484478508366, 0.2942559083715429, 0.9510836168191679, 0.30233250586004656
]

scores_in_distribution_2 = [
    0.9873177209764414, 0.9884152809022425, 0.9605641445153872, 0.9605641445153872, 1.0,
    0.7304693357842654, 0.9601249679481326, 0.8495724117418015, -0.10548523206751059, 1.0,
    0.5721609661226685, 0.9858575294333539, 0.7442391638876139, 0.8209893242549666, 1.0,
    0.25119589313315893, 0.894001710258226, 0.9850781558586451, 0.7784006805682487
]

scores_out_of_distribution_2 = [
    0.9650042503566775, 0.9652953285918466, -0.04833308770438349, -0.04833308770438349, 1.0,
    0.002522640460600911, 0.1071993468715533, 0.0029159853162085136, -0.1181434599156117, 1.0,
    0.03245750244693357, 0.9669106443426156, 0.1527181688270709, -0.04725232738647467, 1.0,
    -0.07626137742240524, 0.42648349976084676, 0.9626631303224855, 0.5124894198407112
]

# Plot for LLaMA 7b In-distribution and Out of distribution
axs[0].plot(features, scores_1, marker='o', linestyle='--', color='b', label='7b In-distribution dataset')
axs[0].plot(features, scores_2, marker='o', linestyle='--', color='r', label='7b Out of distribution dataset')
axs[0].set_ylabel("7b Score")
axs[0].set_title("LLaMA 7b: In distribution vs OOD query/documents")
axs[0].legend()
axs[0].grid(True)

# Plot for LLaMA 13b In-distribution and Out of distribution
axs[1].plot(features, scores_in_distribution_2, marker='o', linestyle='--', color='b', label='13b In-distribution dataset')
axs[1].plot(features, scores_out_of_distribution_2, marker='o', linestyle='--', color='r', label='13b Out of distribution dataset')
axs[1].set_ylabel("13b Score")
axs[1].set_xlabel("Feature")
axs[1].set_title("LLaMA 13b: In distribution vs OOD query/documents")
axs[1].legend()
axs[1].grid(True)

plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('ep4_comp.png')