import numpy as np
import matplotlib.pyplot as plt

file = np.load('/Users/sajai/Downloads/data2/FWW_100_0_1_150_fast.npz')
print(file)
yields = file["yields"][:,:,1]- file["yields"][:,:,0]
kCDs = file["k12s"]
kDCs = file["k21s"]

# Create a meshgrid for plotting
KCD, KDC = np.meshgrid(kCDs, kDCs)

# Reshape yields if needed
if yields.shape != (len(kCDs), len(kDCs)):
    yields = yields.reshape(len(kCDs), len(kDCs))

# Create the contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(KCD, KDC, yields.T, levels=20, cmap="viridis")  
plt.colorbar(contour, label='Yield Value')  # Add a color bar
plt.xlabel('kCDs')
plt.ylabel('kDCs')
plt.title('Contour Plot of Compass Sensitivity Values')
plt.xscale('log')  # Log scale for kCDs
plt.yscale('log')  # Log scale for kDCs
plt.show()