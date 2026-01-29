import numpy as np
import cv2
import matplotlib.pyplot as plt

#filename='1531884111449198-ad3a4187e72e4a29b2281c2fe1adfc4d'
filename='1526915363947660-800e57a347b144cdaa5367ae87953e06'
#Load based on filename

metas=np.load('../all_train_metas_v3_with_angles/' + filename + '_metas.npy',allow_pickle=True).item()
print(metas)


basemap=cv2.imread('../all_train_basemaps_segmented_v3/' + filename + '_base_map_image.png',cv2.IMREAD_COLOR)
basemap=cv2.cvtColor(basemap,cv2.COLOR_BGR2RGB)
# basemap=np.flipud(basemap)

gen_map=cv2.imread('../all_train_maps_segmented_gt_v3/map/' + filename + '_generated_map_image.png',cv2.IMREAD_COLOR)
gen_map=cv2.cvtColor(gen_map,cv2.COLOR_BGR2RGB)

print(basemap.shape)

print(gen_map.shape)

print(metas['map_relative_yaw'])

#Print in degrees
print(np.degrees(metas['map_relative_yaw']))

#Create a figure with 2 subplots visualizing the basemap and the generated map 
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(basemap)
axs[0].set_title('Base Map')
axs[0].axis('off')
axs[1].imshow(gen_map)
axs[1].set_title('Generated Map')
axs[1].axis('off')

plt.savefig('comparison_map.png')