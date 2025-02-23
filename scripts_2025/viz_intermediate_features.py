import torch
import matplotlib.pyplot as plt

stitched_map=torch.load('stitched_img.pth')
stitched_map=stitched_map.cpu().numpy()
#Visualize the stitched map
plt.imshow(stitched_map[0].transpose(1,2,0))
plt.show()
plt.savefig('stitched_map.png')

basemap=torch.load('basemap_img.pth')
basemap=basemap.cpu().numpy()
#Visualize the basemap
plt.imshow(basemap[0].transpose(1,2,0))
plt.show()
plt.savefig('basemap.png')