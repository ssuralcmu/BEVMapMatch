# %%
import json
unique_name="grid_v8_WeightedBCELoss-lr0.0003-bs1-frac1.0-seed42"
json_file="results_"+unique_name+".json"
with open(json_file) as f:
    results = json.load(f)

# %%
#Load and display the images
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


# print(item["stitched_img_path"][0])
# print(item["basemap_img_path"][0])

for i in range(0, 10):
    item = results[i]
    basemap_img=Image.open(item["basemap_img_path"][0]).convert('RGB')
    basemap_img = np.array(basemap_img)
    basemap_img = np.flipud(basemap_img)
    cv2.imwrite("results/basemap_img_"+str(i)+".png", basemap_img)
    basemap_img = Image.fromarray(basemap_img)

    print("IOU Percentage for item "+str(i)+": "+str(item["iou_percentage"]))

    transform_base = transforms.Compose([
            transforms.Resize((1000, 1000)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    basemap_img = transform_base(basemap_img)
    #Convert to numpy array
    basemap_img = basemap_img.numpy()
    #Save image
    basemap_img = np.transpose(basemap_img, (1, 2, 0))
    basemap_img = (basemap_img * 255).astype(np.uint8)
    basemap_img = cv2.cvtColor(basemap_img, cv2.COLOR_RGB2BGR)

    metas_path=item["metas_path"][0]
    metas = np.load(metas_path, allow_pickle=True).item()

    center_x, center_y = basemap_img.shape[0] // 2, basemap_img.shape[1] // 2
    x_val = center_x - metas['perturbation'][0]
    y_val = center_y - metas['perturbation'][1]

    # print("center_x: ", center_x)
    # print("center_y: ", center_y)
    # print("x_val: ", x_val)
    # print("y_val: ", y_val)

    #Visualize the x_val and y_val on the basemap_img
    basemap_img = cv2.circle(basemap_img, (x_val, y_val), 10, (0, 0, 255), -1)
    basemap_img = cv2.putText(basemap_img, "x_val: "+str(x_val), (x_val, y_val-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imwrite("results/basemap_img_with_circle"+str(i)+".png", basemap_img)

    stitched_img=cv2.imread(item["stitched_img_path"][0])
    cv2.imwrite("stitched_img.png", stitched_img)

    #Visualize both the x_val, y_val and the predicted coordinates on the basemap_img
    predictions = item['predictions']
    #This is a 10x10 matrix with some values as 1 and some as 0. If the basemap of 1000x1000 is divided into 10x10, then we have to find the coordinates of the center of the ones on the matrix in the basemap image
    predictions = np.array(predictions)
    predictions = predictions.reshape(10,10)
    predictions = np.where(predictions == 1)
    predictions_x = predictions[0]
    predictions_y = predictions[1]
    #Convert to the coordinates in the basemap image
    predictions_x = predictions_x * (basemap_img.shape[0] // 10) + (basemap_img.shape[0] // 20)
    predictions_y = predictions_y * (basemap_img.shape[1] // 10) + (basemap_img.shape[1] // 20)
    #Visualize the predictions on the basemap_img
    for j in range(len(predictions_x)):
        basemap_img = cv2.circle(basemap_img, (predictions_x[j], predictions_y[j]), 10, (255, 0, 0), -1)
        basemap_img = cv2.putText(basemap_img, "Predicted", (predictions_x[j], predictions_y[j]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imwrite("results/basemap_img_with_predictions"+str(i)+".png", basemap_img)


# # %%
# avg_iou=0
# for item in results:
#     avg_iou+=item["iou_percentage"]

# avg_iou=avg_iou/len(results)

# print("Average IOU: ", avg_iou)

# # %%
# #Visualize the predictions and labels with matplotlib

# for i in range(30,40):
#     item=results[i]
#     print("IOU Percentage for item "+str(i)+": "+str(item["iou_percentage"]))
#     predictions = item["predictions"]
#     labels = item["labels"]
#     predictions = np.array(predictions)
#     labels = np.array(labels)
#     #Reshape to 10x10
#     predictions = predictions.reshape(10,10)
#     labels = labels.reshape(10,10)
#     plt.figure()
#     plt.title("Labels for item "+str(i))
#     plt.imshow(labels)
#     plt.savefig("results/labels_item_"+str(i)+".png")
#     plt.figure()
#     plt.title("Predictions for item "+str(i))
#     plt.imshow(predictions)
#     plt.savefig("results/predictions_item_"+str(i)+".png")


# # %%



