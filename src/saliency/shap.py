import shap
import numpy as np
import os
import matplotlib.pyplot as plt


class OralShap:

    def create_maps_shap(model, test_loader):

        for blocks in model.model.named_modules():
            for mod in blocks:
                if hasattr(mod, "inplace"):
                    # print(mod)
                    mod.inplace = False

        class_names = np.array(['Neoplastic', 'Aphthous', 'Traumatic'])

        os.makedirs('test_shap', exist_ok=True)
        model.eval()
        for batch_index, (images, _) in enumerate(test_loader):
            print(f"batch number {batch_index}")
            #for image_index, image in enumerate(images):
            e = shap.GradientExplainer(model, images, batch_size=64)
            shap_values, indexes = e.shap_values(images, ranked_outputs=1, nsamples=2)
            index_names = np.vectorize(lambda x: class_names[x])(indexes)
            # Iterate through the images and saliency maps
            i = 0
            for image, saliency_map, index in zip(images, shap_values, indexes):
                # Get the class name corresponding to the index
                class_name = class_names[index]

                # Overlay the saliency map on the image
                image_for_plot = image.permute(1, 2, 0).cpu().numpy()

                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)  # Convert image tensor to a NumPy array
                ax.imshow(saliency_map[0][0], cmap='jet', alpha=0.5)  # Overlay saliency map

                # Set axis properties (optional)
                ax.axis('off')
                ax.set_title(f'Class: {class_name}')

                # Save the figure with the overlaid saliency map
                plt.savefig(os.path.join('test_shap', f'saliency_map_{batch_index}_image_number_{i}_class{index}.png'), bbox_inches='tight')
                plt.close()
                i += 1




