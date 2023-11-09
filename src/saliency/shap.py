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

        class_names = ['Neoplastic', 'Aphthous', 'Traumatic']

        os.makedirs('test_shap', exist_ok=True)
        model.eval()
        for batch_index, (images, _) in enumerate(test_loader):
            e = shap.GradientExplainer(model, images, batch_size=len(images))
            shap_values, indexes = e.shap_values(images, ranked_outputs=1, nsamples=64)
            shap_array = shap_values[0]
            for i in range(len(indexes)):
                # get the class name corresponding to the index
                class_name = class_names[indexes[i]]

                # overlay the saliency map on the image
                image_for_plot = images[i].permute(1, 2, 0).cpu().numpy()
                current_shap_value = shap_array[i]
                shap_image = current_shap_value[0]

                fig, ax = plt.subplots()
                ax.imshow(image_for_plot)

                ax.imshow((shap_image*255).astype('uint8'), cmap='jet', alpha=0.5)  # Overlay saliency map

                # save the figure with the overlaid saliency map
                plt.savefig(os.path.join('test_shap', f'saliency_map_batch_{batch_index}_image_number_{i}_class_{class_name}.png'), bbox_inches='tight')
                plt.close()




