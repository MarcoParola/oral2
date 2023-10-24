from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import os
import hydra


class OralGradCam:

    def generate_saliency_maps_grad_cam(model, dataloader, predictions):
        # put in evaluation mode the model
        model.eval()
        # find last convolutional layer to compute gradients for grad-cam
        target_layers = OralGradCam.find_last_conv_layer(model)
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

        # iterate the dataloader passed
        for batch_index, (images, _) in enumerate(dataloader):
            for image_index, image in enumerate(images):
                # this is needed to work with a single image
                input_tensor = image.unsqueeze(0)

                # get the label predicted for current image
                predicted_class = predictions[batch_index * len(images) + image_index].item()

                # use the predicted label as target for grad-cam
                targets = [ClassifierOutputTarget(predicted_class)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(image.permute(1, 2, 0).numpy(), grayscale_cam, use_rgb=True)

                # put the generated map over the starting image
                visualization_image = Image.fromarray((visualization * 255).astype(np.uint8))
                # create the folder in which save the images
                os.makedirs(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps', exist_ok=True)
                visualization_image.save(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/grad_cam_maps/saliency_map_batch_{batch_index}_image_{image_index}.jpg')

                # TODO: write proper tests for grad-cam saliency maps
                # this is just to be sure that the corresponds to the ones with the overlapped map
                # if(image_index==0 and batch_index==0):
                #    image = image.permute(1, 2, 0).numpy()
                #    image = Image.fromarray((image * 255).astype(np.uint8))
                #    image.save("first_image_no_map.jpg")



    def find_last_conv_layer(model):
        layer_names = [name for name, module in model.model.named_children() if "layer" or "Fire" in name]
        last_layer_name = layer_names[-1]
        last_block = getattr(model.model, last_layer_name)[-1]
        target_layers = [last_block]
        return target_layers

