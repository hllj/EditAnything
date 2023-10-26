import gradio as gr
import os

from editany import create_demo as create_demo_edit_anything
from sam2image import create_demo as create_demo_generate_anything
from editany_beauty import create_demo as create_demo_beauty
from editany_handsome import create_demo as create_demo_handsome
from editany_lora import EditAnythingLoraModel, init_sam_model, init_blip_processor, init_blip_model
from huggingface_hub import hf_hub_download, snapshot_download

DESCRIPTION = f'''# [Edit Anything](https://github.com/sail-sg/EditAnything)
**Edit anything and keep the layout by segmenting anything in the image.**
'''

sam_generator, mask_predictor = init_sam_model()
blip_processor = init_blip_processor()
blip_model = init_blip_model()

sd_models_path = snapshot_download("shgao/sdmodels")

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('🖌Edit Anything'):
            # model = EditAnythingLoraModel(base_model_path="stabilityai/stable-diffusion-2-inpainting",
            #                               controlmodel_name='LAION Pretrained(v0-4)-SD21',
            #                               lora_model_path=None, use_blip=True, extra_inpaint=False,
            #                               sam_generator=sam_generator,
            #                               mask_predictor=mask_predictor,
            #                               blip_processor=blip_processor,
            #                               blip_model=blip_model)
            # create_demo_edit_anything(model.process, model.process_image_click)
            model = EditAnythingLoraModel(base_model_path="runwayml/stable-diffusion-v1-5",
                                          controlmodel_name='LAION Pretrained(v0-4)-SD15',
                                          lora_model_path=None, use_blip=True, extra_inpaint=True,
                                          sam_generator=sam_generator,
                                          mask_predictor=mask_predictor,
                                          blip_processor=blip_processor,
                                          blip_model=blip_model)
            create_demo_edit_anything(model.process, model.process_image_click)

demo.queue(api_open=False).launch(server_name='0.0.0.0', share=False, debug=True)
