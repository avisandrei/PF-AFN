# import jsonpickle
import pickle
import base64
import io
import sys
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
from flask import Flask, request, Response, send_file
from pixellib.tune_bg import alter_bg

from data.base_dataset import *
from models.afwm import AFWM
from models.networks import ResUnetGenerator, load_checkpoint
from options.test_options import TestOptions

opt = TestOptions().parse()

warp_model = AFWM(opt, 3)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
gen_model.eval()
gen_model.cuda()
load_checkpoint(gen_model, opt.gen_checkpoint)

# Initialize the Flask application
app = Flask(__name__)


def encoded_image_bytes_to_tensor(image_bytes_base64, is_edge=False, is_input=False):
    image_bytes = base64.decodebytes(image_bytes_base64)
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)
    if is_input:
        change_bg = alter_bg(model_type="pb")
        change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
        image = change_bg.color_frame(image, colors=(255, 255, 255), detect="person")
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB' if not is_edge else 'L')
    params = get_params(opt, image.size)
    transform = get_transform(opt, params) if not is_edge else get_transform(opt, params, method=Image.NEAREST,
                                                                             normalize=False)
    return transform(image).unsqueeze(0)


# route http posts to this method
@app.route('/upload-image', methods=['POST'])
def upload_image():
    # pickle.dump(request.values, open('request.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    real_image = None if 'image' not in request.values else encoded_image_bytes_to_tensor(
        request.values['image'].encode(), is_input=True)
    clothes = None if 'cloth' not in request.values else encoded_image_bytes_to_tensor(
        request.values['cloth'].encode())
    edge = None if 'edge' not in request.values else encoded_image_bytes_to_tensor(
        request.values['edge'].encode(), is_edge=True)
    if not (real_image is None or clothes is None or edge is None):
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge
        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                                    mode='bilinear', padding_mode='zeros')
        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        needed = (p_tryon.squeeze().permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        needed = (needed * 255).astype(np.uint8)
        needed = cv2.cvtColor(needed, cv2.COLOR_RGB2BGR)
        _, image_array = cv2.imencode('.jpg', needed)
        return send_file(io.BytesIO(base64.encodebytes(image_array.tobytes())), mimetype='JPEG')
    return Response()


# start flask app
app.run(host="0.0.0.0", port=5000, threaded=True)
